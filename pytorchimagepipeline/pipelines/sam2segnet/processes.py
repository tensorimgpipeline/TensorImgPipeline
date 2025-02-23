from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torchvision
from segment_anything import SamPredictor, sam_model_registry

import wandb
import wandb.sdk as wandb_sdk
from pytorchimagepipeline.abstractions import PipelineProcess
from pytorchimagepipeline.pipelines.sam2segnet.utils import get_palette

if TYPE_CHECKING:
    from pytorchimagepipeline.pipelines.sam2segnet.observer import Sam2SegnetObserver


class PredictMasks(PipelineProcess):
    def __init__(self, observer: Sam2SegnetObserver, force: bool) -> None:
        super().__init__(observer, force)
        sam_checkpoint = Path("data/models/sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.device = observer.device
        self.dataset = observer.data.sam_dataset
        self.mask_creator = observer.mask_creator
        if hasattr(observer, "progress"):
            self.progress_manager = observer.progress

    def execute(self) -> None:
        self.sam.to(self.device)
        predictor = SamPredictor(self.sam)

        for data in self.dataset:
            image, bboxes, bbox_classes, filestem = data
            bbox_classes_idx = torch.tensor(
                [self.dataset.class_idx[class_] for class_ in bbox_classes],
                dtype=torch.float32,
                device=self.device,
            )

            predictor.set_image(image)

            if bboxes:
                bboxes_tensor = torch.stack([bbox.to_tensor(device=predictor.device) for bbox in bboxes])
                transformed_boxes = predictor.transform.apply_boxes_torch(bboxes_tensor, image.shape[:2])
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

                masks_with_classes = masks * bbox_classes_idx.view(-1, 1, 1, 1)
                merged_mask = self.mask_creator.create_mask(masks_with_classes)
            else:
                merged_mask = torch.zeros(image.shape[:2], dtype=torch.uint8, device=self.device)

            palette = get_palette()
            mask_as_pil = torchvision.transforms.functional.to_pil_image(merged_mask)
            mask_as_pil.putpalette(palette)
            mask_path = self.dataset.target_location / f"{filestem}.png"
            mask_as_pil.save(mask_path)

    def skip(self) -> bool:
        return self.dataset.all_created() and not self.force


class TrainModel(PipelineProcess):
    def __init__(self, observer: Sam2SegnetObserver, force: bool) -> None:
        super().__init__(observer, force)
        if hasattr(observer, "progress"):
            self.progress_manager = observer.progress
        self.device = observer.device
        self.model = observer.network.model_instance
        self.model.to(self.device)

        # Hyperparameters
        self.hyperparams: dict[str, Any] | wandb_sdk.wandb_config.Config
        if hasattr(observer, "wandb"):
            self.wandb_logger = observer.wandb
            self.wandb_logger.global_step = 0
            self.hyperparams = wandb.config
        else:
            self.hyperparams = observer.hyperparams.hyperparams

        # Data
        self.datasets = observer.data
        batch_size = self.hyperparams.get("batch_size", 20)
        trainset = self.datasets.segnet_dataset_train
        valset = self.datasets.segnet_dataset_val
        testset = self.datasets.segnet_dataset_test

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # Training components
        components = observer.training_components
        ignore_index = self.datasets.data_container.ignore
        self.criterion = components.Criterion(**self.hyperparams.get("criterion", {}), ignore_index=ignore_index)
        self.optimizer = components.Optimizer(self.model.parameters(), **self.hyperparams.get("optimizer", {}))
        self.scheduler = components.Scheduler(
            self.optimizer, **self.hyperparams.get("scheduler", components.scheduler_params)
        )
        self.num_epochs = self.hyperparams.get("num_epochs", 20)

    def skip(self) -> bool:
        return False

    def execute(self) -> None:
        _epoch_step = self.get_epoch_step()
        _epoch_step(self.num_epochs)

    def get_train_step(self) -> Callable[[], float]:
        def _train_step() -> float:
            self.model.train()
            running_loss = 0.0
            for data in self.train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(inputs)
                main_pred, aux_pred = output.get("out"), output.get("aux", None)
                main_loss = self.criterion(main_pred, labels)
                aux_loss = self.criterion(aux_pred, labels) if aux_pred is not None else torch.zeros_like(main_loss)
                loss = main_loss + self.hyperparams.get("aux_lambda", 0.4) * aux_loss
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if hasattr(self, "wandb_logger"):
                    self.wandb_logger.log_metrics({"train_loss": loss.item()})
            self.scheduler.step()
            if hasattr(self, "wandb_logger"):
                self.wandb_logger.log_metrics({"epoch_loss": running_loss / len(self.train_loader)})
            return running_loss

        return _train_step

    def get_validate_step(self) -> Callable[[], float]:
        def _validate_step() -> float:
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in self.val_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
            if hasattr(self, "wandb_logger"):
                self.wandb_logger.log_metrics({"val_loss": val_loss / len(self.val_loader)})
            return val_loss

        return _validate_step

    def get_test_step(self) -> Callable[[], float]:
        def _test_step() -> float:
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for data in self.test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()
            if hasattr(self, "wandb_logger"):
                self.wandb_logger.log_metrics({"test_loss": test_loss / len(self.test_loader)})
            return test_loss

        return _test_step

    def get_epoch_step(self) -> Callable[[int], None]:
        _train_step = self.get_train_step()
        _validate_step = self.get_validate_step()
        _test_step = self.get_test_step()

        def _epoch_step(total: int) -> None:
            for _ in range(total):
                _train_step()

                if self.datasets.val_available():
                    _validate_step()

            if self.datasets.test_available():
                _test_step()

        return _epoch_step
