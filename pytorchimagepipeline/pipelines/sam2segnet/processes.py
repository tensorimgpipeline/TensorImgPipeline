from pathlib import Path

import torch
import torchvision
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

import wandb
from pytorchimagepipeline.abstractions import AbstractObserver, PipelineProcess
from pytorchimagepipeline.pipelines.sam2segnet.utils import get_palette


class PredictMasks(PipelineProcess):
    def __init__(self, observer: AbstractObserver, force: bool) -> None:
        super().__init__(observer, force)
        sam_checkpoint = Path("data/models/sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.device = observer.get_permanence("device").device
        self.dataset = observer.get_permanence("data").sam_dataset
        self.mask_creator = observer.get_permanence("mask_creator")
        self.progress_manager = observer.get_permanence("progress_manager")

    def execute(self):
        self.sam.to(self.device)
        predictor = SamPredictor(self.sam)

        bar = tqdm(len(self.dataset), desc="Predicting masks")
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

            bar.update(1)
            bar.refresh()

    def skip(self):
        return self.dataset.all_created() and not self.force


class TrainModel(PipelineProcess):
    def __init__(self, observer, force):
        super().__init__(observer, force)
        self.progress_manager = observer.get_permanence("progress_manager")
        self.device = observer.get_permanence("device").device
        self.model = observer.get_permanence("network").model_instance
        self.model.to(self.device)

        # Hyperparameters
        self.wandb_logger = observer.get_permanence("wandb_logger")
        self.wandb_logger.global_step = 0
        self.hyperparams = wandb.config

        # Data
        self.datasets = observer.get_permanence("data")
        batch_size = self.hyperparams.get("batch_size", 20)
        trainset = self.datasets.segnet_dataset_train
        valset = self.datasets.segnet_dataset_val
        testset = self.datasets.segnet_dataset_test

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # Training components
        components = observer.get_permanence("training_components")
        ignore_index = self.datasets.data_container.ignore
        self.criterion = components.Criterion(**self.hyperparams.get("criterion", {}), ignore_index=ignore_index)
        self.optimizer = components.Optimizer(self.model.parameters(), **self.hyperparams.get("optimizer", {}))
        self.scheduler = components.Scheduler(
            self.optimizer, **self.hyperparams.get("scheduler", components.scheduler_params)
        )
        self.num_epochs = self.hyperparams.get("num_epochs", 20)

    def skip(self):
        return False

    def execute(self):
        _epoch_step = self.get_epoch_step()
        _epoch_step(self.num_epochs)

    def get_train_step(self) -> callable:
        @self.progress_manager.progress_task("train", visible=False)
        def _train_step(train_id, total, progress):
            self.model.train()
            running_loss = 0.0
            for idx, data in enumerate(self.train_loader):
                progress.advance(train_id)
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
                self.wandb_logger.log_metrics({"train_loss": loss.item()})
                progress.update(train_id, status=f"Loss: {running_loss / (idx + 1)}")
            self.scheduler.step()
            self.wandb_logger.log_metrics({"epoch_loss": running_loss / len(self.train_loader)})
            return running_loss

        return _train_step

    def get_validate_step(self) -> callable:
        @self.progress_manager.progress_task("val", visible=False)
        def _validate_step(val_id, total, progress):
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx, data in enumerate(self.val_loader):
                    progress.advance(val_id)
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    progress.update(val_id, status=f"Mean Val Loss: {val_loss / (idx + 1)}")
            self.wandb_logger.log_metrics({"val_loss": val_loss / len(self.val_loader)})
            return val_loss

        return _validate_step

    def get_test_step(self) -> callable:
        @self.progress_manager.progress_task("test", visible=False)
        def _test_step(test_id, total, progress):
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for idx, data in enumerate(self.test_loader):
                    progress.advance(test_id)
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()
                    progress.update(test_id, status=f"Loss: {test_loss / (idx + 1)}")
            self.wandb_logger.log_metrics({"test_loss": test_loss / len(self.test_loader)})
            return test_loss

        return _test_step

    def get_epoch_step(self) -> callable:
        _train_step = self.get_train_step()
        _validate_step = self.get_validate_step()
        _test_step = self.get_test_step()

        @self.progress_manager.progress_task("epoch", visible=False)
        def _epoch_step(epoch_id, total, progress):
            num_train = len(self.train_loader)
            num_val = len(self.val_loader)
            num_test = len(self.test_loader)
            for epoch in range(total):
                progress.advance(epoch_id)
                running_loss = _train_step(num_train)

                if self.datasets.val_available():
                    val_loss = _validate_step(num_val)
                status = f"Epoch: {epoch + 1}, Loss: {running_loss / num_train}"
                progress.update(epoch_id, status=status)

            if self.datasets.test_available():
                test_loss = _test_step(num_test)
                status = f"Epoch: {epoch + 1}, Loss: {running_loss / num_train}, Val Loss: {val_loss / num_val}, Test Loss: {test_loss / num_test}"
                progress.update(epoch_id, status=status)

        return _epoch_step
