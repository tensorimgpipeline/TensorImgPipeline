# Random selected images for final result
# Training: [tensor([28]), tensor([46]), tensor([60]), tensor([63]), tensor([90])]
# Validation: [tensor([10]), tensor([18]), tensor([35]), tensor([57]), tensor([79]]]
# Test: [tensor([25]), tensor([41]), tensor([49]), tensor([61]), tensor([95])]

# What do we want to Plot?
# 1. Training and Validation Images every 1/10*total_epochs epoch.
# 2. Training, Validation and Test Images at the end.
# How do we want to plot it?
# 1. 5xX grid with image alone, image with overlapping mask, mask alone,
#   mask against GT if available, Difference of Mask with GT if available.

from collections import OrderedDict
from typing import Any, Protocol

import torch
import wandb

from tipi.abstractions import PipelineProcess


class ProgressTaskCallable(Protocol):
    """Protocol for functions decorated with progress_task."""

    def __call__(self, total: int) -> None: ...


class ResultProcess(PipelineProcess):
    """
    ResultProcess is a class that handles the logging of images and their corresponding masks during
    the training, validation, and testing phases of a machine learning pipeline.

    Attributes:
        progress_manager: An instance of the progress manager obtained from the manager.
        total_epochs: The total number of epochs configured in wandb.
        train_images_indices: A list of indices for the training images.
        val_images_indices: A list of indices for the validation images.
        test_images_indices: A list of indices for the test images.
        datasets: The datasets containing training, validation, and test sets.
        trainset: The training dataset.
        valset: The validation dataset.
        testset: The test dataset.
    Methods:
        execute():
            Executes the logging process for train, validation, and test images.
        _get_log_train_images() -> callable:
            Returns a function that logs the training images.
        _get_log_val_images() -> callable:
            Returns a function that logs the validation images.
        _get_log_test_images() -> callable:
            Returns a function that logs the test images.
        _log_image(image, mask, pred_mask, dataset, idx):
            Logs the image, predicted mask, and mask difference to wandb.
        skip() -> bool:
            Returns False indicating that this process should not be skipped.
    """

    def __init__(self, controller: Any, force: bool, selected_images: dict[str, list[int]]):
        """Initialize ResultProcess.

        Args:
            controller: Controller/manager providing access to permanences.
            force: Whether to force execution.
            selected_images: Dictionary mapping dataset types to image indices.
        """
        super().__init__(controller, force)
        self.progress_manager = controller.get_permanence("progress_manager")

        self.device = controller.get_permanence("device").device
        self.model = controller.get_permanence("network").model_instance
        self.model.to(self.device)

        self.train_images_indices = selected_images.get("train")
        self.val_images_indices = selected_images.get("val")
        self.test_images_indices = selected_images.get("test")

        self.datasets = controller.get_permanence("data")
        self.trainset = self.datasets.segnet_dataset_train
        self.valset = self.datasets.segnet_dataset_val
        self.testset = self.datasets.segnet_dataset_test

    def execute(self) -> None:
        train_image_log = self._get_log_train_images()
        val_image_log = self._get_log_val_images()
        test_image_log = self._get_log_test_images()

        if self.train_images_indices:
            train_image_log(len(self.train_images_indices))
        if self.datasets.val_available() and self.val_images_indices:
            val_image_log(len(self.val_images_indices))
        if self.datasets.test_available() and self.test_images_indices:
            test_image_log(len(self.test_images_indices))

    def _get_log_train_images(self) -> ProgressTaskCallable:
        @self.progress_manager.progress_task("result", visible=False)
        def _inner_log_image(total: int, task_id: int, progress: Any) -> None:
            image_stack = []
            mask_stack = []
            pred_mask_stack = []
            for idx in range(total):
                progress.advance(task_id)
                selected_images = self.train_images_indices[idx]  # type: ignore[index]
                image, mask = self.trainset[selected_images]
                image_stack.append(image)
                mask_stack.append(mask)
                pred_mask = self._get_pred_mask(self._inference_model(image))
                pred_mask_stack.append(pred_mask)
            images = torch.cat(image_stack, 2)
            masks = torch.cat(mask_stack, 1)
            pred_masks = torch.cat(pred_mask_stack, 1)
            self._log_image(images, masks, pred_masks, "train")

        return _inner_log_image  # type: ignore[no-any-return]

    def _get_log_val_images(self) -> ProgressTaskCallable:
        @self.progress_manager.progress_task("result", visible=False)
        def _inner_log_image(total: int, task_id: int, progress: Any) -> None:
            image_stack = []
            mask_stack = []
            pred_mask_stack = []
            for idx in range(total):
                progress.advance(task_id)
                selected_images = self.val_images_indices[idx]  # type: ignore[index]
                image, mask = self.valset[selected_images]
                image_stack.append(image)
                mask_stack.append(mask)
                pred_mask = self._get_pred_mask(self._inference_model(image))
                pred_mask_stack.append(pred_mask)
            images = torch.cat(image_stack, 2)
            masks = torch.cat(mask_stack, 1)
            pred_masks = torch.cat(pred_mask_stack, 1)
            self._log_image(images, masks, pred_masks, "val")

        return _inner_log_image  # type: ignore[no-any-return]

    def _get_log_test_images(self) -> ProgressTaskCallable:
        @self.progress_manager.progress_task("result", visible=False)
        def _inner_log_image(total: int, task_id: int, progress: Any) -> None:
            image_stack = []
            mask_stack = []
            pred_mask_stack = []
            for idx in range(total):
                progress.advance(task_id)
                selected_images = self.test_images_indices[idx]  # type: ignore[index]
                image, mask = self.testset[selected_images]
                image_stack.append(image)
                mask_stack.append(mask)
                pred_mask = self._get_pred_mask(self._inference_model(image))
                pred_mask_stack.append(pred_mask)
            images = torch.cat(image_stack, 2)
            masks = torch.cat(mask_stack, 1)
            pred_masks = torch.cat(pred_mask_stack, 1)
            self._log_image(images, masks, pred_masks, "test")

        return _inner_log_image  # type: ignore[no-any-return]

    def _inference_model(self, image: Any) -> Any:
        self.model.eval()
        with torch.no_grad():
            return self.model(image.unsqueeze(0).to(self.device))

    def _get_pred_mask(self, pred: Any) -> Any:
        if isinstance(pred, OrderedDict):
            pred = pred["out"]
        return pred.argmax(dim=1).squeeze(0).cpu()

    def _get_mask_difference(self, mask: Any, pred_mask: Any) -> Any:
        mask[mask == 255] = 0
        mask_difference = mask - pred_mask
        if mask_difference.min() < 0:
            mask_difference = mask_difference + mask_difference.min().abs()
        return mask_difference.to(torch.uint8)

    def _log_image(self, image: Any, mask: Any, pred_mask: Any, dataset: str) -> None:
        class_labels = self._swap_labels(self.datasets.data_container.classes)
        just_image = wandb.Image(image, caption=f"{dataset} images")
        image_with_mask = wandb.Image(
            image,
            masks={"ground_truth": {"mask_data": mask.numpy(), "class_labels": class_labels}},
            caption=f"{dataset} images with mask",
        )
        image_with_pred = wandb.Image(
            image,
            masks={"predictions": {"mask_data": pred_mask.numpy(), "class_labels": class_labels}},
            caption=f"{dataset} images with predictions",
        )
        mask_difference = wandb.Image(self._get_mask_difference(mask, pred_mask), caption=f"{dataset} mask difference")
        wandb.log({f"{dataset}_images": just_image})
        wandb.log({f"{dataset}_images_with_pred": image_with_pred})
        wandb.log({f"{dataset}_images_with_mask": image_with_mask})
        wandb.log({f"{dataset}_mask_difference": mask_difference})

    def _swap_labels(self, labels: dict[Any, Any]) -> dict[Any, Any]:
        return {v: k for k, v in labels.items()}

    def skip(self) -> bool:
        return False
