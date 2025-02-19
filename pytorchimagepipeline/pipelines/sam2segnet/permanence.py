from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass, field
from logging import warning
from pathlib import Path

try:
    from tomllib import load as toml_load  # type ignore[import-not-found]
except ImportError:
    try:
        from tomli import load as toml_load  # type: ignore  # noqa: PGH003 # type: ignore[unused-import]
    except ImportError:
        sys.exit("Error: This program requires either tomllib or tomli but neither is available")

import cv2
import torch
import torch.nn.functional as F
import torchvision
import torchvision.models.segmentation as segmentation
import torchvision.transforms.v2 as transforms
from torchvision.datasets import VisionDataset
from torchvision.io import decode_image
from torchvision.tv_tensors import Mask

from pytorchimagepipeline.abstractions import Permanence
from pytorchimagepipeline.core.permanences import ProgressManager
from pytorchimagepipeline.pipelines.sam2segnet.errors import (
    FormatNotSupportedError,
    MaskNotAvailable,
    MaskShapeError,
    ModeError,
    ModelNotSupportedError,
)
from pytorchimagepipeline.pipelines.sam2segnet.utils import parse_voc_xml


@dataclass
class PascalVocFormat:
    root: Path
    mean_std: dict = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    data: list[str] = field(default_factory=list)
    ignore = 255

    def __post_init__(self):
        with (self.root / "mean_std.json").open() as file_obj:
            self.mean_std = json.load(file_obj)

        with (self.root / "classes.json").open() as file_obj:
            self.classes = json.load(file_obj)

    def __len__(self):
        return len(self.data)

    def get_data(self, mode):
        data_file = self.root / f"ImageSets/Segmentation/{mode}.txt"

        if data_file.exists():
            with data_file.open() as file_obj:
                self.data = file_obj.read().split("\n")
        else:
            self.data = []


class Sam2SegnetProgressManager(ProgressManager):
    """
    A class to manage and display progress for the Sam2Segnet pipeline.

    Example TOML Config:
        ```toml
        [permanences.progress_manager]
        type = "Sam2SegnetProgressManager"
        ```

    Attributes:
        progress_dict (dict): A dictionary to store progress bars for different stages.

    Methods:
        __init__(console=None):
            Initializes the progress manager with optional console output.
    """

    def __init__(self, console=None):
        """
        Initialize the class with optional console parameter.

        Args:
            console (optional): An optional console object for logging or displaying progress. Defaults to None.

        Attributes:
            progress_dict (dict): A dictionary to store progress bars for different tasks.
                - "epoch": Progress bar for tracking epochs with a specific color.
                - "train_val_test": Progress bar for tracking training, validation, and testing with a specific color.

        Methods:
            _init_live(): Initializes live progress display.
        """
        super().__init__(console)

        self.progress_dict |= {
            "create_masks": self._create_progress(color="#22ff55", with_status=True),
            "epoch": self._create_progress(color="#ff2255", with_status=True),
            "train_val_test": self._create_progress(color="#5522ff", with_status=True),
        }

        self._init_live()


@dataclass
class TrainingComponents(Permanence):
    """
    TrainingComponents is a class that provides the components required for training a segmentation model.

    Example TOML Config:
    ```toml
    [permanences.training_components]
    type = "TrainingComponents"
    params = { optimizer = "SGD", scheduler = "StepLR", criterion = "CrossEntropyLoss" }
    ```

    Attributes:
        optimizer (str): The name of the optimizer to use.
        scheduler (str): The name of the learning rate scheduler to use.
        criterion (str): The name of the loss function to use.
        Optimizer (torch.optim.Optimizer): The optimizer class.
        Scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler class.
        Criterion (torch.nn.Module): The loss function class.

    Methods:
        cleanup():
            Placeholder method for cleanup operations.
    """

    optimizer: str
    scheduler: str
    criterion: str

    def __post_init__(self):
        self._load_optimizer()
        self._load_scheduler()
        self._load_criterion()

    def _load_optimizer(self):
        optimizers = {
            "SGD": torch.optim.SGD,
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "RMSprop": torch.optim.RMSprop,
        }
        self.Optimizer = optimizers[self.optimizer]

    def _load_scheduler(self):
        schedulers = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        schedulers_default_params = {
            "StepLR": {"step_size": 30},
            "MultiStepLR": {"milestones": [30, 60, 90]},
            "ExponentialLR": {"gamma": 0.95},
            "ReduceLROnPlateau": {},
        }
        self.Scheduler = schedulers[self.scheduler]
        self.scheduler_params = schedulers_default_params[self.scheduler]

    def _load_criterion(self):
        criteria = {
            "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
            "BCELoss": torch.nn.BCELoss,
            "MSELoss": torch.nn.MSELoss,
            "L1Loss": torch.nn.L1Loss,
        }
        self.Criterion = criteria[self.criterion]

    def cleanup(self):
        pass


@dataclass
class HyperParameters(Permanence):
    """
    HyperParameters is a class that provides hyperparameters for the segmentation process.

    Example TOML Config:
    ```toml
    [permanences.hyperparams]
    type = "Hyperparameters"
    params = { config_file = "path/to/hyper_config.toml" }
    ```

    Attributes:
        hyper_config (Path | str): path to hyperparameters config file.

    Methods:
        cleanup():
            Placeholder method for cleanup operations.
    """

    config_file: Path | str

    def __post_init__(self):
        self.config_file = Path(self.config_file)
        self.hyperparams = self._load_hyperparams()

    def _load_hyperparams(self):
        with self.config_file.open(mode="rb") as file:
            return toml_load(file)

    def calculate_batch_size(self, device: torch.device):
        # Calculate the batch size based on the available memory
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - reserved_memory

        # Assuming each sample takes approximately 100MB of memory
        sample_memory = self.hyperparams.get("predicted_sample_size", 100) * 1024 * 1024
        batch_size = available_memory // sample_memory

        batch_max_size = self.hyperparams.get("batch_size_max", 20)

        self.hyperparams["batch_size"] = batch_max_size if batch_size > batch_max_size else batch_size

    def cleanup(self):
        pass


@dataclass
class Datasets(Permanence):
    """
    Datasets class is a class which provides torch datasets.

    Example TOML Config:
    ```toml
    [permanences.data]
    type = "Datasets"
    params = { root = "data/datasets/pascal", data_format = "pascalvoc" }
    ```

    Attributes:
        root (Path): The root directory for the dataset.
        data_format (str): The format of the dataset.

    Methods:
        __post_init__(): Initializes the sam_dataset attribute with a VisionDataset instance.
        cleanup(): Placeholder method for cleanup operations.
    """

    root: Path | str
    data_format: str

    def __post_init__(self):
        self.root = Path(self.root)
        self._load_data_container(self.data_format)
        self.sam_dataset: VisionDataset = SamDataset(self.root, data_container=self.data_container)
        train_transforms, val_test_transforms = self._get_transforms()

        self.segnet_dataset_train: VisionDataset = SegnetDataset(
            self.root, data_container=self.data_container, transforms=train_transforms, mode="train"
        )
        self.segnet_dataset_val: VisionDataset = SegnetDataset(
            self.root, data_container=self.data_container, transforms=val_test_transforms, mode="val"
        )
        self.segnet_dataset_test: VisionDataset = SegnetDataset(
            self.root, data_container=self.data_container, transforms=val_test_transforms, mode="test"
        )

    def cleanup(self):
        pass

    def _load_data_container(self, data_format):
        supported_formats = ["pascalvoc"]

        if data_format == "pascalvoc":
            self.data_container = PascalVocFormat(self.root)
        else:
            raise FormatNotSupportedError(format, supported_formats)

    def val_available(self):
        return len(self.segnet_dataset_val) > 0

    def test_available(self):
        return len(self.segnet_dataset_test) > 0

    def _get_transforms(self):
        """
        Based on https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
        Creates and returns the transformation pipelines for training and validation/testing datasets.
        The training transformations include:
        - Conversion to PIL image
        - Random horizontal flip
        - Random vertical flip
        - Random rotation within 15 degrees
        - Conversion to tensor
        - Normalization with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
        The validation/testing transformations include:
        - Conversion to PIL image
        - Conversion to tensor
        - Normalization with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
        Returns:
            tuple: A tuple containing the training transformations and validation/testing transformations.
        """
        mean = self.data_container.mean_std["mean"]
        std = self.data_container.mean_std["std"]

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
        ])

        val_test_transforms = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
        ])

        return train_transforms, val_test_transforms


@dataclass
class MaskCreator(Permanence):
    """
    MaskCreator is a class that performs various morphological operations on binary masks.

    Example TOML Config:
    ```toml
    [permanences.mask_creator]
    type = "MaskCreator"
    params = { morph_size = 1 , border_size = 1, ignore_value = 255 }
    ```

    Attributes:
        morph_size (int): Size of the morphological kernel. Default is 3.
        border_size (int): Size of the border to be created around the mask. Default is 4.
        ignore_value (int): Value to be used for ignored regions in the mask. Default is 255.
        current_masks (torch.Tensor): The current masks being processed.

    Methods:
        cleanup():
            Placeholder method for cleanup operations.

        set_current_masks(masks):
            Sets the current masks to the provided masks.

        create_mask(masks, masks_classes):
            Creates a mask by performing a series of morphological operations and merging masks with classes.

        merge_masks(mask_classes):
            Merges stacked binary masks where higher indices have priority.

        _create_border():
            Creates a border around the current masks using dilation.

        _erode(kernel_size=3, padding=1):
            Erodes the current masks using a max pooling operation.

        _dilate(kernel_size=3, padding=1):

        _opening():
            Performs an opening operation (erosion followed by dilation) on the current masks.

        _closing():
            Performs a closing operation (dilation followed by erosion) on the current masks.

        _get_kernel_size():
            Calculates and returns the kernel size based on the morphological size.
    """

    morph_size: int = 3
    border_size: int = 4
    ignore_value: int = 255
    current_masks: torch.Tensor = None

    def cleanup(self):
        pass

    def set_current_masks(self, masks: torch.ByteTensor) -> None:
        self.current_masks = masks

    def create_mask(
        self, masks: torch.FloatTensor | torch.cuda.FloatTensor
    ) -> torch.ByteTensor | torch.cuda.ByteTensor:
        """
        Creates and processes a mask tensor by applying a series of morphological operations and
        merging masks based on their position.

        Args:
            masks (torch.FloatTensor | torch.cuda.FloatTensor): A tensor containing the initial masks.

        Returns:
            torch.ByteTensor | torch.cuda.ByteTensor: The processed mask tensor after applying
            closing, opening, border creation and merging operations.
        """
        self.set_current_masks(masks)
        self._check_masks()
        self._closing()
        self._opening()
        self._create_border()
        self._merge_masks()
        return self.current_masks.type(torch.uint8)

    def _check_masks(self):
        """
        Checks the validity of the current masks.

        Raises:
            MaskNotAvailable: If `self.current_masks` is None.
            MaskShapeError: If `self.current_masks` does not have 4 dimensions.

        Warnings:
            If `self.current_masks` is not of type `torch.float`, a warning is issued and the masks are converted to float.
        """
        if self.current_masks is None:
            raise MaskNotAvailable()
        if len(self.current_masks.shape) != 4:
            raise MaskShapeError(self.current_masks.shape)
        if self.current_masks.dtype != torch.float32:
            warning(UserWarning("Masks are not in float32 format. Converting to float32."))
            self.set_current_masks(self.current_masks.type(torch.float32))

    def _merge_masks(self):
        """
        Merge stacked binary masks where higher indices have priority

        Args:
            stacked_masks (torch.Tensor): Shape (N, H, W) where N is number of masks

        Returns:
            torch.Tensor: Shape (H, W) merged mask
        """
        result = torch.zeros_like(self.current_masks[0])
        for mask in self.current_masks:
            result[mask > 0] = mask[mask > 0]
        self.set_current_masks(result)

    def _create_border(self):
        """
        Creates a border around the current masks by performing max pooling with a specified kernel size and padding.
        The mask + current mask are saved as the new current mask.

        Args:
            None

        Returns:
            None
        """
        kernel_size = 2 * self.border_size + 1
        padding = self.border_size

        # Perform max pooling to create the border effect
        dilated = self._dilate(kernel_size, padding)

        # Border is where dilated is 1 but mask is 0
        border = (dilated - self.current_masks).bool() * self.ignore_value

        # Add the border to the mask
        self.set_current_masks(self.current_masks + border)

    def _erode(self, kernel_size=3, padding=1):
        """
        Dilates the current masks using a max pooling operation.
        This Implementation is only used for binary masks.
        Morphological Operations for grayscale like described in the folowing articel are not implemented:
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm


        Args:
            kernel_size (int, optional): The size of the kernel to use for dilation. Default is 3.
            padding (int, optional): The amount of padding to add to the masks before dilation. Default is 1.

        Returns:
            torch.Tensor: The dilated masks.
        """
        masks = self.current_masks
        dilated = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=padding)
        return dilated

    def _dilate(self, kernel_size=3, padding=1):
        """
        Dilates the current masks using a max pooling operation.
        This Implementation is only used for binary masks.
        Morphological Operations for grayscale like described in the folowing articel are not implemented:
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm


        Args:
            kernel_size (int, optional): The size of the kernel to use for dilation. Default is 3.
            padding (int, optional): The amount of padding to add to the masks before dilation. Default is 1.

        Returns:
            torch.Tensor: The dilated masks.
        """
        masks = self.current_masks
        dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=padding)
        return dilated

    def _opening(self):
        kernel_size = self._get_kernel_size()
        padding = self.morph_size
        self.set_current_masks(self._erode(kernel_size=kernel_size, padding=padding))
        self.set_current_masks(self._dilate(kernel_size=kernel_size, padding=padding))

    def _closing(self):
        kernel_size = self._get_kernel_size()
        padding = self.morph_size
        self.current_masks = self._dilate(kernel_size=kernel_size, padding=padding)
        self.current_masks = self._erode(kernel_size=kernel_size, padding=padding)

    def _get_kernel_size(self):
        return 2 * self.morph_size + 1


@dataclass
class Network(Permanence):
    """
    Network is a class that provides a network to perform semantic segmentation.

    Example TOML Config:
    ```toml
    [permanences.network]
    type = "Network"
    params = { model = "deeplabv3_resnet50", num_classes = 21, pretrained = true }
    ```

    Attributes:
        model (str): The name of the model to use.
        num_classes (int): The number of classes in the dataset.
        pretrained (bool): Whether to use a pretrained model or not.
        model_instance (torch.nn.Module): The instance of the model.

    Methods:
        cleanup():
            Placeholder method for cleanup operations.
    """

    model: str
    num_classes: int
    pretrained: bool

    model_instance: torch.nn.Module = None

    def __post_init__(self):
        self.implemented_models = {
            "fcn_resnet50": segmentation.fcn_resnet50,
            "fnc_resnet101": segmentation.fcn_resnet101,
            "deeplabv3_resnet50": segmentation.deeplabv3_resnet50,
            "deeplabv3_resnet101": segmentation.deeplabv3_resnet101,
            "deeplabv3_mobilenet_v3_large": segmentation.deeplabv3_mobilenet_v3_large,
            "lsrap_mobilenet_v3_large": segmentation.lraspp_mobilenet_v3_large,
        }
        self.pretrained_weights = {
            "fcn_resnet50": segmentation.FCN_ResNet50_Weights.DEFAULT,
            "fnc_resnet101": segmentation.FCN_ResNet101_Weights.DEFAULT,
            "deeplabv3_resnet50": segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
            "deeplabv3_resnet101": segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
            "deeplabv3_mobilenet_v3_large": segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
            "lsrap_mobilenet_v3_large": segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        }
        self._load_model()

    def cleanup(self):
        self._load_model()

    def _load_model(self):
        if self.model not in self.implemented_models:
            raise ModelNotSupportedError(self.model, self.implemented_models)
        get_model_func = self.implemented_models[self.model]
        weights = self.pretrained_weights[self.model] if self.pretrained else None

        self.model_instance = get_model_func(weights=weights, num_classes=self.num_classes)


class SamDataset(VisionDataset):
    def __init__(self, root=None, data_container=None):
        self.root = Path(root)

        self.dataobj = type(data_container)(**dataclasses.asdict(data_container))
        self.dataobj.get_data("train")

        with (self.root / "classes.json").open() as file_obj:
            self.class_idx = json.load(file_obj)

        self.target_location = self.root / "SegmentationClassSAM"

    def __len__(self):
        return len(self.dataobj.data)

    def __getitem__(self, index):
        # Read filestem
        filestem = self.dataobj.data[index]

        # Read Image
        img = cv2.imread(self.root / "JPEGImages" / f"{filestem}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read Annotations
        annotation, error = parse_voc_xml(self.root / "Annotations" / f"{filestem}.xml")
        if error:
            raise error
        bboxes = [obj.bndbox for obj in annotation.objects]
        bbox_classes = [obj.name for obj in annotation.objects]

        return img, bboxes, bbox_classes, filestem

    def all_created(self):
        files = (Path(stem).with_suffix(".png") for stem in self.dataobj.data)
        return torch.tensor(list(map(Path.exists, map(self.target_location.joinpath, files))), dtype=bool).all()

    def save_item(self, index, mask):
        mask = mask.squeeze()
        torchvision.utils.save_image(mask, str(self.target_location / self.images[index].name), "png")


class SegnetDataset(VisionDataset):
    def __init__(self, root=None, transforms=None, mode="train", data_container=None):
        super().__init__(root, transforms=transforms)
        if mode not in ["train", "val", "test"]:
            raise ModeError(mode)
        self.root = Path(root)
        self.transforms = transforms
        self.mode = mode

        self.dataobj = type(data_container)(**dataclasses.asdict(data_container))
        self.dataobj.get_data(mode)

    def __len__(self):
        return len(self.dataobj)

    def __getitem__(self, index):
        filestem = self.dataobj.data[index]

        # Read Image
        img = decode_image(self.root / "JPEGImages" / f"{filestem}.jpg")

        # Read Mask
        mask = Mask(decode_image(self.root / "SegmentationClassSAM" / f"{filestem}.png"))

        if self.transforms:
            img, mask = self.transforms(img, mask)
        mask = mask.to(torch.long).squeeze()
        return img, mask
