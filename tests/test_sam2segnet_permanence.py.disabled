import logging
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import torch
from PIL import Image

from pytorchimagepipeline.pipelines.sam2segnet.permanence import (
    Datasets,
    MaskCreator,
    MaskNotAvailable,
    MaskShapeError,
    SamDataset,
    SegnetDataset,
)


@pytest.fixture
def mask_creator():
    return MaskCreator(morph_size=1, border_size=1, ignore_value=255)


@pytest.fixture
def masks(request):
    if request.param == "small":
        masks = torch.zeros((4, 1, 4, 7), dtype=torch.float32)
        masks[0, 0, 3, 6] = 7
        masks[1, 0, 0:2, 1:3] = 5
        masks[2, 0, 1:3, 4:6] = 3
        masks[3, 0, 1:, :2] = 9
        return masks
    elif request.param == "large":
        masks = torch.zeros((4, 1, 40, 70), dtype=torch.float32)
        masks[0, 0, 30:, 60:] = 7
        masks[1, 0, :20, 10:30] = 5
        masks[2, 0, 10:30, 40:60] = 3
        masks[3, 0, 10:, :20] = 9
        return masks


@pytest.mark.parametrize("masks", ["small", "large"], indirect=True)
def test_create_mask_shape(masks, mask_creator):
    expected_shape = masks.shape[1:]
    result = mask_creator.create_mask(masks)
    assert result.shape == expected_shape, "The shape of the result mask is incorrect."


@pytest.mark.parametrize("masks", ["small", "large"], indirect=True)
def test_create_mask_values(masks, mask_creator, request):
    result = mask_creator.create_mask(masks)
    assert torch.all(result >= 0) and torch.all(
        result <= 255
    ), "The values in the result mask are out of expected range."
    assert result.dtype == torch.uint8, "The dtype of the result mask is incorrect."
    scale = 200 / result.shape[1]
    mask_img_0 = torch.nn.functional.interpolate(
        masks[0].view(1, *result.shape).type(torch.uint8), scale_factor=scale
    ).squeeze()
    mask_img_1 = torch.nn.functional.interpolate(
        masks[1].view(1, *result.shape).type(torch.uint8), scale_factor=scale
    ).squeeze()
    mask_img_2 = torch.nn.functional.interpolate(
        masks[2].view(1, *result.shape).type(torch.uint8), scale_factor=scale
    ).squeeze()
    mask_img_3 = torch.nn.functional.interpolate(
        masks[3].view(1, *result.shape).type(torch.uint8), scale_factor=scale
    ).squeeze()
    result_img = torch.nn.functional.interpolate(result.view(1, *result.shape), scale_factor=scale).squeeze()
    request.node.user_properties.append(("image_tensor", "Mask 0", mask_img_0))
    request.node.user_properties.append(("image_tensor", "Mask 1", mask_img_1))
    request.node.user_properties.append(("image_tensor", "Mask 2", mask_img_2))
    request.node.user_properties.append(("image_tensor", "Mask 3", mask_img_3))
    request.node.user_properties.append(("image_tensor", "Combined Mask", result_img))


def test_check_masks_none(mask_creator):
    with pytest.raises(MaskNotAvailable):
        mask_creator._check_masks()


def test_check_masks_invalid_shape(mask_creator):
    mask_creator.set_current_masks(torch.zeros(3, 224, 224))  # 3D tensor instead of 4D
    with pytest.raises(MaskShapeError):
        mask_creator._check_masks()


def test_check_masks_invalid_dtype(mask_creator, caplog):
    mask_creator.set_current_masks(torch.zeros(1, 3, 224, 224, dtype=torch.uint8))  # int32 instead of float
    with caplog.at_level(logging.WARNING):
        mask_creator._check_masks()

    assert "Masks are not in float32 format. Converting to float32." in caplog.text
    assert mask_creator.current_masks.dtype == torch.float32


def test_check_masks_valid(mask_creator):
    mask_creator.set_current_masks(torch.zeros(1, 3, 224, 224, dtype=torch.float32))
    mask_creator._check_masks()  # Should not raise any exceptions


@pytest.mark.parametrize(
    "masks, kernel_size, padding, expected",
    [
        (
            torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32),
            3,
            1,
            torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32),
        ),
        (
            torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32),
            3,
            0,
            torch.tensor([[[[0]]]], dtype=torch.float32),
        ),
        (
            torch.tensor(
                [[[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]]],
                dtype=torch.float32,
            ),
            3,
            1,
            torch.tensor(
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
                dtype=torch.float32,
            ),
        ),
        (
            torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32),
            5,
            2,
            torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32),
        ),
        (
            torch.zeros((1, 1, 3, 3), dtype=torch.float32),
            3,
            1,
            torch.zeros((1, 1, 3, 3), dtype=torch.float32),
        ),
        (
            torch.ones((1, 1, 3, 3), dtype=torch.float32),
            3,
            1,
            torch.ones((1, 1, 3, 3), dtype=torch.float32),
        ),
    ],
    ids=[
        "3x3_input_3x3_kernel_with_padding",
        "3x3_input_3x3_kernel_no_padding",
        "5x5_input_3x3_kernel_with_padding",
        "3x3_input_5x5_kernel_with_padding",
        "3x3_kernel_zeros",
        "3x3_kernel_ones",
    ],
)
def test_erode(mask_creator, masks, kernel_size, padding, expected, request):
    mask_creator.set_current_masks(masks)
    eroded = mask_creator._erode(kernel_size=kernel_size, padding=padding)
    assert torch.equal(eroded, expected), f"Expected {expected}, but got {eroded}"
    scale = 200 / masks.shape[2]
    mask_img = torch.nn.functional.interpolate(masks.type(torch.uint8), scale_factor=scale).squeeze()
    result_img = torch.nn.functional.interpolate(eroded.type(torch.uint8), scale_factor=scale).squeeze()
    request.node.user_properties.append(("image_tensor", "Mask", mask_img))
    request.node.user_properties.append(("image_tensor", "Result", result_img))


@pytest.mark.parametrize(
    "masks, kernel_size, padding, expected",
    [
        (
            torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32),
            3,
            1,
            torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32),
        ),
        (
            torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32),
            3,
            0,
            torch.tensor([[[[1]]]], dtype=torch.float32),
        ),
        (
            torch.tensor(
                [[[[0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]]]],
                dtype=torch.float32,
            ),
            3,
            1,
            torch.tensor(
                [[[[0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1]]]],
                dtype=torch.float32,
            ),
        ),
        (
            torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32),
            5,
            2,
            torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32),
        ),
        (
            torch.zeros((1, 1, 3, 3), dtype=torch.float32),
            3,
            1,
            torch.zeros((1, 1, 3, 3), dtype=torch.float32),
        ),
        (
            torch.ones((1, 1, 3, 3), dtype=torch.float32),
            3,
            1,
            torch.ones((1, 1, 3, 3), dtype=torch.float32),
        ),
    ],
    ids=[
        "3x3_input_3x3_kernel_with_padding",
        "3x3_input_3x3_kernel_no_padding",
        "5x5_input_3x3_kernel_with_padding",
        "3x3_input_5x5_kernel_with_padding",
        "3x3_kernel_zeros",
        "3x3_kernel_ones",
    ],
)
def test_dilate(mask_creator, masks, kernel_size, padding, expected, request):
    mask_creator.set_current_masks(masks)
    dilated = mask_creator._dilate(kernel_size=kernel_size, padding=padding)
    assert torch.equal(dilated, expected), f"Expected {expected}, but got {dilated}"
    scale = 200 / masks.shape[2]
    mask_img = torch.nn.functional.interpolate(masks.type(torch.uint8), scale_factor=scale).squeeze()
    result_img = torch.nn.functional.interpolate(dilated.type(torch.uint8), scale_factor=scale).squeeze()
    request.node.user_properties.append(("image_tensor", "Mask", mask_img))
    request.node.user_properties.append(("image_tensor", "Result", result_img))


@pytest.fixture
def mock_get_transforms():
    with patch(
        "pytorchimagepipeline.pipelines.sam2segnet.permanence.Datasets._get_transforms", return_value=(None, None)
    ):
        yield


@pytest.fixture
def datasets(tmp_datasets):
    data_format = "pascalvoc"
    return Datasets(root=tmp_datasets, data_format=data_format)


@pytest.fixture
def datasets_empty_val_test(tmp_datasets):
    data_format = "pascalvoc"
    datasets = Datasets(root=tmp_datasets, data_format=data_format)
    datasets.segnet_dataset_val.dataobj.data = []
    datasets.segnet_dataset_test.dataobj.data = []
    return datasets


@pytest.fixture(scope="session")
def tmp_datasets(tmp_path_factory):
    dataset_dir = tmp_path_factory.mktemp("Dataset")
    sets_dir = dataset_dir / "ImageSets/Segmentation"
    sets_dir.mkdir(parents=True)
    image_dir = dataset_dir / "JPEGImages"
    image_dir.mkdir(parents=True)
    annotation_dir = dataset_dir / "Annotations"
    annotation_dir.mkdir(parents=True)
    segsam_dir = dataset_dir / "SegmentationClassSAM"
    segsam_dir.mkdir(parents=True)

    # Create train, val and test sets
    image_train_list = sets_dir / "train.txt"
    image_train_list.write_text("image1\nimage2\nimage3", encoding="utf-8")
    image_val_list = sets_dir / "val.txt"
    image_val_list.write_text("image4\nimage5\nimage6", encoding="utf-8")
    image_test_list = sets_dir / "test.txt"
    image_test_list.write_text("image7\nimage8\nimage9", encoding="utf-8")

    # Create classes.json
    content = '{"class0": 1, "class1": 2, "class2": 3}'
    classes_file = dataset_dir / "classes.json"
    classes_file.write_text(content, encoding="utf-8")

    # Create mean_std.json
    content = '{"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}'
    mean_std_file = dataset_dir / "mean_std.json"
    mean_std_file.write_text(content, encoding="utf-8")

    def compute_test_annotations(idx):
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "TestData"
        ET.SubElement(root, "filename").text = f"image{idx}.jpg"
        ET.SubElement(root, "path").text = str(image_dir / f"image{idx}.jpg")
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "TEMPORARY"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "224"
        ET.SubElement(size, "height").text = "224"
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        for i in range(3):
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = f"class{i}"
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(112 - 16 * i)
            ET.SubElement(bndbox, "ymin").text = str(112 - 16 * i)
            ET.SubElement(bndbox, "xmax").text = str(112 + 16 * i)
            ET.SubElement(bndbox, "ymax").text = str(112 + 16 * i)

        tree = ET.ElementTree(root)
        return tree

    def compute_test_images(idx):
        r, g, b = 0, 32, 64
        r += (16 * idx) & 0xFF
        g += (16 * idx) & 0xFF
        b += (16 * idx) & 0xFF
        rgb = np.array([[[r, g, b]]], dtype=np.uint8)
        ones = np.ones((224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(ones * rgb)
        return img

    def compute_test_masks():
        combined = np.zeros((224, 224), dtype=np.uint8)

        num_masks = 3
        split_base = 224 // (num_masks + 1) * 2
        ignore_width = 4

        for i in range(num_masks):
            # Create split positions
            split_pos = -split_base + (i * split_base)
            pos_ignore = split_pos - ignore_width
            next_pos = split_pos + split_base - ignore_width

            # Create ignore and label masks
            ignore = np.tri(224, 224, split_pos, dtype=np.uint8) - np.tri(224, 224, pos_ignore, dtype=np.uint8)
            label = np.tri(224, 224, next_pos, dtype=np.uint8) - np.tri(224, 224, split_pos, dtype=np.uint8)
            combined += (ignore * 255) + (label * (i + 1))

        split_pos = -split_base + ((i + 1) * split_base)
        pos_ignore = split_pos - ignore_width
        next_pos = split_pos + split_base - ignore_width
        ignore = np.tri(224, 224, split_pos, dtype=np.uint8) - np.tri(224, 224, pos_ignore, dtype=np.uint8)
        img = Image.fromarray(combined + (ignore * 255))
        return img

    for idx in range(1, 10):
        # Create Images
        img = compute_test_images(idx)
        fn = image_dir / f"image{idx}.jpg"
        img.save(fn)

        # Create Annotations
        tree = compute_test_annotations(idx)
        tree.write(annotation_dir / f"image{idx}.xml")

        # Create Segmentation Masks
        mask = compute_test_masks()
        mask_fn = segsam_dir / f"image{idx}.png"
        mask.save(mask_fn)
    return dataset_dir


def test_post_init_sam_dataset(datasets):
    data0 = datasets.sam_dataset[0]
    assert isinstance(datasets.sam_dataset, SamDataset)
    assert datasets.sam_dataset.root == datasets.root
    assert len(datasets.sam_dataset) == 3
    assert len(data0) == 4


def test_post_init_segnet_dataset_train(datasets):
    assert isinstance(datasets.segnet_dataset_train, SegnetDataset), "Expected SegnetDataset object."
    assert datasets.segnet_dataset_train.root == datasets.root, "Root path is incorrect."
    assert len(datasets.segnet_dataset_train) == 3, "Expected 3 Datapoints."
    assert len(datasets.segnet_dataset_train[0]) == 2, "Expected a tuple with 2 elements."
    assert datasets.segnet_dataset_train.dataobj.data == ["image1", "image2", "image3"]


def test_post_init_segnet_dataset_val(datasets):
    assert isinstance(datasets.segnet_dataset_val, SegnetDataset), "Expected SegnetDataset object."
    assert datasets.segnet_dataset_val.root == datasets.root, "Root path is incorrect."
    assert len(datasets.segnet_dataset_val) == 3, "Expected 3 Datapoints."
    assert len(datasets.segnet_dataset_val[0]) == 2, "Expected a tuple with 2 elements."
    assert datasets.segnet_dataset_val.dataobj.data == ["image4", "image5", "image6"]


def test_post_init_segnet_dataset_test(datasets):
    assert isinstance(datasets.segnet_dataset_test, SegnetDataset), "Expected SegnetDataset object."
    assert datasets.segnet_dataset_test.root == datasets.root, "Root path is incorrect."
    assert len(datasets.segnet_dataset_test) == 3, "Expected 3 Datapoints."
    assert len(datasets.segnet_dataset_test[0]), "Expected a tuple with 2 elements."
    assert datasets.segnet_dataset_test.dataobj.data == ["image7", "image8", "image9"]


def test_post_init_segnet_dataset_empty_val_test(datasets_empty_val_test):
    assert len(datasets_empty_val_test.segnet_dataset_train) == 3, "Expected 3 Datapoints."
    assert len(datasets_empty_val_test.segnet_dataset_train[0]) == 2, "Expected a tuple with 2 elements."
    assert datasets_empty_val_test.segnet_dataset_train.dataobj.data == ["image1", "image2", "image3"]
    assert len(datasets_empty_val_test.segnet_dataset_val) == 0, "Expected 0 Datapoints."
    assert datasets_empty_val_test.segnet_dataset_val.dataobj.data == []
    assert len(datasets_empty_val_test.segnet_dataset_test) == 0, "Expected 0 Datapoints."
    assert datasets_empty_val_test.segnet_dataset_test.dataobj.data == []


def test_val_available(datasets):
    assert datasets.val_available(), "Expected val_available to be True."


def test_val_available_false(datasets_empty_val_test):
    assert not datasets_empty_val_test.val_available(), "Expected val_available not to be True."


def test_test_available(datasets):
    assert datasets.test_available(), "Expected test_available to be True."


def test_test_available_false(datasets_empty_val_test):
    assert not datasets_empty_val_test.test_available(), "Expected test_available not to be True."
