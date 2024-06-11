import os
import shutil
import pytest

import glob
import cv2
import numpy as np
from omegaconf import OmegaConf

from instance_augmentation.pipeline.dataset_generator import DatasetGenerator
from instance_augmentation.tests.reader import DummyMaskDatasetReader
from instance_augmentation.pseudolabel_dataset import create_annotations
from instance_augmentation.pipeline.readers import CustomDatasetReader
from instance_augmentation.augment import Augmenter


def test_end_to_end():
    if os.path.exists("test_e2e") and os.path.isdir("test_e2e"):
        shutil.rmtree("test_e2e")
    os.makedirs("test_e2e")
    create_annotations("./fixture/dogs_test", "test_e2e/generated", dataset_type="custom", class_names=["dog"])
    reader = CustomDatasetReader("./fixture/dogs_test", {}, "test_e2e/generated/annotations.json")
    dataset_generator = DatasetGenerator.from_params(
        dataset_reader=reader,
        save_folder="test_e2e/generated",
        preprocessing="resize",
        target_image_size=1024,
        base_inpainting_model="SG161222/RealVisXL_V3.0",
        generator="inpaint_sdxl_adapter",
        num_samples=1,
        num_inference_steps=20,
        control_methods=["t2i_depth", "t2i_sketch"],
        control_weights=[0.9, 0.5],
    )
    dataset_generator.run()
    augmenter = Augmenter("test_e2e/generated", p=1.0)
    for image_path in glob.glob("./fixture/dogs_test/*"):
        image_name = os.path.split(image_path)[1]
        original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        augmented_image = augmenter.augment_image(original_image, image_name)
        original_image = cv2.resize(original_image, (augmented_image.shape[1], augmented_image.shape[0]))
        cv2.imwrite(f"test_e2e/{image_name}", cv2.cvtColor(np.hstack((original_image, augmented_image)), cv2.COLOR_RGB2BGR))
    assert len(glob.glob("test_e2e/generated/images/*")) == 3


@pytest.mark.parametrize(
    "config_name,expected_files", [("./fixture/configs/mask.yaml", 1), ("./fixture/configs/coco_4.yaml", 4)]
)
def test_pipeline(config_name, expected_files):
    if os.path.exists("test_results") and os.path.isdir("test_results"):
        shutil.rmtree("test_results")
    os.makedirs("test_results")
    generate_dataset(config_name)
    assert len(glob.glob("./test_results/generated/train/images/*")) == expected_files


def test_pipeline_from_params():
    reader = DummyMaskDatasetReader()
    dataset_generator = DatasetGenerator.from_params(
        dataset_reader=reader,
        save_folder="test_results",
    )
    dataset_generator.run()


def generate_dataset(config_path: str):
    config = OmegaConf.load(config_path)
    dataset_generator = DatasetGenerator.from_config(config)
    dataset_generator.run()
