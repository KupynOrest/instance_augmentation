import os
import shutil
import pytest

import glob
from omegaconf import OmegaConf

from instance_augmentation.pipeline.dataset_generator import DatasetGenerator
from instance_augmentation.tests.reader import DummyMaskDatasetReader
from instance_augmentation.pseudolabel_dataset import create_annotations
from instance_augmentation.pipeline.readers import CustomDatasetReader


def test_end_to_end():
    os.makedirs("test_results", exist_ok=True)
    create_annotations("./fixture/dogs_test", "test_results/generated", dataset_type="custom", class_names=["dog"])
    reader = CustomDatasetReader("./fixture/dogs_test", {}, "test_results/generated/annotations.json")
    dataset_generator = DatasetGenerator.from_params(
        dataset_reader=reader,
        save_folder="test_results/generated",
        preprocessing="resize",
        target_image_size=1280,
        base_inpainting_model="SG161222/RealVisXL_V3.0",
        generator="inpaint_sdxl_adapter",
        num_samples=1,
        num_inference_steps=50,
        control_methods=["t2i_depth", "t2i_sketch"],
        control_weights=[0.9, 0.5],
    )
    dataset_generator.run()
    assert len(glob.glob("./test_results/generated/train/images/*")) == 1


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
