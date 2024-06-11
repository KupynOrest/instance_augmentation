from typing import Dict, Any

import yaml
from fire import Fire
from omegaconf import OmegaConf

from instance_augmentation.pipeline.dataset_generator import DatasetGenerator


def load_yaml(x: str) -> Dict[str, Any]:
    with open(x) as fd:
        config = yaml.load(fd, yaml.FullLoader)
        return config


def generate_dataset(config_path: str = "config/coco_dataset.yaml"):
    config = OmegaConf.load(config_path)
    dataset_generator = DatasetGenerator.from_config(config)
    dataset_generator.run()


if __name__ == "__main__":
    Fire(generate_dataset)
