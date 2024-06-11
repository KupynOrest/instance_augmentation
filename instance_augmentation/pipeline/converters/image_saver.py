from typing import List
import os

from PIL import Image
import numpy as np

from instance_augmentation.pipeline.entities import DatasetItem, GeneratedResult
from instance_augmentation.pipeline.converters import BaseConverter


class ImageSaver(BaseConverter):
    def __init__(self, save_folder: str):
        super().__init__(save_folder, ["images"])

    def save_sample(self, image: Image, annotation: DatasetItem, folders: List[str], filename: str):
        for folder in folders:
            image.save(os.path.join(self.save_folder, folder, annotation.split, "images", f"{filename}.jpg"))

    def save_annotation(self, generation: GeneratedResult, annotation: DatasetItem, index: int = 0):
        image_index = annotation.index if annotation.index is not None else self._index
        self._index += 1
        self.save_sample(
            generation.original_image, annotation, ["real", "real_generated"], f"{annotation.split}_{image_index}_0"
        )
        self.save_sample(
            Image.fromarray(
                np.hstack(
                    (np.array(generation.original_image), np.array(generation.image), generation.rendered_instances)
                )
            ),
            annotation,
            ["generated", "real_generated"],
            f"{annotation.split}_{image_index}_{index + 1}",
        )
