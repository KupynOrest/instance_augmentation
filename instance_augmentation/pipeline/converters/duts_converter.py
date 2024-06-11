from typing import List
import os
import cv2

from PIL import Image

from instance_augmentation.pipeline.entities import DatasetItem, GeneratedResult
from instance_augmentation.pipeline.converters import BaseConverter


class DUTSConverter(BaseConverter):
    def __init__(self, save_folder: str):
        super().__init__(save_folder, ["images", "masks"])

    def save_sample(self, image: Image, annotation: DatasetItem, folders: List[str], filename: str):
        for folder in folders:
            image.save(os.path.join(self.save_folder, folder, annotation.split, "images", f"{filename}.jpg"))
            mask = cv2.imread(annotation.annotation_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (image.size[0], image.size[1]), cv2.INTER_LANCZOS4)
            cv2.imwrite(os.path.join(self.save_folder, folder, annotation.split, "masks", f"{filename}.png"), mask)

    def save_annotation(self, generation: GeneratedResult, annotation: DatasetItem, index: int = 0):
        image_index = annotation.index if annotation.index is not None else self._index
        self._index += 1
        self.save_sample(
            generation.original_image, annotation, ["real", "real_generated"], f"{annotation.split}_{image_index}_0"
        )
        self.save_sample(generation.image, annotation, ["generated", "real_generated"], f"{annotation.split}_{image_index}_{index + 1}")
