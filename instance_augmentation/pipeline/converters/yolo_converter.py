from typing import List
import os

from PIL import Image
import numpy as np

from instance_augmentation.pipeline.entities import DatasetItem, GeneratedResult
from instance_augmentation.pipeline.converters import BaseConverter


class YOLOConverter(BaseConverter):
    def __init__(self, save_folder: str):
        super().__init__(save_folder, ["images", "labels"])

    def _get_bboxes(self, image: Image, item: DatasetItem):
        bboxes = []
        width, height = image.size
        for anno in item.annotations:
            bbox = [0, 0, 0, 0]
            bbox[0] = ((anno.bbox[0] + anno.bbox[2]) / 2) / width
            bbox[1] = ((anno.bbox[1] + anno.bbox[3]) / 2) / height
            bbox[2] = (anno.bbox[2] - anno.bbox[0]) / width
            bbox[3] = (anno.bbox[3] - anno.bbox[1]) / height
            bboxes.append([anno.category_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        return bboxes

    def save_sample(self, image: Image, annotation: DatasetItem, folders: List[str], filename: str):
        bboxes = self._get_bboxes(image, annotation)
        for folder in folders:
            image_save_path = os.path.join(self.save_folder, folder, annotation.split, "images", f"{filename}.jpg")
            if not os.path.isfile(image_save_path):
                image.save(image_save_path)
            if len(bboxes) > 0:
                np.savetxt(
                    os.path.join(self.save_folder, folder, annotation.split, "labels", f"{filename}.txt"), bboxes
                )

    def save_annotation(self, generation: GeneratedResult, annotation: DatasetItem, index: int = 0):
        image_index = annotation.index if annotation.index is not None else self._index
        self._index += 1
        self.save_sample(
            generation.original_image, annotation, ["real", "real_generated"], f"{annotation.split}_{image_index}_0"
        )
        self.save_sample(generation.image, annotation, ["generated", "real_generated"], f"{annotation.split}_{image_index}_{index + 1}")
