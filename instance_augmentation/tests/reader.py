from typing import List

import cv2

from instance_augmentation.pipeline.entities import Annotation, DatasetItem
from instance_augmentation.pipeline.readers.base_reader import BaseDatasetReader
from instance_augmentation.pipeline.readers.coco_dataset import COCODatasetReader


class COCO4Reader(COCODatasetReader):
    @staticmethod
    def _get_folder_path(split: str):
        return "data"

    def read_dataset(self) -> List[DatasetItem]:
        dataset = self._read_subset("./fixture/coco-4/labels.json", "train")
        return dataset


class DummyMaskDatasetReader(BaseDatasetReader):
    def __init__(self):
        super().__init__()

    def read_dataset(self) -> List[DatasetItem]:
        mask = cv2.imread("./fixture/pes_patron.png")[:, :, -1]
        return [
            DatasetItem(
                base_path="./fixture",
                image_path="pes_patron.jpg",
                annotations=[Annotation(label="dog", mask=mask)],
                split="train",
            )
        ]
