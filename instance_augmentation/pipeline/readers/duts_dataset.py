import os
import glob
from tqdm import tqdm
import logging
from typing import List

import cv2
import numpy as np

from instance_augmentation.pipeline.entities import Annotation, DatasetItem
from instance_augmentation.pipeline.readers.base_reader import BaseDatasetReader
from instance_augmentation.pipeline.image_captioner import ImageCaptioner

logger = logging.Logger(__name__)
MIN_CONTOUR_AREA = 50
MIN_SIZE = 10


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)


class DUTSDatasetReader(BaseDatasetReader):
    def __init__(self, data_path: str) -> None:
        self.image_captioner = ImageCaptioner()
        super().__init__(data_path)

    @staticmethod
    def _valid_contour(w: int, h: int) -> bool:
        valid_area = w * h > MIN_CONTOUR_AREA
        return valid_area and w > MIN_SIZE and h > MIN_SIZE

    def _parse_mask(self, mask: np.ndarray) -> List[Annotation]:
        binary_mask = (mask > 128).astype(np.uint8)
        annotations = []
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._valid_contour(w, h):
                single_mask = np.zeros_like(mask)
                single_mask[y : y + h, x : x + w] = mask[y : y + h, x : x + w]
                annotations.append(Annotation(label="", bbox=[x, y, x + w, y + h], mask=single_mask))
        return annotations

    def _get_caption(self, image: np.ndarray, annotation: Annotation) -> str:
        x, y, x1, y1 = annotation.bbox
        result = np.array(image[y:y1, x:x1, :])
        label = self.image_captioner.generate_caption(result)
        return label

    def _assign_splits(self, dataset: List[DatasetItem]) -> List[DatasetItem]:
        for item in dataset:
            item.split = "train"
        return dataset

    def _read_subset(self, image_paths: List[str], split: str) -> List[DatasetItem]:
        dataset = []
        for index in tqdm(range(len(image_paths))):
            image_path = image_paths[index]
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            mask_path = image_path.replace("Image", "Mask").replace(".jpg", ".png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            annotations = self._parse_mask(mask)
            for annotation in annotations:
                annotation.label = self._get_caption(image, annotation)
            dataset.append(
                DatasetItem(
                    base_path=self.data_path,
                    image_path=image_path.replace(f"{self.data_path}/", ""),
                    annotations=annotations,
                    annotation_path=mask_path,
                    split=split,
                    index=index,
                )
            )
        dataset = self._assign_splits(dataset)
        return dataset

    def read_dataset(self) -> List[DatasetItem]:
        dataset_folder = self.data_path
        train_subset = self._read_subset(
            glob.glob(os.path.join(dataset_folder, "DUTS-TR", "DUTS-TR-Image", "*.jpg")), "train"
        )
        return train_subset
