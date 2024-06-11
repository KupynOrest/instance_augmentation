from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
import albumentations as A

from instance_augmentation.pipeline.entities import Annotation
from instance_augmentation.pipeline.preprocessors.base_preprocessor import BaseDataPreprocessor


class ResizeDataPreprocessor(BaseDataPreprocessor):
    def __init__(self, resize: int):
        super().__init__()
        self.resize = A.LongestMaxSize(max_size=resize, always_apply=True)

    @staticmethod
    def _rescale_bbox(bbox: np.ndarray, image: np.ndarray, resize_image: np.ndarray) -> np.ndarray:
        bbox = bbox.copy()
        bbox[0] = bbox[0] * resize_image.shape[0] / image.shape[0]
        bbox[1] = bbox[1] * resize_image.shape[1] / image.shape[1]
        bbox[2] = bbox[2] * resize_image.shape[0] / image.shape[0]
        bbox[3] = bbox[3] * resize_image.shape[1] / image.shape[1]
        return bbox

    def __call__(self, image: Image.Image, annotations: List[Annotation]) -> Tuple[Image.Image, List[Annotation]]:
        resized_image = self.resize(image=np.array(image))["image"]
        for annotation in annotations:
            if annotation.mask is not None:
                annotation.mask = cv2.resize(np.array(annotation.mask), (resized_image.shape[1], resized_image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            if annotation.bbox is not None:
                annotation.bbox = self._rescale_bbox(annotation.bbox, np.array(image), resized_image)
        return Image.fromarray(resized_image), annotations
