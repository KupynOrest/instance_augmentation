from typing import List, Dict, Any, Union, Optional, Tuple
import os
import random

import cv2
import numpy as np
from PIL import Image
from pytorch_toolbelt.utils import rle_decode

from instance_augmentation.db_wrapper import DBWrapper
from instance_augmentation.pipeline.utils import blend


class Augmenter:
    def __init__(self, generated_data_path: str, p: float = 0.5, rgb: bool = True):
        self.data_path = generated_data_path
        self.database = DBWrapper(os.path.join(generated_data_path, "database.json"))
        self.p = p
        self.rgb = rgb

    def load_image(self, base_path: str, filename: str) -> np.ndarray:
        """
        Load image from base_path and filename
        :param base_path: base path to the dataset
        :param filename: filename of the image
        :return: image as np.ndarray
        """
        metadata = self._get_metadata(filename)
        original_image = cv2.imread(os.path.join(base_path, filename))
        if self.rgb:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        if metadata is None or len(metadata["annotations"]) == 0:
            return original_image
        return self.augment(image=original_image, metadata=metadata)

    def augment_image(self, image: Union[np.ndarray, Image.Image], filename: str) -> Union[Image.Image, np.ndarray]:
        """
        Augment image from base_path and filename
        :param image: image as np.ndarray or PIL.Image
        :param filename: filename of the image
        :return: image as np.ndarray
        """
        convert_to_pil = False
        if isinstance(image, Image.Image):
            convert_to_pil = True
            image = np.array(image)
        metadata = self._get_metadata(filename)
        if metadata is not None and len(metadata["annotations"]) > 0:
            augmented_image = self.augment(image=image, metadata=metadata)
        else:
            augmented_image = image
        if convert_to_pil:
            augmented_image = Image.fromarray(augmented_image)
        return augmented_image

    def _get_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        metadata = self.database.search(filename)
        if metadata is None:
            return None
        return metadata

    @staticmethod
    def _calculate_intersection_area(bbox1: List[int], bbox2: List[int]) -> float:
        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        return x_overlap * y_overlap

    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> bool:
        intersection_area = self._calculate_intersection_area(bbox1, bbox2)
        return intersection_area > 0

    def _get_original_annotations(self, annotations: List[Dict[str, Any]], inpainted_annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotations_to_inpaint = []
        for annotation in annotations:
            for inpainted_annotation in inpainted_annotations:
                if self._bbox_overlap(annotation["bbox"], inpainted_annotation["bbox"]):
                    annotations_to_inpaint.append(annotation)
        return annotations_to_inpaint

    def augment(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        augmented_image = image.copy()
        instances = cv2.imread(os.path.join(self.data_path, "images", metadata["generated_filename"]))
        if self.rgb:
            instances = cv2.cvtColor(instances, cv2.COLOR_BGR2RGB)
        annotations = metadata["annotations"]
        if metadata["num_annotations"] == 1 and len(annotations) == 1:
            return instances if random.random() < self.p else augmented_image
        augmented_image, inpainted_annotations = self.mix_annotations(augmented_image, instances, annotations)
        original_annotations = self._get_original_annotations(metadata["original_annotations"], inpainted_annotations)
        augmented_image = self._inpaint_original_annotations(augmented_image, image,
                                                             original_annotations)
        return augmented_image

    def _inpaint_original_annotations(self, augmented_image: np.ndarray, original_image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
        for annotation in annotations:
            mask = rle_decode(annotation["mask"], (augmented_image.shape[0], augmented_image.shape[1]), np.uint8)[..., None]
            augmented_image = (original_image * mask) + (augmented_image * (1 - mask))
        return augmented_image

    def mix_annotations(
        self, original_image: np.ndarray, instances: np.ndarray, annotations: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        inpainted_annotations = []
        for annotation in annotations:
            if random.random() < self.p:
                mask = rle_decode(annotation["mask"], (instances.shape[0], instances.shape[1]), np.uint8)
                original_image = blend(original_image, instances, mask)
                inpainted_annotations.append(annotation)
        return original_image, inpainted_annotations


class MaskAugmenter(Augmenter):
    def __init__(self, generated_data_path: str, p: float = 0.5, blend: bool = False, rgb: bool = True):
        super().__init__(generated_data_path, p, rgb)
        self.blend = blend

    def _load_mask(self, mask_path: str) -> np.ndarray:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask

    def load_image_and_mask(self, base_path: str, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and mask from base_path and filename
        :param base_path: base path to the dataset
        :param image_path: path to the image
        :param mask_path: path to the mask
        :return: image as np.ndarray and mask as np.ndarray
        """
        metadata = self._get_metadata(image_path)
        original_image = cv2.imread(os.path.join(base_path, image_path))
        mask = self._load_mask(os.path.join(base_path, mask_path))
        if self.rgb:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        if metadata is None or len(metadata["annotations"]) == 0:
            return original_image, mask
        return self.augment_image_and_mask(image=original_image, mask=mask, metadata=metadata)

    def blend_data(
            self,
            original_image: np.ndarray,
            original_mask: np.ndarray,
            augmented_image: np.ndarray,
            augmented_mask: np.ndarray,
            annotations: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        final_image = original_image.copy()
        for annotation in annotations:
            if random.random() < self.p:
                mask = rle_decode(annotation["mask"], (original_image.shape[0], original_image.shape[1]), np.uint8)
                x1, y1, x2, y2 = self._rescale_bbox(annotation["bbox"], original_image, augmented_image)
                final_image = cv2.resize(final_image, (augmented_image.shape[1], augmented_image.shape[0]),
                                         interpolation=cv2.INTER_LANCZOS4)
                original_mask = cv2.resize(original_mask, (augmented_image.shape[1], augmented_image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                final_image = blend(final_image, augmented_image, mask)
                original_mask = augmented_mask
        return final_image, original_mask

    @staticmethod
    def _rescale_bbox(bbox: np.ndarray, image: np.ndarray, resize_image: np.ndarray) -> np.ndarray:
        bbox = bbox.copy()
        bbox[0] = bbox[0] * resize_image.shape[0] / image.shape[0]
        bbox[1] = bbox[1] * resize_image.shape[1] / image.shape[1]
        bbox[2] = bbox[2] * resize_image.shape[0] / image.shape[0]
        bbox[3] = bbox[3] * resize_image.shape[1] / image.shape[1]
        return np.array(bbox).astype("int")

    def _equal_shape(self, image_a: np.ndarray, image_b: np.ndarray) -> bool:
        return image_a.shape[:2] == image_b.shape[:2]

    def augment_image_and_mask(self, image: np.ndarray, mask: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        image_path = os.path.join(self.data_path, "images", metadata["generated_filename"])
        instances = cv2.imread(image_path)
        mask_path = image_path.replace("images", "masks").replace(".jpg", ".png")
        if not os.path.isfile(mask_path):
            return image, mask
        augmented_mask = self._load_mask(mask_path)
        if self.rgb:
            instances = cv2.cvtColor(instances, cv2.COLOR_BGR2RGB)
        annotations = metadata["annotations"]
        if metadata["num_annotations"] > 1:
            return (image, mask)
        return self.blend_data(original_image=image, original_mask=mask, augmented_image=instances, augmented_mask=augmented_mask, annotations=annotations)
