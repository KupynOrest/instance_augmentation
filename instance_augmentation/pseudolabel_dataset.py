import os
import json
from typing import Dict, Any, Optional, List

import cv2
from pycocotools import mask as mask_util
import numpy as np
from glob import glob
from tqdm import tqdm
from fire import Fire

from pipeline.instance_segmentation import InstanceSegmentationPredictor


class AnnotationPredictor:
    def __init__(self, images_folder: str, class_names: Optional[List[str]] = None):
        self.images_folder = images_folder
        self.predictor = InstanceSegmentationPredictor(class_names=class_names)
        self.coco_format = {
            "info": {},
            "licenses": [],
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(class_names)],
            "images": [],
            "annotations": []
        }
        self.image_id = 1
        self.annotation_id = 1

    def get_images(self):
        images = glob(f"{self.images_folder}/*")
        return images

    def binary_mask_to_coco(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Convert a binary mask in the form of a NumPy array to COCO annotation mask format.

        Parameters:
            mask (ndarray): Binary mask in the form of a NumPy array.

        Returns:
            dict: COCO annotation mask in the form of a dictionary with 'counts' and 'size' keys.
        """
        # Convert the binary mask to a COCO RLE format mask
        mask = np.asfortranarray(mask)
        rle = mask_util.encode(mask)

        # Convert the RLE to a format suitable for COCO annotations
        coco_rle = {
            'counts': rle['counts'].decode('utf-8'),
            'size': list(mask.shape)  # height, width
        }
        return coco_rle

    def get_category_id(self, class_name):
        for category in self.coco_format["categories"]:
            if category["name"] == class_name or category["name"].split(" ")[0] == class_name:
                return category["id"]
        return None

    def get_annotations(self):
        images = self.get_images()
        for image_path in tqdm(images):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            annotation_data = self.predictor(image)
            _, filename = os.path.split(image_path)
            image_info = {
                "id": self.image_id,
                "file_name": filename,
                "width": image.shape[1],
                "height": image.shape[0],
            }
            self.coco_format["images"].append(image_info)

            for bbox, label, mask in zip(annotation_data.bboxes, annotation_data.classes, annotation_data.masks):
                x, y, x1, y1 = bbox.astype(np.int32).tolist()
                mask = mask.astype(np.uint8)
                category_id = self.get_category_id(label)
                if category_id is not None:
                    annotation_info = {
                        "id": self.annotation_id,
                        "image_id": self.image_id,
                        "category_id": self.get_category_id(label),
                        "bbox": [x, y, x1 - x, y1 - y],
                        "area": int(np.sum(mask)),
                        "iscrowd": 0,
                        "segmentation": self.binary_mask_to_coco(mask),
                        "ignore": 0
                    }
                    self.coco_format["annotations"].append(annotation_info)
                    self.annotation_id += 1
            self.image_id += 1
        return self.coco_format


class VOCAnnotationPredictor(AnnotationPredictor):
    def __init__(self, images_folder: str, file_list: str):
        pascal_voc_classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "monitor",
        ]
        super().__init__(images_folder=images_folder, class_names=pascal_voc_classes)
        with open(file_list, "r") as f:
            self.image_list = f.read().split("\n")

    def get_images(self):
        images = glob(f"{self.images_folder}/*")
        images = [image for image in images if os.path.basename(image).split(".")[0] in self.image_list]
        return images


def create_annotations(images_folder: str, output_folder: str, dataset_type: str = "voc", class_names: Optional[List[str]] = None):
    """
    Create annotations for a dataset
    :param images_folder: folder containing images
    :param output_file: output file
    :param dataset_type: type of dataset (voc or custom)
    """
    os.makedirs(output_folder, exist_ok=True)
    if dataset_type == "voc":
        predictor = VOCAnnotationPredictor(images_folder, "data/train_aug.txt")
    else:
        predictor = AnnotationPredictor(images_folder, class_names=class_names)
    annotations = predictor.get_annotations()
    with open(f"{output_folder}/annotations.json", "w") as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    Fire(create_annotations)
