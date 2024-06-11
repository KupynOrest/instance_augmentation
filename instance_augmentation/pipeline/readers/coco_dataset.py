import os
import logging
from typing import List, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as mutils

from instance_augmentation.pipeline.image_captioner import ImageCaptioner
from instance_augmentation.pipeline.entities import Annotation, DatasetItem
from instance_augmentation.pipeline.readers.base_reader import BaseDatasetReader
from instance_augmentation.pipeline.diffusion_pipeline import DEPTH_PROCESSOR

logger = logging.Logger(__name__)
MAX_OBJECTS = 10
MIN_AREA = 1000


def annotation2mask(segmentation, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """

    def _ann_to_rle(segmentation, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if type(segmentation) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mutils.frPyObjects(segmentation, height, width)
            rle = mutils.merge(rles)
        elif type(segmentation["counts"]) == list:
            # uncompressed RLE
            rle = mutils.frPyObjects(segmentation, height, width)
        else:
            # rle
            rle = segmentation
        return rle

    rle = _ann_to_rle(segmentation, height, width)
    mask = mutils.decode(rle)
    return mask


class COCODatasetReader(BaseDatasetReader):
    def __init__(self, data_path: str, dataset_config: Dict[str, Any]):
        self.dataset_config = dataset_config
        self.image_captioner = ImageCaptioner("what is the person doing?")
        self.categories = dataset_config.get("categories", None)
        self.depth_preprocessor = DEPTH_PROCESSOR
        self.min_area = dataset_config.get("min_area", MIN_AREA)
        self.max_objects = dataset_config.get("max_objects", MAX_OBJECTS)
        self.save_folder = dataset_config.get("save_folder", None)
        super().__init__(data_path)

    def _parse_n_largest_objects(self, annotations: List[Annotation]) -> List[Annotation]:
        if self.max_objects <= 0:
            return annotations
        annotations = sorted(annotations, key=lambda x: x.area, reverse=True)
        for annotation in annotations[self.max_objects:]:
            annotation.keep_original = True
        return annotations

    def _should_generate(self, label: str, area: int):
        if self.categories is not None and label not in self.categories:
            return False
        if area < self.min_area:
            return False
        return True

    def _parse_anno(self, coco, ann_obj, image_path) -> List[Annotation]:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        annotations = []
        depth_image = np.array(self.depth_preprocessor(image))
        for annotation in ann_obj:
            x, y, w, h = list(map(int, annotation["bbox"]))
            mask = annotation2mask(annotation["segmentation"], height, width)
            label = coco.loadCats(annotation["category_id"])[0]["name"]
            keep_original = not self._should_generate(label, annotation["area"])
            # Prompting for small objects
            if annotation["area"] < MIN_AREA:
                if label == "person":
                    label = "person++, full body"
                label = f"{label}, 8k, hdr"
            annotations.append(
                Annotation(
                    mean_depth=float(np.mean(depth_image[np.where(mask == 1)])),
                    category_id=annotation["category_id"],
                    label=label,
                    bbox=[x, y, x + w, y + h],
                    mask=mask,
                    small_object=annotation["area"] < MIN_AREA,
                    keep_original=keep_original,
                    area=annotation["area"],
                )
            )
        annotations = self._parse_n_largest_objects(annotations)
        annotations = sorted(annotations, key=lambda x: x.mean_depth)
        return annotations

    @staticmethod
    def _get_folder_path(split: str):
        if split == "train":
            return "train2017"
        return "val2017"

    def _read_subset(self, annotation_path: str, split: str) -> List[DatasetItem]:
        dataset = []
        coco = COCO(annotation_path)
        img_ids = coco.getImgIds()

        for index in tqdm(range(len(img_ids))):
            img_obj = coco.loadImgs(img_ids[index])[0]
            image_id = img_obj["file_name"]
            image_path = os.path.join(self.data_path, self._get_folder_path(split), image_id)
            ann_ids = coco.getAnnIds(imgIds=img_ids[index], iscrowd=None)
            ann_obj = coco.loadAnns(ann_ids)
            annotations = self._parse_anno(coco, ann_obj, image_path)
            dataset.append(
                DatasetItem(
                    base_path=os.path.join(self.data_path, self._get_folder_path(split)),
                    image_path=image_id,
                    annotations=annotations,
                    split=split,
                    index=index,
                )
            )
        return dataset

    def read_dataset(self) -> List[DatasetItem]:
        dataset_folder = self.data_path
        train_subset = self._read_subset(
            os.path.join(dataset_folder, "annotations", "instances_train2017.json"), "train"
        )
        return train_subset
