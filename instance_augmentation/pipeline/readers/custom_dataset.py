import logging
from typing import List, Dict, Any

from instance_augmentation.pipeline.entities import DatasetItem
from instance_augmentation.pipeline.readers.coco_dataset import COCODatasetReader

logger = logging.Logger(__name__)
MAX_OBJECTS = 10
MIN_AREA = 1000


class CustomDatasetReader(COCODatasetReader):
    def __init__(self, data_path: str, dataset_config: Dict[str, Any], json_path: str):
        self.json_path = json_path
        super().__init__(data_path, dataset_config)

    @staticmethod
    def _get_folder_path(split: str):
        return ""

    def read_dataset(self) -> List[DatasetItem]:
        dataset = self._read_subset(self.json_path, "train")
        return dataset
