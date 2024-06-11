from .duts_dataset import DUTSDatasetReader
from .coco_dataset import COCODatasetReader
from .base_reader import BaseDatasetReader
from .custom_dataset import CustomDatasetReader

__all__ = ["DUTSDatasetReader", "COCODatasetReader", "BaseDatasetReader", "CustomDatasetReader"]
