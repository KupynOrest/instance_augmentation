from abc import ABC, abstractmethod
from typing import List, Optional
import os

from instance_augmentation.pipeline.entities import DatasetItem, GeneratedResult


class BaseConverter(ABC):
    def __init__(self, save_folder: str, folders: Optional[List[str]] = None):
        self.save_folder = save_folder
        if folders is not None:
            self._prepare_folders(folders)
        self._index = 0

    def _prepare_folders(self, folders):
        for split in ["train", "valid", "test"]:
            for folder in ["real", "generated", "real_generated"]:
                for data_folder in folders:
                    os.makedirs(os.path.join(self.save_folder, folder, split, data_folder), exist_ok=True)

    @abstractmethod
    def save_annotation(self, generation: GeneratedResult, annotation: DatasetItem, index: int = 0):
        pass
