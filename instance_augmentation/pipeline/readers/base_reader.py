from abc import ABC, abstractmethod
from typing import List, Optional

from instance_augmentation.pipeline.entities import DatasetItem


class BaseDatasetReader(ABC):
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self._index = 0
        self.dataset = self.read_dataset()

    def _assign_splits(self, dataset: List[DatasetItem]) -> List[DatasetItem]:
        pass

    @abstractmethod
    def read_dataset(self) -> List[DatasetItem]:
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.dataset):
            item = self.dataset[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration
