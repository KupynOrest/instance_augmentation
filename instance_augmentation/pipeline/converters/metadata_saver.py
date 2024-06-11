import os
from os import environ

import cv2
from tinydb import TinyDB

from instance_augmentation.pipeline.entities import DatasetItem, GeneratedResult
from instance_augmentation.pipeline.converters import BaseConverter


class MetadataSaver(BaseConverter):
    def __init__(self, save_folder: str):
        super().__init__(save_folder)
        os.makedirs(os.path.join(save_folder, "images"), exist_ok=True)
        filename = "database.json"
        if "SLURM_ARRAY_TASK_ID" in environ:
            task_id = int(environ["SLURM_ARRAY_TASK_ID"])
            filename = f"database_{task_id}.json"
        self.database = TinyDB(os.path.join(save_folder, filename))

    def _get_sample_dict(self, generation: GeneratedResult, data_sample: DatasetItem, index: int = 0):
        image_index = data_sample.index if data_sample.index is not None else self._index
        annotations = (
            generation.valid_annotations if generation.valid_annotations is not None else data_sample.annotations
        )
        filename = f"{data_sample.split}_{image_index}_{index}.jpg"
        return {
            "generated_filename": filename,
            "num_annotations": len([x for x in data_sample.annotations if not x.keep_original]),
            "annotations": [anno.to_dict() for anno in annotations],
            "original_annotations": [anno.to_dict() for anno in generation.original_annotations]
            if generation.original_annotations is not None
            else [],
            "original_filename": data_sample.image_path,
            "split": data_sample.split,
        }

    def save_annotation(self, generation: GeneratedResult, data_sample: DatasetItem, index: int = 0):
        sample_metadata = self._get_sample_dict(generation, data_sample, index)
        self._index += 1
        self.database.insert(sample_metadata)
        cv2.imwrite(
            os.path.join(self.save_folder, "images", sample_metadata["generated_filename"]),
            cv2.cvtColor(generation.rendered_instances, cv2.COLOR_RGB2BGR),
        )
