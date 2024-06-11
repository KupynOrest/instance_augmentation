from fire import Fire

from instance_augmentation.pipeline.dataset_generator import DatasetGenerator
from instance_augmentation.pipeline.readers import DUTSDatasetReader


def generate(duts_path: str, save_folder: str, samples_per_real: int = 3) -> None:
    reader = DUTSDatasetReader(duts_path)
    dataset_generator = DatasetGenerator.from_params(
        dataset_reader=reader,
        save_folder=save_folder,
        preprocessing="resize",
        target_image_size=1280,
        base_inpainting_model="SG161222/RealVisXL_V3.0",
        generator="inpaint_sdxl_adapter",
        num_samples=samples_per_real,
        num_inference_steps=50,
        control_methods=["t2i_depth", "t2i_sketch"],
        control_weights=[0.9, 0.5],
    )
    dataset_generator.run()


if __name__ == "__main__":
    Fire(generate)
