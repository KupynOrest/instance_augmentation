import os

import cv2
from fire import Fire
from glob import glob
from tqdm import tqdm
from pytorch_toolbelt.utils import vstack_header, hstack_autopad

from instance_augmentation.augment import Augmenter


def vizualize_images(gen_voc_path: str, voc_base_path: str, save_dir: str, save_comparison: bool = False) -> None:
    os.makedirs(save_dir, exist_ok=True)
    images = glob(os.path.join(voc_base_path, "JPEGImages", "*"))
    augmenter = Augmenter(gen_voc_path, p=1.0)
    image_count = 0
    for image_path in tqdm(images):
        _, image_name = os.path.split(image_path)
        original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        augmented_image = augmenter.augment_image(original_image, image_name)
        if (original_image == augmented_image).all():
            continue
        _, filename = os.path.split(image_path)
        if save_comparison:
            comparison_image = hstack_autopad(
                [
                    vstack_header(original_image, "Original Image"),
                    vstack_header(augmented_image, "Augmented Image")
                ]
            )
            cv2.imwrite(os.path.join(save_dir, filename), cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(os.path.join(save_dir, filename), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
        image_count += 1
        if image_count == 1000:
            break


if __name__ == "__main__":
    Fire(vizualize_images)
