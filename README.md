<div align="center">

# Dataset Enhancement with Instance-Level Augmentations

[![Paper](https://img.shields.io/badge/arXiv-2406.08249-brightgreen)](https://arxiv.org/abs/2406.08249)
[![Conference](https://img.shields.io/badge/ECCV-2024-blue)](https://eccv2024.ecva.net/)
[![Project WebPage](https://img.shields.io/badge/Project-webpage-%23fc4d5d)](https://www.robots.ox.ac.uk/~vgg/research/instance-augmentation/)

</div>

This is an official repository for the paper
```
Dataset Enhancement with Instance-Level Augmentations
Orest Kupyn, Christian Rupprecht
ECCV 2024
```

Instance Augmentation method augment images by redrawing individual objects in the scene retaining their original shape. This allows training with the unchanged class label (e.g. class, segmentation, detection, etc.). The generations are highly diverse and match the scene composition

Original             |  Augmented             |  Augmented
:-------------------------:|:-------------------------:|:-------------------------:
![](images/mp.jpeg)  |  ![](images/mp_1.png)  |  ![](images/mp_2.png)


# Augmented Datasets

This repository contains links to several augmented datasets that can be used for various computer vision tasks, such as object detection, instance segmentation, and saliency detection.

## Datasets

1. **COCO Augmented Car**:
   - Link: [https://thor.robots.ox.ac.uk/instance-augmentation/COCO_Augmented_Car.tar](https://thor.robots.ox.ac.uk/instance-augmentation/COCO_Augmented_Car.tar)
   - Description: An anonymized version of the COCO dataset, focusing on the "car" class.

2. **COCO Augmented People**:
   - Link: [https://thor.robots.ox.ac.uk/instance-augmentation/COCO_Augmented_People.tar](https://thor.robots.ox.ac.uk/instance-augmentation/COCO_Augmented_People.tar)
   - Description: An anonymized verison of the COCO dataset, focusing on the "person" class.

3. **COCO Augmented**:
   - Link: [https://thor.robots.ox.ac.uk/instance-augmentation/COCO_Augmented.tar](https://thor.robots.ox.ac.uk/instance-augmentation/COCO_Augmented.tar)
   - Description: An augmented version of the entire COCO dataset.

4. **DUTS Augmented**:
   - Link: [https://thor.robots.ox.ac.uk/instance-augmentation/DUTS_Augmented.tar](https://thor.robots.ox.ac.uk/instance-augmentation/DUTS_Augmented.tar)
   - Description: An augmented version of the DUTS dataset, which is commonly used for saliency detection.

5. **DUTS SDXL (Experimental)**:
   - Link: [https://thor.robots.ox.ac.uk/instance-augmentation/DUTS_SDXL.tar](https://thor.robots.ox.ac.uk/instance-augmentation/DUTS_SDXL.tar)
   - Description: A larger, augmented version of the DUTS dataset.

6. **SHA512 Checksums**:
   - Link: [https://thor.robots.ox.ac.uk/instance-augmentation/SHA512SUMS](https://thor.robots.ox.ac.uk/instance-augmentation/SHA512SUMS)
   - Description: A file containing the SHA512 checksums for the above augmented datasets, which can be used to verify the integrity of the downloaded files.

Please note that these augmented datasets are provided for research purposes. If you plan to use these datasets in your projects, make sure to follow the appropriate licensing and citation requirements.

## Installation

The code uses **Python 3.8**.

#### Create a Conda virtual environment and Install The Package:

Make sure you have Conda installed.

```bash
make env
```

#### Run Test for the Package:

```bash
make pytest
```

#### Run on a folder of images:

An example is available in tests/test_pipeline.py - test_end_to_end

To predict instance masks:
```python
from instance_augmentation.pseudolabel_dataset import create_annotations

create_annotations("path_to_image_folder", "path_to_save_results", dataset_type="custom", class_names=["dog", "cat", "any_other_classes"])
```

To generate augmented images:
```python
from instance_augmentation.pipeline.dataset_generator import DatasetGenerator
from instance_augmentation.pipeline.readers import CustomDatasetReader

reader = CustomDatasetReader("path_to_image_folder", {}, "path_to_save_results/annotations.json")
dataset_generator = DatasetGenerator.from_params(
        dataset_reader=reader,
        save_folder="path_to_save_results",
        preprocessing="resize",
        target_image_size=1024,
        base_inpainting_model="SG161222/RealVisXL_V3.0",
        generator="inpaint_sdxl_adapter",
        num_samples=1,
        num_inference_steps=20,
        control_methods=["t2i_depth", "t2i_sketch"],
        control_weights=[0.9, 0.5],
    )
    dataset_generator.run()
```

To apply augmentations:
```python
import os
import cv2
import glob
from instance_augmentation.augment import Augmenter

augmenter = Augmenter("path_to_save_results", p=1.0)
for image_path in glob.glob("path_to_image_folder/*"):
    image_name = os.path.split(image_path)[1]
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    augmented_image = augmenter.augment_image(original_image, image_name)

```

## Citation

If you use the the method or this code - implicitly or explicitly - for your research projects, please cite the following paper:

```
@article{kupyn2024dataset,
    title = {Dataset Enhancement with Instance-Level Augmentations},
    author = {Kupyn, Orest and Rupprecht, Christian},
    journal = {arXiv preprint arXiv:2406.08249},
    year = {2024}
  }

```
