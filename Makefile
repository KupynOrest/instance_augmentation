# Need to specify bash in order for conda activate to work.
.ONESHELL:
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


env:
	conda remove --name instance_aug --all;
	conda env create -f environment.yml
	$(CONDA_ACTIVATE) instance_aug
	pip install -e .
	pip install pytest pytest-cov
	git clone https://github.com/IDEA-Research/GroundingDINO.git
	cd GroundingDINO
	git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
	pip install -q -e .
	cd ..
	pip install 'git+https://github.com/facebookresearch/segment-anything.git'
	cd instance_augmentation/models
	wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
	wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

pytest:
	$(CONDA_ACTIVATE) instance_aug
	cd instance_augmentation/tests && PYTHONPATH=.. pytest . && cd ..
