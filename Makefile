SHELL := /bin/bash
include Makefile.cfg
# Description: Self documenting Makefile that has all the targets...
#
# Copyright (C) 2025 J.Cincotta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#
export TORCH_CUDA_ARCH_LIST="12.4.1"
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda

help: ## This help
	@echo "Welfare Obs Project"
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

update-submodules: ## Update all submodules using GIT
	cd wildlife-datasets;git pull origin main
	cd wildlife-tools;git pull origin main
	git submodule update --remote --merge

init-submodules: ## Initial submodule setup
	git submodule update --init --recursive

force-rebuild: ## Forced ReBuild Docker Environment
	docker build --no-cache -t welfare-obs -f Dockerfile .

build: ## Build Docker Environment
	docker build -t welfare-obs -f Dockerfile .

jupyter: build ## Start Jupyter
	echo $(DATASET_ROOT)
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data --name welfare-obs-instance welfare-obs /script/jupyter.sh /project

connect: ## Connect to CUDA Container
	docker exec -it welfare-obs-instance bash

train-model: ## Train the models based on config
	docker exec -it welfare-obs-instance /project/bin/py.sh /project/train_model.py

check-cuda: ## Check CUDA is working
	docker exec -it welfare-obs-instance /project/bin/py.sh /project/check_cuda.py

setup-local: ## Set up local environment
	@python3.11 -m venv venv
	@source venv/bin/activate;python -m pip install --upgrade pip
	@source venv/bin/activate;python -m pip install flake8 pytest
	@source venv/bin/activate;if [ -f bin/requirements.txt ]; then pip install -r bin/requirements.txt; fi
	@source venv/bin/activate;if [ -f bin/requirements2.txt ]; then pip install -r bin/requirements2.txt; fi

setup-calibrate-cameras: ## Setup calibrate cameras application
	$(MAKE) -C calibrate-camera setup

calibrate-cameras: ## Run the calibrate cameras application (local machine venv)
	$(MAKE) -C calibrate-camera calibrate-camera

