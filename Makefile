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
	@echo "⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖⁖"
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

#### SETUP ####

setup-local: ## Set up local environment (for using an IDE during development)
	@python3.13 -m venv venv
	@source venv/bin/activate;python -m pip install --upgrade pip
	@source venv/bin/activate;python -m pip install flake8 pytest
	@source venv/bin/activate;if [ -f bin/requirements.txt ]; then pip install -r bin/requirements.txt; fi
	@source venv/bin/activate;if [ -f bin/requirements2.txt ]; then pip install -r bin/requirements2.txt; fi
	@source venv/bin/activate;if [ -f bin/jupyter.txt ]; then pip install -r bin/jupyter.txt; fi

update-submodules: ## Update all submodules using GIT
	cd wildlife-datasets;git pull origin main
	cd wildlife-tools;git pull origin main
	git submodule update --remote --merge

init-submodules: ## Initial submodule setup
	git submodule update --init --recursive

#### NVIDIA JETSON ####

jetson-build: ## Build Docker Environment (Jetson)
	cp -r /usr/local/cuda-12.2 ./cuda-12.2
	docker build --progress=plain -t welfare-obs -f JetsonDockerfile .

jetson-run-pipeline: ## Run the Jetson CUDA pipeline (Jetson Xavier headless)
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --runtime nvidia --privileged --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/run_pipeline.py -c jetson-non-rtsp-test.json

jetson-run-pipeline-single: ## Run the Jetson CUDA pipeline (Jetson Xavier headless)
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --runtime nvidia --privileged --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/run_pipeline.py -c jetson-non-rtsp-test-single.json

#### MACINTOSH OSX ####

mac-build: ## Build Docker Environment (Mac)
	docker build -t welfare-obs -f MacDockerfile .

mac-run-pipeline: ## Run the CUDA pipeline (Mac headless)
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --privileged --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/run_pipeline.py -c mac-non-rtsp-test.json

mac-run-pipeline-single: ## Run the CUDA pipeline (Mac headless)
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --privileged --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/run_pipeline.py -c mac-non-rtsp-test-single.json

#### RASPBERRY PI ####

rpi-build: ## Build Docker Environment (RPi)
	docker build -t welfare-obs -f RpiDockerfile .

rpi-run-pipeline: ## Run the pipeline (RPi headless)
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --privileged --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/run_pipeline.py -c rpi-non-rtsp-test.json

rpi-run-pipeline-single: ## Run the pipeline (RPi headless)
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --privileged --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/run_pipeline.py -c rpi-non-rtsp-test-single.json

#### LINUX X64 CUDA ####

cuda-jupyter: cuda-build ## Start Jupyter (for CUDA)
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs /script/jupyter.sh /project

cuda-train-model: cuda-build ## Train model based on the Config (Only works on X86 CUDA) 
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/train_model.py

cuda-train-dino: cuda-build ## Train Dino model based on the Config (Only works on X86 CUDA) 
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/train_dino.py

cuda-force-rebuild: ## Forced ReBuild Docker Environment
	docker build --no-cache -t welfare-obs -f Dockerfile .

cuda-build: ## Build for CUDA
	docker build -t welfare-obs -f Dockerfile .

cuda-run-pipeline: ## Run the CUDA pipeline (headless)
	echo $(DATASET_ROOT)
	mkdir -p $(DATASET_ROOT)/hugging-face-cache
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/run_pipeline.py -c non-rtsp-example.json
	# old method using Jupyter headless: /script/run_ipynb.sh /project pipeline.ipynb

#### COMMON CONNECTION TOOLS ####

connect: ## Connect to Container
	docker exec -it welfare-obs-instance bash

tensorboard: ## Start tensorboard on running instance (either cuda-jupyter or cuda-train-model)
	docker exec -it welfare-obs-instance tensorboard --logdir /project/data --port 8008 --host 0.0.0.0

# train-model: ## Train the models based on config (Only works on X86 CUDA)
# 	docker exec -it welfare-obs-instance /project/bin/py.sh /project/train_model.py

check-cuda: ## Check CUDA is working
	docker exec -it welfare-obs-instance /project/bin/py.sh /project/check_cuda.py

#### LOCAL CALIBRATION TOOLS WITH USER INTERFACES ####

setup-calibrate-cameras: ## Setup calibrate cameras application
	$(MAKE) -C calibrate-camera-tool setup

run-calibrate-cameras: ## Run the calibrate cameras application (local machine venv)
	$(MAKE) -C calibrate-camera-tool run

setup-gcp-tool: ## Setup calibrate cameras application
	$(MAKE) -C camera-gcp-tool setup

run-camera-gcp-tool: ## Run the calibrate cameras application (local machine venv)
	$(MAKE) -C camera-gcp-tool run
