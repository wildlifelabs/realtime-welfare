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
	docker build -t welfare-obs -f JetsonDockerfile .

jetson-run-pipeline: ## Run the Jetson CUDA pipeline (Jetson Xavier headless)
	echo $(DATASET_ROOT)
	docker run --shm-size=1g -it --runtime nvidia --privileged --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data --name welfare-obs-instance welfare-obs /script/run_ipynb.sh /project pipeline.ipynb

#### MACINTOSH OSX ####

mac-build: ## Build Docker Environment (Mac)
	docker build -t welfare-obs -f MacDockerfile .

mac-run-pipeline: ## Run the CUDA pipeline (Mac headless)
	echo $(DATASET_ROOT)
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data --name welfare-obs-instance welfare-obs /script/run_ipynb.sh /project pipeline.ipynb

#### RASPBERRY PI ####

rpi-build: ## Build Docker Environment (RPi)
	docker build -t welfare-obs -f RpiDockerfile .

rpi-run-pipeline: ## Run the pipeline (RPi headless)
	echo $(DATASET_ROOT)
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data --name welfare-obs-instance welfare-obs /script/run_ipynb.sh /project pipeline.ipynb

#### LINUX X64 CUDA ####

cuda-jupyter: cuda-build ## Start Jupyter (for CUDA)
	echo $(DATASET_ROOT)
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data --name welfare-obs-instance welfare-obs /script/jupyter.sh /project

cuda-force-rebuild: ## Forced ReBuild Docker Environment
	docker build --no-cache -t welfare-obs -f Dockerfile .

cuda-build: ## Build for CUDA
	docker build -t welfare-obs -f Dockerfile .

cuda-run-pipeline: ## Run the CUDA pipeline (headless)
	echo $(DATASET_ROOT)
	docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data --name welfare-obs-instance welfare-obs /script/run_ipynb.sh /project pipeline.ipynb

#### COMMON CONNECTION TOOLS ####

connect: ## Connect to Container
	docker exec -it welfare-obs-instance bash

train-model: ## Train the models based on config (Only works on X86 CUDA)
	docker exec -it welfare-obs-instance /project/bin/py.sh /project/train_model.py

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

render-code: ## Print the codebase to PDF
	# Requires `enscript` and `ghostscript` to be installed
	@source venv/bin/activate;python ./bin/render_source.py -s ".py" -i welfareobs -o ./code-output/welfareobs -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s ".json" -i config -o ./code-output/config -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s ".py" -i calibrate-camera-tool -o ./code-output/calibrate-camera-tool -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s ".py" -i camera-gcp-tool -o ./code-output/camera-gcp-tool -t "html" --flatten
	# Pip
	@source venv/bin/activate;python ./bin/render_source.py -s ".txt" -i ./bin -o ./code-output/pip -t "html" --flatten
	# Root level utilities
	@source venv/bin/activate;python ./bin/render_source.py -s ".py" -i ./train_model.py -o ./code-output/training --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s ".json" -i ./config.json -o ./code-output/training --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s ".py" -i ./check_cuda.py -o ./code-output/bin --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s ".py" -i ./bin/render_source.py -o ./code-output/bin --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s ".sh" -i ./bin -o ./code-output/bin -t "html" --flatten
	# Docker Files
	@source venv/bin/activate;python ./bin/render_source.py -s "Dockerfile" -i ./Dockerfile -o ./code-output/docker --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s "Dockerfile" -i ./RpiDockerfile -o ./code-output/docker --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s "Dockerfile" -i ./JetsonDockerfile -o ./code-output/docker --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s "Dockerfile" -i ./MacDockerfile -o ./code-output/docker --single -t "html" --flatten
	# Makefiles
	@source venv/bin/activate;python ./bin/render_source.py -s "Makefile" -i ./Makefile -o ./code-output/main-project --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s "Makefile" -i ./calibrate-camera-tool/Makefile -o ./code-output/calibrate-camera-tool --single -t "html" --flatten
	@source venv/bin/activate;python ./bin/render_source.py -s "Makefile" -i ./camera-gcp-tool/Makefile -o ./code-output/camera-gcp-tool --single -t "html" --flatten

render-tree: ## render code heirarchy
	# Requires `tree` to be installed.
	@tree --gitignore -C -H "" --nolinks -o code-output/tree.html
