SHELL := /bin/bash
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

setup: ## Install the virtual environment (assumes Python 3.10 is installed)
	@python3.10 -m venv venv
	@source venv/bin/activate;python -m pip install --upgrade pip
	@source venv/bin/activate;python -m pip install flake8 pytest
	@source venv/bin/activate;if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

calibrate-camera: ## Calibrate a camera image
	@source venv/bin/activate;python calibrate_camera.py

