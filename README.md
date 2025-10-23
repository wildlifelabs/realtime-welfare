# Realtime Welfare

Docker framework for evaluating the WelfareObs realtime welfare package

Observing animal movements and behaviours is crucial for many scientific, welfare, and conservation purposes, but 
challenging to track particular individuals within a cohort under observation passively. This repo contains the code
for our research in which we tracked four individual giraffes (_Giraffa camelopardalis_) in an ex-situ habitat using 
a modular approach, tracking individual movement within an adaptable AI model framework. We used commodity security 
cameras to create locomotion ethograms, developed using a multiscopic approach, combining a state-of-the-art AI 
re-identification model, computer vision methods and low-power"edge AI" computing hardware. 

The techniques employed have demonstrated the feasibility of real-time animal re-identification and tracking using 
object detection, individual re-identification, and homography. Our software can be used for benchmarking real-time 
animal welfare, with edge computing a viable option for real-time AI processing. 

---
# Makefile Targets
The code was designed to run on an Ubuntu Linux environment and should work on most any Linux server with a CUDA GPU.
Note that configuration files may need to be adapted to support batch sizes for your GPU. 

```
help                         This help

setup-local                  Set up local environment (for using an IDE during development)
update-submodules            Update all submodules using GIT
init-submodules              Initial submodule setup

jetson-build                 Build Docker Environment (Jetson)
jetson-run-pipeline          Run the Jetson CUDA pipeline (Jetson Xavier headless)
jetson-run-pipeline-single   Run the Jetson CUDA pipeline (Jetson Xavier headless)

mac-build                    Build Docker Environment (Mac)
mac-run-pipeline             Run the CUDA pipeline (Mac headless)
mac-run-pipeline-single      Run the CUDA pipeline (Mac headless)

rpi-build                    Build Docker Environment (RPi)
rpi-run-pipeline             Run the pipeline (RPi headless)
rpi-run-pipeline-single      Run the pipeline (RPi headless)

cuda-jupyter                 Start Jupyter (for CUDA)
cuda-train-model             Train model based on the Config (Only works on X86 CUDA)
cuda-train-dino              Train Dino model based on the Config (Only works on X86 CUDA)
cuda-force-rebuild           Forced ReBuild Docker Environment
cuda-build                   Build for CUDA
cuda-run-pipeline            Run the CUDA pipeline (headless)
connect                      Connect to Container
tensorboard                  Start tensorboard on running instance (either cuda-jupyter or cuda-train-model)
check-cuda                   Check CUDA is working

setup-calibrate-cameras      Setup calibrate cameras application
run-calibrate-cameras        Run the calibrate cameras application (local machine venv)

setup-gcp-tool               Setup calibrate cameras application
run-camera-gcp-tool          Run the calibrate cameras application (local machine venv)

```

Just use `make` to list the targets (help)

* Make sure you update `Makefile.cfg` to point to your dataset for running the fine-tuning of the reidentification model (`make train-model`).
* See the `example_Makefile.cfg` for, you guessed it, and example of how to create this configuration file. It just needs to point to the data folder...

The inference code was designed to work on NVidia Jetson Xavier, Mac Mini M4 (any ARM based Mac should work fine), Raspberry PI (8GB minimum)
or a Linux server with a CUDA GPU (We used an RTX 6000 ADA)

---
# Requirements
* [Docker](https://docs.docker.com/engine/install/)
* [NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* Make and Git (use `sudo apt install build-essential`)

---
# Jupyter Notebooks


### `validation-Inference-MegaDescriptor-L-384.ipynb`
Validation workbook demonstrating inference as per the original MegaDescriptor paper with the Taronga dataset


### `validation-Training-MegaDescriptor-L-384.ipynb`
Validation workbook demonstrating training as per the original MegaDescriptor paper using the Taronga dataset


### `workbench-detectron.ipynb`
Evaluate the detectron model. Note that using the notebook requires having trained the model. This must be done using the `train-model` makefile target. 


### `workbench-pipeline.ipynb`
Run the pipeline model evaluation based on the provided configuration. The paper is designed to perform with the `FauxCamera` to ensure repeatable inference benchmarking results. You can modify configurations to evaluate or use realtime RTSP instead.


### `workbench-transforms.ipynb`
Custom TorchVision transform for padded scaling that maintains aspect ratios


### `workbench-dino.ipynb`
(Work in progress) evaluate a generic dino model


### `analysis-confusion.ipynb`  
Evaluate the performance of the reidentification model and render a confusion matrix. 


### `analysis-gcp.ipynb`  
Evaluate the performance of the GCP and gather metrics used in the tables.


### `analysis-location.ipynb`  
Perform the location tracking and render the heatmap ethogram based on the output CSV datafile of the pipeline.


### `analysis-performance.ipynb`
Analyse the output of the pipeline profiler CSV file. Not used in the paper directly, but part of the analysis toolkit.

---

# Configuration Files

### `/config.json`
Configuration for training the re-identification model.


### `/config-dino.cfg`
(Work in progress) Configuration for training the Dino-based re-identification model.


### `/Makefile.cfg`
Used by the Makefile for mapping the training dataset root location into the Docker container


### `/config`
The `/config` directory is used by the pipeline components. The pipeline 'orchestration' file is the starting file used by the pipeline. 
You can find examples of this in the Makefile targets. An example is below:
```bash
docker run --shm-size=1g -it --privileged --gpus all --rm -p 8888:8888 -p 8008:8008 -v ./:/project -v $(DATASET_ROOT):/project/data -v $(DATASET_ROOT)/hugging-face-cache:/root/.cache --name welfare-obs-instance welfare-obs python /project/run_pipeline.py -c non-rtsp-example.json
```
The `non-rtsp-example.json` is the orchestration file in this instance - and you can then explore the JSON
to understand how the rest of the pipeline works. 

Handlers are found in `welfareobs/welfareobs/handlers/` and you will see how handlers expect certain kinds of inputs 
which you can route from some other handler in the design of the orchestration file. Handlers have a `config` parameter 
and often these are a path to another configuration file. You can find examples of each of these files 
in the `/config` directory.

---
# TODO: document each of the handlers.

---
# Jetson Support
To evalue this project on Jetson Xavier, you need to have gone through the process of building PyTorch 2.7 for Jetson using the steps provided in the `pytorch-on-jetson` submodule. 

---

# Submodules

### Checking Out:

```bash
git clone --recurse-submodules https://github.com/wildlifelabs/realtime-welfare.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```


### Updating:

To update all submodules to their latest committed versions:

```bash
git submodule update --remote --merge
```

