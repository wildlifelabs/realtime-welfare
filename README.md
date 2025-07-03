# Realtime Welfare

Docker framework for evaluating the WelfareObs realtime welfare package

Observing animal movements and behaviours is crucial for many scientific, welfare, and conservation purposes, but challenging to track particular individuals within a cohort under observation passively. I tracked four individual giraffes (_Giraffa camelopardalis_) in an ex-situ habitat using a modular approach, tracking individual movement within an adaptable AI model framework. I used commodity security cameras to create locomotion ethograms, developed using a multiscopic approach, combining a state-of-the-art AI re-identification model, computer vision methods and low-power “edge AI” computing hardware. I have provided a publicly available codebase in Python on GitHub (https://github.com/wildlifelabs/realtime-welfare). The techniques employed have demonstrated the feasibility of real-time animal re-identification and tracking using object detection, individual re-identification, and homography. My software can be used for benchmarking real-time animal welfare, with edge computing a viable option for real-time AI processing. I highlighted the potential for model optimisations to improve performance, as well as identifying limitations of my approach, including the effects of occlusion and non-planar surfaces. My approach provides considerable promise for more effective, efficient, and scalable animal re-identification and tracking solutions, ultimately supporting continuous animal welfare and conservation objectives for zoos.


## Makefile Targets
```
help                        This help

setup-local                 Set up local environment (for using an IDE during development)

update-submodules           Update all submodules using GIT
init-submodules             Initial submodule setup

jetson-build                Build Docker Environment (Jetson)
jetson-run-pipeline         Run the Jetson CUDA pipeline (Jetson Xavier headless)
jetson-run-pipeline-single  Run the Jetson CUDA pipeline (Jetson Xavier headless)

mac-build                   Build Docker Environment (Mac)
mac-run-pipeline            Run the CUDA pipeline (Mac headless)
mac-run-pipeline-single     Run the CUDA pipeline (Mac headless)

rpi-build                   Build Docker Environment (RPi)
rpi-run-pipeline            Run the pipeline (RPi headless)
rpi-run-pipeline-single     Run the pipeline (RPi headless)

cuda-jupyter                Start Jupyter (for CUDA)
cuda-force-rebuild          Forced ReBuild Docker Environment
cuda-build                  Build for CUDA
cuda-run-pipeline           Run the CUDA pipeline (headless)

connect                     Connect to Container

train-model                 Train the models based on config (Only works on X86 CUDA)
check-cuda                  Check CUDA is working

setup-calibrate-cameras     Setup calibrate cameras application
run-calibrate-cameras       Run the calibrate cameras application (local machine venv)

setup-gcp-tool              Setup calibrate cameras application
run-camera-gcp-tool         Run the calibrate cameras application (local machine venv)

render-code                 Print the codebase to PDF
render-tree                 render code heirarchy

```

Just use `make` to list the targets (help)

* Make sure you update `Makefile.cfg` to point to your dataset for running the fine-tuning of the reidentification model (`make train-model`).
* See the `example_Makefile.cfg` for, you guessed it, and example of how to create this configuration file. It just needs to point to the data folder...

---

## Jupyter Notebooks



### detectron.ipynb
Evaluate the detectron model. Note that using the notebook requires having trained the model. This must be done using the `train-model` makefile target. 


### pipeline.ipynb
Run the pipeline model evaluation based on the provided configuration. The paper is designed to perform with the `FauxCamera` to ensure repeatable inference benchmarking results. You can modify configurations to evaluate or use realtime RTSP instead.


### confusion.ipynb  
Evaluate the performance of the reidentification model and render a confusion matrix. 


### gcp.ipynb  
Evaluate the performance of the GCP and gather metrics used in the tables.


### location.ipynb  
Perform the location tracking and render the heatmap ethogram based on the output CSV datafile of the pipeline.


### performance.ipynb
Analyse the output of the pipeline profiler CSV file. Not used in the paper directly, but part of the analysis toolkit.


---

## Jetson Support
To evalue this project on Jetson Xavier, you need to have gone through the process of building PyTorch 2.7 for Jetson using the steps provided in the `pytorch-on-jetson` submodule. 

---

## Checking Out:

```bash
git clone --recurse-submodules <repo_url>
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```


## Updating:

To update all submodules to their latest committed versions:

```bash
git submodule update --remote --merge
```

If you want the latest upstream commits:

```bash
cd <module path>
git pull origin main
```

Then commit the updated submodule reference:

```bash
cd ..
git add <module path>
git commit -m "Updated submodule library"
```

