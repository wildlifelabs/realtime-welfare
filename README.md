# Realtime Welfare

Docker framework for evaluating the WelfareObs realtime welfare package

## Makefile Targets
```
help                      This help

setup-local               Set up local environment (for using an IDE during development)
update-submodules         Update all submodules using GIT
init-submodules           Initial submodule setup

jetson-build              Build Docker Environment (Jetson)
jetson-run-pipeline       Run the Jetson CUDA pipeline (Jetson Xavier headless)

mac-build                 Build Docker Environment (Mac)
mac-run-pipeline          Run the CUDA pipeline (Mac headless)

rpi-build                 Build Docker Environment (RPi)
rpi-run-pipeline          Run the pipeline (RPi headless)

cuda-jupyter              Start Jupyter (for CUDA)
cuda-force-rebuild        Forced ReBuild Docker Environment
cuda-build                Build for CUDA
cuda-run-pipeline         Run the CUDA pipeline (headless)

connect                   Connect to Container
train-model               Train the models based on config (Only works on X86 CUDA)
check-cuda                Check CUDA is working

setup-calibrate-cameras   Setup calibrate cameras application
run-calibrate-cameras     Run the calibrate cameras application (local machine venv)

setup-gcp-tool            Setup calibrate cameras application
run-camera-gcp-tool       Run the calibrate cameras application (local machine venv)

render-code               Print the codebase to PDF
render-tree               render code heirarchy

```

Just use `make` to list the targets (help)

* Make sure you update `Makefile.cfg` to point to your dataset for running the fine-tuning of the reidentification model (`make train-model`).
* See the `example_Makefile.cfg` for, you guessed it, and example of how to create this configuration file. It just needs to point to the data folder...


---

## Jupyter Notebooks


### Detectron
`detectron.ipynb` will evaluate the detectron model. Note that using the notebook requires having trained the model. This must be done using the `train-model` makefile target. 


### Pipeline
`pipeline.ipynb` will run the pipeline model evaluation based on the provided configuration. The paper is designed to perform with the `FauxCamera` to ensure repeatable inference benchmarking results.


### Graphs
`graphs.ipynb` will generate graphs used in the paper, however this only works with the outputs of the pipeline.


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

