# Realtime Welfare

Docker framework for evaluating the WelfareObs realtime welfare package

## Makefile Targets
```
help                    This help
update-submodules       Update all submodules using GIT
init-submodules         Initial submodule setup
force-rebuild           Forced ReBuild Docker Environment
build                   Build Docker Environment
jupyter                 Start Jupyter
connect                 Connect to CUDA Container
train-model             Train the models based on config
setup-local             Set up local environment (for development)
setup-calibrate-cameras Setup calibrate cameras application (local machine venv)
calibrate-cameras       Run the calibrate cameras application (local machine venv)

```

Just use `make` to list the targets (help)

*Make sure you update `Makefile.cfg` to point to your dataset for running the fine-tuning of the reidentification model (`make train-model`).


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

