# WelfareObs Container

Docker framework for evaluating the WelfareObs package

## Makefile Targets
```
help                This help
update-submodules   Update all submodules using GIT
init-submodules     Initial submodule setup
force-rebuild       Forced ReBuild Docker Environment
build               Build Docker Environment
jupyter             Start Jupyter
connect             Connect to CUDA Container
train-model         Train the models based on config
eval-model          Evaluate the model based on config
```

Just use `make` to list the targets (help)

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

