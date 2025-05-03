from itertools import chain
import torch
import timm
import torchvision.transforms as T
from torch.optim import SGD, Adam, SparseAdam, AdamW, ASGD, LBFGS
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.train import ArcFaceLoss, BasicTrainer
from welfareobs.utils.config import Config
from welfareobs.detectron.welfareobs_dataset import WelfareObsDataset
import os
import pandas as pd


"""
Example:

{
    "configs": "wod-1-md",
    "wod-1-md": {
        "comment": "Fine-tuning Mega Descriptor",
        "name": "wod-1-md",
        "root": "/project/data/1028749_First Group",
        "backbone": "hf-hub:BVRA/wildlife-mega-L-384",
        "trainer-batch-size": "32",
        "trainer-workers": "2",
        "trainer-epochs": "1",
        "features-batch-size": "64",
        "features-workers": "2",
        "dimensions": "384",
        "optimizer": "SGD",
        "learning-rate": "0.001",
        "learning-rate-momentum": "0.9"
    },
}

Variations on the Optimizers:
        "optimizer": "SGD",
        "learning-rate": "0.001",
        "learning-rate-momentum": "0.9"
        
        "optimizer": "ASGD",
        "learning-rate": "0.001",
        "learning-rate-lambd": "0.0001"

        "optimizer": "Adam",
        "learning-rate": "0.001",

        "optimizer": "AdamW",
        "learning-rate": "0.001",
        "learning-rate-weight-decay": "0.01"

        "optimizer": "SparseAdam",
        "learning-rate": "0.001",

        "optimizer": "LBFGS",
        "learning-rate": "0.001",
        "learning-rate-max-iterations": "20"

Alternative backbones:
        "swin_large_patch4_window12_384"


"""

# Set data for similarity-aware and random splits
config: Config = Config("/project/config.json")
sets = [o.strip() for o in config["configs"].split(",")]
for ptr in sets:
    dimensions = config.as_int(f"{ptr}.dimensions")
    learning_rate = config.as_float(f"{ptr}.learning-rate")
    use_opt = config[f"{ptr}.optimizer"]
    name = config[f"{ptr}.name"]
    outpath = f"/project/data/results/{name}"
    print(f"Processing {ptr} to {outpath}.")
    os.makedirs(outpath, exist_ok=True)
    dataset=WelfareObsDataset(
        root=config[f"{ptr}.root"],
        annotations_file=config[f"{ptr}.annotations-filename"],
        transform = T.Compose([
            T.Resize(
                size=(size,size),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                max_size=None,
                antialias=True
            )            
            # T.Resize(size=dimensions),
            # T.CenterCrop(size=[dimensions, dimensions]),
            T.ToTensor(),
            # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        img_load="full", # "bbox_mask",
        col_path="path",
        col_label="identity",
        load_label=True
    )
    backbone = timm.create_model(config[f"{ptr}.backbone"], num_classes=0, pretrained=True)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, dimensions, dimensions)
        embedding_size = backbone(dummy_input).shape[1]
    print(f"Objective loss function is configured for {dataset.num_classes} features and {embedding_size} embedding.")
    objective = ArcFaceLoss(
        num_classes=dataset.num_classes,
        embedding_size=embedding_size,
        margin=0.5,
        scale=64
    )
    params = chain(backbone.parameters(), objective.parameters())
    optimizer = None
    if use_opt == "SGD":
        optimizer = SGD(params=params, lr=learning_rate, momentum=config.as_float(f"{ptr}.learning-rate-momentum"))
    elif use_opt == "ASGD":
        optimizer = ASGD(params=params, lr=learning_rate, lambd=config.as_float(f"{ptr}.learning-rate-lambd"))
    elif use_opt == "Adam":
        optimizer = Adam(params=params, lr=learning_rate)
    elif use_opt == "AdamW":
        optimizer = AdamW(params=params, lr=learning_rate, weight_decay=config.as_float(f"{ptr}.learning-rate-weight-decay"))
    elif use_opt == "SparseAdam":
        optimizer = SparseAdam(params=params, lr=learning_rate)
    elif use_opt == "LBFGS":
        optimizer = LBFGS(params=params, lr=learning_rate, max_iter=config.as_int(f"{ptr}.learning-rate-max-iterations"))
    if optimizer is None:
        raise SyntaxError("Unsupported optimizer")
    min_lr = optimizer.defaults.get("lr") * 1e-3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=min_lr)
    trainer = BasicTrainer(
        dataset=dataset,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=config.as_int(f"{ptr}.trainer-batch-size"),
        accumulation_steps=8,
        num_workers=config.as_int(f"{ptr}.trainer-workers"),
        epochs=config.as_int(f"{ptr}.trainer-epochs"),
        device='cuda',
    )
    print("Training...")
    trainer.train()
    trainer.save(outpath)
    extractor = DeepFeatures(backbone,
                             device='cuda',
                             batch_size=config.as_int(f"{ptr}.features-batch-size"),
                             num_workers=config.as_int(f"{ptr}.features-workers")
                             )
    print("Extracting features...")
    features = extractor(dataset)
    features.save(os.path.join(outpath, f"similarity.pkl"))
    ####
    # TODO: add validation pass

    
