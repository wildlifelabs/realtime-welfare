import torch
import torch.nn as nn
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.data import WildlifeDataset, FeatureDataset
import timm


class ReIdHead(nn.Module):
    def __init__(self,
                 input_dim: int,
                 model_name: str,
                 checkpoint_filename: str|None = None,
                 batch_size: int = 128,
                 num_workers: int = 1,
                 device: str = "cpu",
                 features_database: FeatureDataset|None = None
                 ):
        super().__init__()
        # we expose this for pre-run validation only
        self.input_dim = input_dim
        intermediate_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        if checkpoint_filename is not None:
            intermediate_model.load_state_dict(torch.load(checkpoint_filename, weights_only=False)['model'])
        self.extractor = DeepFeatures(
            intermediate_model,
            batch_size=batch_size,
            num_workers=0, # this fixes a reinit bug
            device=device
        )
        self.features_database = features_database

    def forward(self, x, labels=None):
        matcher = CosineSimilarity()
        result = KnnClassifier(
            k=1,
            database_labels=self.features_database.labels_string
        )(
            matcher(query=self.extractor([x]), database=self.features_database)
        )
        return result
        # print(result)

