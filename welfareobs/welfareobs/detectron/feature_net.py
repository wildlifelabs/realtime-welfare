import torch
import torch.nn as nn
from tqdm import tqdm


class FeatureNet(nn.Module):
    def __init__(self, 
                 model_name, 
                 num_classes, 
                 image_dimensions=384, 
                 hidden_dim=256, 
                 depth=2, 
                 dropout=0.3, 
                 device="cuda"
                ):
        super().__init__()
        self.extractor = torch.hub.load("facebookresearch/dinov2", model_name)
        self.extractor.to(device)
        self.embedding_size = None
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_dimensions, image_dimensions)
            self.embedding_size = self.extractor(dummy_input.to(device)).shape[1]
        layers = []
        layers.append(nn.Linear(self.embedding_size, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.model = nn.Sequential(*layers)
        self.device = device

    def forward(self, x):
        return self.model(x)

    def load(self, file_name="checkpoint.pth"):
        self.model.load_state_dict(torch.load(file_name, weights_only=False, map_location=torch.device(self.device))['model'])

    def save(self, file_name="checkpoint.pth"):
        checkpoint = {}
        checkpoint["model"] = self.model.state_dict()
        torch.save(checkpoint, file_name)

    def extract(self, dataset):
        x = []
        y = [] 
        for item in tqdm(dataset, desc=f"Embeddings: ", mininterval=1, ncols=100):
            tmp_x, tmp_y = item
            x.append(self.extractor(tmp_x.unsqueeze(0).to(self.device)))
            y.append(tmp_y)
        return torch.tensor(x), torch.tensor(y)
        
