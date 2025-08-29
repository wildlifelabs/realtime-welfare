# -*- coding: utf-8 -*-
"""
Module Name: welfareobs_trainer.py
Description: this code comes from wildlife-tools/wildlife_tools/train/trainer.py 
however it has been adapted to incorporate support for checkpointing and tensorboard
metrics.

It's also worth noting that the original code was released under the MIT license.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
import os
import random
import numpy as np
import torch
import torch.backends.cudnn
from tqdm import tqdm
from wildlife_tools.train import ArcFaceLoss, BasicTrainer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings


class WelfareObsTrainer(BasicTrainer):
    """
    Implements basic training loop for Pytorch models.
    Checkpoints includes random states - any restarts from checkpoint preservers reproducibility.

    Args:
        dataset ():
            Training dataset that gives (x, y) tensor pairs.
        model (dict):
            Pytorch nn.Module for model / backbone.
        objective (dict):
            Pytorch nn.Module for objective / loss function.
        optimizer:
            Pytorch optimizer.
        scheduler (optional):
            Pytorch scheduler.
        epochs (int):
            Number of training epochs.
        device (str, default: 'cuda'):
            Device to be used for training.
        batch_size (int, default: 128):
            Training batch size.
        num_workers (int, default: 1):
            Number of data loading workers in torch DataLoader.
        accumulation_steps (int, default: 1):
            Number of gradient accumulation steps.
        epoch_callback:
            Callback function to be called after each epoch.

    """

    def __init__(
        self,
        working_directory,
        dataset,
        model,
        objective,
        optimizer,
        epochs,
        scheduler=None,
        device="cuda",
        batch_size=128,
        num_workers=1,
        accumulation_steps=1,
        checkpoint_epochs=False
    ):
        self.working_directory = working_directory
        self.dataset = dataset
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.epoch = 0
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.accumulation_steps = accumulation_steps
        self.checkpoint_epochs = checkpoint_epochs
        self.writer = SummaryWriter(log_dir=working_directory)

    def train(self):
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        for e in range(self.epochs):
            epoch_data = self.train_epoch(loader)
            if self.checkpoint_epochs:
                self.save(file_name=f"checkpoint_{self.epoch}.pth")
            self.epoch += 1
            
    def train_epoch(self, loader):
        model = self.model.train()
        losses = []
        accuracies = []
        precisions = []
        total = 0
        correct = 0
        count = len(loader)
        predictions = []
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {self.epoch}: ", mininterval=1, ncols=100)):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            out = model(x)
            loss = self.objective(out, y)
            logits = self.objective.loss.get_logits(out)
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
            loss.backward()
            if (i - 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tmp_loss = loss.detach().cpu()
            tmp_accuracy = accuracy_score(y.cpu(), predicted.cpu())
            tmp_precision = precision_score(y.cpu(), predicted.cpu(), average=None)
            losses.append(tmp_loss)
            accuracies.append(tmp_accuracy)
            precisions.append(tmp_precision[0])
            
            self.writer.add_scalar('Training/Loss', tmp_loss, self.epoch * count + i)
            self.writer.add_scalar('Training/Accuracy', tmp_accuracy, self.epoch * count + i)
            self.writer.add_scalar('Training/Precision', tmp_precision[0], self.epoch * count + i)
        if self.scheduler:
            self.scheduler.step()
        self.writer.add_scalar('Epoch/Loss', np.mean(losses), self.epoch * count + count)
        self.writer.add_scalar('Epoch/Accuracy', np.mean(accuracies), self.epoch * count + count)
        self.writer.add_scalar('Epoch/Precision', np.mean(precisions), self.epoch * count + count)
        return {"train_loss_epoch_avg": np.mean(losses)}

    def save(self, file_name="checkpoint.pth", save_rng=True, **kwargs):
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        checkpoint = {}
        checkpoint["model"] = self.model.state_dict()
        checkpoint["objective"] = self.objective.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        checkpoint["epoch"] = self.epoch
        if save_rng:
            checkpoint["rng_states"] = self.get_random_states()
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        torch.save(checkpoint, os.path.join(self.working_directory, file_name))

    def load(self, file_name="checkpoint.pth", load_rng=True):
        checkpoint = torch.load(os.path.join(self.working_directory, file_name), map_location=torch.device(self.device), weights_only=False)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        if "objective" in checkpoint:
            self.objective.load_state_dict(checkpoint["objective"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        if "rng_states" in checkpoint and load_rng:
            self.set_random_states(checkpoint["rng_states"])
        if "scheduler" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
    def set_seed(self, seed=0):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def set_random_states(self, states):
        if "os_rng_state" in states and states["os_rng_state"]:
            os.environ["PYTHONHASHSEED"] = states["os_rng_state"]
        if "random_rng_state" in states:
            random.setstate(states["random_rng_state"])
        if "numpy_rng_state" in states:
            np.random.set_state(states["numpy_rng_state"])
        if "torch_rng_state" in states:
            torch.set_rng_state(states["torch_rng_state"])
        if "torch_cuda_rng_state" in states:
            torch.cuda.set_rng_state(states["torch_cuda_rng_state"])
        if "torch_cuda_rng_state_all" in states:
            torch.cuda.set_rng_state_all(states["torch_cuda_rng_state_all"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def get_random_states(self):
        """Gives dictionary of random states for reproducibility."""
        states = {}
        states["os_rng_state"] = os.environ.get("PYTHONHASHSEED")
        states["random_rng_state"] = random.getstate()
        states["numpy_rng_state"] = np.random.get_state()
        states["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            states["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
            states["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return states        
    

    
