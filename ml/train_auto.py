import torch
from tqdm import tqdm

"""
Contains PyTorch model code to instantiate a Auto Encoder model.
"""
import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# constants
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NOISE_FACTOR = 0.5


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def make_dir():
    image_dir = 'Saved_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def save_decoded_image(img, name):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, name)


class Trainer:
    def __init__(self, model, optimizer=None, criterion=None, device=None):
        """Initialize the trainer"""
        self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=.001)

        self.criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.model = self.model.to(device)

    def get_model(self):
        return self.model

    # TODO: Add function to divide train to tain and val
    def train(self, num_epochs, train_dataloader, val_dataloader=None):
        """Trains the model and logs the results"""
        # Set result dict
        results = {"train_loss": [], "train_acc": []}
        if val_dataloader is not None:
            results["val_loss"] = []
            results["val_acc"] = []

        print("Starting training")
        # Start training
        for epoch in tqdm(range(num_epochs)):
            train_loss = self.train_epoch(
                dataloader=train_dataloader)
            results["train_loss"].append(train_loss)

            # Validate only if we have a val dataloader
            if val_dataloader is not None:
                val_loss = self.eval_epoch(dataloader=val_dataloader)
                results["val_loss"].append(val_loss)

        return results

    def train_epoch(self, dataloader):
        """Trains one epoch"""
        self.model.train()
        total_loss = 0.
        total_correct = 0.
        for i, batch in enumerate(dataloader):
            # Send to device
            img, _ = batch  # we do not need the image labels
            # TODO: Make this into a sep fun
            # add noise to the image data
            X = img + NOISE_FACTOR * torch.randn(img.shape)
            # clip to make the values fall between 0 and 1
            X = np.clip(X, 0., 1.)
            X = X.to(self.device)

            # Train step
            self.optimizer.zero_grad()  # Clear gradients.
            outs = self.model(X)  # Perform a single forward pass.
            loss = self.criterion(outs, X)

            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.

            # Compute metrics
            total_loss += loss.detach().item()
        #     total_correct += torch.sum(torch.argmax(outs,
        #                                dim=-1) == y).detach().item()
        # total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        # return total_loss, total_acc
        return total_loss

    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.
        total_correct = 0.
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # Eval
            outs = self.model(X)
            loss = self.criterion(outs, y)

            # Compute metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs,
                                       dim=-1) == y).detach().item()
        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc

    def test_image_reconstruction(self, testloader):
        for batch in testloader:
            img, _ = batch
            img_noisy = img + NOISE_FACTOR * torch.randn(img.shape)
            img_noisy = np.clip(img_noisy, 0., 1.)
            img_noisy = img_noisy.to(self.device)
            outputs = self.model(img_noisy)
            outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
            # save_image(img_noisy, 'noisy_test_input.png')
            # save_image(outputs, 'denoised_test_reconstruction.png')
            return img_noisy, outputs
