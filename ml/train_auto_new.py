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
    def __init__(self, encoder, decoder, params_to_optimize, optimizer=None, loss_fn=None, device=None):
        """Initialize the trainer"""
        self.encoder = encoder
        self.decoder = decoder

        # HYPER-PARAMETERS
        self.lr = 0.001
        self.noise_factor = 0.3

        # Define the loss function
        self.loss_fn = torch.nn.MSELoss() if loss_fn is None else loss_fn

        # Set the random seed for reproducible results
        torch.manual_seed(42)

        # Making device agnostic
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        # # Initialize the two networks
        # d = 4

        # #model = Autoencoder(encoded_space_dim=encoded_space_dim)
        # self.encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
        # self.decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)
        # params_to_optimize = [
        #     {'params': encoder.parameters()},
        #     {'params': decoder.parameters()}
        # ]

        # Define the optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(
                params_to_optimize, lr=self.lr, weight_decay=1e-05)

        # Move both the encoder and the decoder to the selected device
        self.encoder.to(device)
        self.decoder.to(device)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def add_noise(inputs, noise_factor=0.3):
        noisy = inputs+torch.randn_like(inputs) * noise_factor
        noisy = torch.clip(noisy, 0., 1.)
        return noisy

    # TODO: Add function to divide train to tain and val
    def train(self, num_epochs, train_dataloader, val_dataloader=None):
        """Trains the model and logs the results"""
        # Training cycle
        noise_factor = 0.3
        results = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs):
            print("Starting training")
            print('EPOCH %d/%d' % (epoch + 1, num_epochs))
            # Training (use the training function)
            train_loss = self.train_epoch(
                encoder=self.encoder,
                decoder=self.decoder,
                device=self.device,
                dataloader=train_dataloader,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                noise_factor=noise_factor)
            # Validation  (use the testing function)
            val_loss = self.test_epoch(
                encoder=self.encoder,
                decoder=self.decoder,
                device=self.device,
                dataloader=val_dataloader,
                loss_fn=self.loss_fn,
                noise_factor=noise_factor)

            # Print Validationloss
            results['train_loss'].append(train_loss)
            results['val_loss'].append(val_loss)
            print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(
                epoch + 1, num_epochs, train_loss, val_loss))
            # plot_ae_outputs_den(encoder, decoder, noise_factor=noise_factor)

        return results

    def train_epoch(self, dataloader):
        """Trains one epoch"""
        # Set train mode for both the encoder and the decoder
        self.encoder.train()
        self.decoder.train()
        train_loss = []

        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        # with "_" we just ignore the labels (the second element of the dataloader tuple)
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_noisy = self.add_noise(image_batch, self.noise_factor)
            image_batch = image_batch.to(self.device)
            image_noisy = image_noisy.to(self.device)
            # Encode data
            encoded_data = self.encoder(image_noisy)
            # Decode data
            decoded_data = self.decoder(encoded_data)
            # Evaluate loss
            loss = self.loss_fn(decoded_data, image_batch)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Print batch loss
            print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

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

    # Testing function

    def test_epoch(self, dataloader):
        # Set evaluation mode for encoder and decoder
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for image_batch, _ in dataloader:
                # Move tensor to the proper device
                image_noisy = self.add_noise(image_batch, self.noise_factor)
                image_noisy = image_noisy.to(self.device)
                # Encode data
                encoded_data = self.encoder(image_noisy)
                # Decode data
                decoded_data = self.decoder(encoded_data)
                # Append the network output and the original image to the lists
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            # Evaluate global loss
            val_loss = self.loss_fn(conc_out, conc_label)
        return val_loss.data

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
