# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F


# class LinearModel(torch.nn.Module):
#     def __init__(self, hyperparameters: dict):
#         super(LinearModel, self).__init__()

#         # Get model config
#         self.input_dim = hyperparameters['input_dim']
#         self.output_dim = hyperparameters['output_dim']
#         self.hidden_dims = hyperparameters['hidden_dims']
#         self.negative_slope = hyperparameters.get("negative_slope", .2)

#         # Create layer list
#         self.layers = torch.nn.ModuleList([])
#         all_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
#         for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
#             self.layers.append(torch.nn.Linear(in_dim, out_dim))

#         self.num_layers = len(self.layers)

#     def forward(self, x):
#         for i in range(self.num_layers - 1):
#             x = self.layers[i](x)
#             x = torch.nn.functional.leaky_relu(
#                 x, negative_slope=self.negative_slope)
#         x = self.layers[-1](x)
#         return torch.nn.functional.softmax(x, dim=-1)

# # FASHIONMNIST


# class Autoencoder(nn.Module):
#     def __init__(self, hyperparameters: dict):
#         super(Autoencoder, self).__init__()

#         # TODO: Add Get model config
#         # encoder layers
#         self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
#         self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
#         self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)

#         # decoder layers
#         self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
#         self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
#         self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
#         self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
#         self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

#     def forward(self, x):
#         # encode
#         x = F.relu(self.enc1(x))
#         x = self.pool(x)
#         x = F.relu(self.enc2(x))
#         x = self.pool(x)
#         x = F.relu(self.enc3(x))
#         x = self.pool(x)
#         x = F.relu(self.enc4(x))
#         x = self.pool(x)  # the latent space representation

#         # decode
#         x = F.relu(self.dec1(x))
#         x = F.relu(self.dec2(x))
#         x = F.relu(self.dec3(x))
#         x = F.relu(self.dec4(x))
#         x = F.sigmoid(self.out(x))
#         return x
