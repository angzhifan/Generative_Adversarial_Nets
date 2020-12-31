"""
MLP GAN Network Architecture
"""
from abc import ABC

import torch.nn as nn


# Generator
class Generator_1(nn.Module):

    def __init__(self, t, d):
        super(Generator_1, self).__init__()
        self.Nz = t
        self.dim = d
        self.main = nn.Sequential(
            nn.Linear(t, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, d),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.main(z)
        return x


# Discriminator
class Discriminator_1(nn.Module):

    def __init__(self, d):
        super(Discriminator_1, self).__init__()
        self.dim = d
        self.main = nn.Sequential(
            nn.Linear(d, 1024),
            nn.Tanh(),  # LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.Tanh(),  # LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Tanh(),  # LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Tanh(),  # LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x