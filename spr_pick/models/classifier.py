from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, features):
        super(LinearClassifier, self).__init__()
        self.features = features
        self.classifier = nn.Conv2d(features.latent_dim, 1, 1)

    @property
    def width(self):
        return self.features.width

    @property
    def latent_dim(self):
        return self.features.latent_dim

    def fill(self, stride=1):
        return self.features.fill(stride=stride)

    def unfill(self):
        self.features.unfill()

    def forward(self, x):
        # print('input')
        # print(x.shape)
        # print('self.features')
        # print(self.features)
        z = self.features(x)
        # print('feature size')
        # print(z.shape)
        y = self.classifier(z)
        # print('output shape')
        # print(y.shape)
        return y

