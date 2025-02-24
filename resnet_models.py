'''
@author: James V. Talwar
Created on 2/24/2025

About: Resnets for testing with SupCon loss. Supported resnets include resnet-18, 50, and 101. Default: resnet-18.
'''

from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ResNet(nn.Module):
    def __init__(self, which = 18, projection_dim = 128):
        super(ResNet, self).__init__()

        assert int(which) in {18, 50, 101}, f"Provided resnet-{which} unsupported. Valid selections include: 18, 50, and 101. Exiting..."

        model_map = {18: models.resnet18(pretrained = False), 
                     50: models.resnet50(pretrained = False),
                     101: models.resnet101(pretrained = False)}
        base_model =  model_map.get(which)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last fully connected layer
        self.projection = nn.Linear(base_model.fc.in_features, projection_dim)  # New projection layer

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.normalize(x, dim = -1, p = 2) # Normalize output of encoder to unit hypersphere (as specified by paper)
        x = self.projection(x)
        
        return x