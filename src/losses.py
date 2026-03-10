import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
    
    def forward(self, target_features, prediction_features, layer):
        target_layer = target_features[layer].detach()
        prediction_layer = prediction_features[layer]
        return 0.5 * torch.sum((prediction_layer - target_layer) ** 2)