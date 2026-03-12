import src.config as config
import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    def __init__(self, target_features):
        super(ContentLoss, self).__init__()
        self.target_layer = target_features[config.content_layer].detach()
    
    def forward(self, prediction_features):
        prediction_layer = prediction_features[config.content_layer]
        return nn.functional.mse_loss(prediction_layer, self.target_layer)
    

class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.A = {}
        for layer in config.style_layers:
            self.A[layer] = self._get_gram_matrix(target_features[layer]).detach()

    def forward(self, pred_features):
        loss = 0
        for layer in config.style_layers:
            loss += self._get_e(pred_features, layer)
        return loss / len(config.style_layers)

    def _get_e(self, features, layer):
        G = self._get_gram_matrix(features[layer])
        C, H, W = features[layer].shape
        m = H * W
        return (1 / (4 * C**2 * m**2)) * torch.sum((G - self.A[layer])**2)

    def _get_gram_matrix(self, feature_map):
        C, H, W = feature_map.shape
        F = feature_map.view(C, H * W)
        return torch.mm(F, F.t()) 