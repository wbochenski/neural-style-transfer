import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    def __init__(self, target_features, content_layer="conv4_2"):
        super(ContentLoss, self).__init__()
        self.content_layer = content_layer
        self.target_layer = target_features[self.content_layer].detach()
    
    def forward(self, prediction_features):
        prediction_layer = prediction_features[self.content_layer]
        return nn.functional.mse_loss(prediction_layer, self.target_layer)
    

class StyleLoss(nn.Module):
    def __init__(self, target_features, style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]):
        super(StyleLoss, self).__init__()
        self.style_layers = style_layers
        self.A = {}
        for layer in self.style_layers:
            self.A[layer] = self._get_gram_matrix(target_features[layer]).detach()

    def forward(self, pred_features):
        loss = 0
        for layer in self.style_layers:
            loss += self._get_e(pred_features, layer)
        return loss / len(self.style_layers)

    def _get_e(self, features, layer):
        G = self._get_gram_matrix(features[layer])
        C, H, W = features[layer].shape
        m = H * W
        return (1 / (4 * C**2 * m**2)) * torch.sum((G - self.A[layer])**2)

    def _get_gram_matrix(self, feature_map):
        C, H, W = feature_map.shape
        F = feature_map.view(C, H * W)
        return torch.mm(F, F.t()) 
    

class TotalLoss(nn.Module):
    def __init__(self, content_features, style_features, alpha_beta_ratio: float = 10e-3, content_layer="conv4_2", style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]):
        super(TotalLoss, self).__init__()
        self.content_loss = ContentLoss(content_features, content_layer)
        self.style_loss = StyleLoss(style_features, style_layers)
        self.alpha_beta_ratio = alpha_beta_ratio

    def forward(self, pred_features):
        return self.alpha_beta_ratio * self.content_loss(pred_features) + (1 - self.alpha_beta_ratio) * self.style_loss(pred_features)