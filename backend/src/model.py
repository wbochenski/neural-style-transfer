from torch import Tensor, nn
import torch
from torchvision.models import VGG19_Weights, vgg19

class FeatureExtractor(nn.Module):
    """Extract intermediate VGG19 feature maps"""
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.layer_map = {
            1: "conv1_1",
            3: "conv1_2",
            6: "conv2_1",
            8: "conv2_2",
            11: "conv3_1",
            13: "conv3_2",
            15: "conv3_3",
            17: "conv3_4",
            20: "conv4_1",
            22: "conv4_2",
            24: "conv4_3",
            26: "conv4_4",
            29: "conv5_1",
            31: "conv5_2",
            33: "conv5_3",
            35: "conv5_4",
        }
        self.features = {}
        self.model = self._build_model()

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        self.features.clear()
        _ = self.model(x)
        return self.features.copy()

    def _get_hook(self, name: str):
        def hook(_module, _input, output):
            self.features[name] = output
        return hook

    def _build_model(self) -> nn.Sequential:
        """Load and setup the pretrained VGG19 model"""
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        # Replace max pooling layers with average pooling layers
        for idx, layer in enumerate(model):
            if isinstance(layer, nn.MaxPool2d):
                model[idx] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Register hooks to collect intermediate features
        for layer_idx, layer_name in self.layer_map.items():
            model[layer_idx].register_forward_hook(self._get_hook(layer_name))

        # Freeze the model parameters and set to evaluation mode
        model.requires_grad_(False)
        model.to(self.device)
        model.eval()

        return model