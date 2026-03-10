import src.config as config
from torch import Tensor, nn
from torchvision.models import VGG19_Weights, vgg19

class FeatureExtractor(nn.Module):
    """Extract intermediate VGG19 feature maps"""
    def __init__(self) -> None:
        super().__init__()
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
        for layer_idx, layer_name in config.layer_map.items():
            model[layer_idx].register_forward_hook(self._get_hook(layer_name))

        # Freeze the model parameters and set to evaluation mode
        model.requires_grad_(False)
        model.to(config.device)
        model.eval()

        return model