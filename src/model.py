import src.config as config
import torchvision
import torch

# Load the pre-trained VGG19 model and take only the CNN layers (features)
model = torchvision.models.vgg19(weights='IMAGENET1K_V1').features

# Replace all MaxPool2d layers with AvgPool2d layers
for i in range(len(model)):
    if isinstance(model[i], torch.nn.MaxPool2d):
        model[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

# Freeze the model parameters and move the model to the appropriate device
model.requires_grad_(False)
model.to(config.device)
model.eval()