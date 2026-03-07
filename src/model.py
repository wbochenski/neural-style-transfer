import torchvision
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.vgg19(weights='IMAGENET1K_V1').features
model.requires_grad_ = False
for i in range(len(model)):
    # print(model[i])
    if type(model[i]) == torch.nn.modules.pooling.MaxPool2d:
        model[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
model.to(device)