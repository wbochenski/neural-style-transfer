from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
import torch


def load_image(path: Path, max_size: int = None) -> torch.Tensor:
    """Load an image from the given path and convert it to a normalized tensor."""
    image = Image.open(path).convert('RGB')

    # Resize keeping the aspect ratio the same
    if max_size is not None:
        image.thumbnail([max_size, max_size])

    # Convert to tensor and normalize using ImageNet mean and std
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms(image)


def decode_image(tensor: torch.Tensor) -> Image.Image:
    """Reverse the normalization and convert tensor to PIL image."""

    # Reverse the normalization using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean

    return v2.ToPILImage()(tensor)