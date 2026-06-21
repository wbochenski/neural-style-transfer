import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import FeatureExtractor
from losses import TotalLoss
import torch
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
from log import get_logger

logger = get_logger()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from torchvision.transforms import v2
from PIL import Image
import torch


def load_image(image: Image, max_size: int = None) -> torch.Tensor:
    """Load an image from the given path and convert it to a normalized tensor."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Resize keeping the aspect ratio the same
    if max_size is not None:
        image.thumbnail([max_size, max_size])

    # Convert to tensor and normalize using ImageNet mean and std
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    return transforms(image)


def decode_image(tensor: torch.Tensor) -> Image.Image:
    """Reverse the normalization and convert tensor to PIL image."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Reverse the normalization using ImageNet mean and std
    tensor = tensor.detach() * std.view(3, 1, 1) + mean.view(3, 1, 1)
    tensor = tensor.clamp(0, 1)
    image = v2.ToPILImage()(tensor.cpu())
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")

    return base64_str


@app.get("/health")
def health_check():
    return {"status": "ok"}


class ImagePayload(BaseModel):
    content_img: str  # base64 string
    style_img: str  # base64 string


@app.post("/style-transfer")
def style_transfer(payload: ImagePayload):
    size = 512
    steps = 300
    content_layer = "conv4_2"
    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    alpha_beta_ratio = 10e-3

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    extractor = FeatureExtractor(device=device)

    content_img = Image.open(BytesIO(base64.b64decode(payload.content_img))).convert(
        "RGB"
    )
    content_img.thumbnail([size, size])
    style_img = Image.open(BytesIO(base64.b64decode(payload.style_img))).convert("RGB")
    style_img.thumbnail([size, size])

    content_tensor = (
        load_image(content_img).to(device).contiguous().requires_grad_(False)
    )
    style_tensor = load_image(style_img).to(device).contiguous().requires_grad_(False)
    pred_tensor = content_tensor.clone().to(device).contiguous().requires_grad_(True)

    content_features = extractor(content_tensor)
    style_features = extractor(style_tensor)

    total_loss = TotalLoss(
        content_features,
        style_features,
        alpha_beta_ratio=alpha_beta_ratio,
        content_layer=content_layer,
        style_layers=style_layers,
    )
    optimizer = torch.optim.LBFGS([pred_tensor])
    lo = (0 - mean.view(3, 1, 1)) / std.view(3, 1, 1)
    hi = (1 - mean.view(3, 1, 1)) / std.view(3, 1, 1)

    def closure():
        optimizer.zero_grad()
        pred_features = extractor(pred_tensor)
        loss = total_loss(pred_features)
        loss.backward()
        return loss

    for _ in range(steps):
        optimizer.step(closure)
        with torch.no_grad():
            pred_tensor.clamp_(lo, hi)

    return {"image": decode_image(pred_tensor)}


if __name__ == "__main__":
    logger.info("API is up!")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
