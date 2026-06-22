import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import FeatureExtractor
from losses import TotalLoss
import torch
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
from log import get_logger
import uuid
import threading
from enum import Enum

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


def load_image(image: Image, max_size: int = None) -> torch.Tensor:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    if max_size is not None:
        image.thumbnail([max_size, max_size])

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    return transforms(image)


def decode_image(tensor: torch.Tensor) -> str:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    tensor = tensor.detach() * std.view(3, 1, 1) + mean.view(3, 1, 1)
    tensor = tensor.clamp(0, 1)
    image = v2.ToPILImage()(tensor.cpu())
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")

    return base64_str


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


jobs: dict[str, dict] = {}
lock = threading.Lock()


class ImagePayload(BaseModel):
    content_img: str
    style_img: str


def run_style_transfer_job(job_id: str, payload: ImagePayload):
    size = 512
    steps = 4
    content_layer = "conv4_2"
    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    alpha_beta_ratio = 10e-3

    content_size = len(payload.content_img)
    style_size = len(payload.style_img)
    logger.info(
        f"Job {job_id}: style transfer started — content: {content_size / 1024:.1f}KB, "
        f"style: {style_size / 1024:.1f}KB, "
        f"steps: {steps}, size: {size}"
    )

    try:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        logger.info(f"Job {job_id}: using device: {device}")

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        extractor = FeatureExtractor(device=device)

        content_img = Image.open(BytesIO(base64.b64decode(payload.content_img))).convert(
            "RGB"
        )
        content_img.thumbnail([size, size])
        style_img = Image.open(BytesIO(base64.b64decode(payload.style_img))).convert("RGB")
        style_img.thumbnail([size, size])
        logger.debug(f"Job {job_id}: decoded images — content: {content_img.size}, style: {style_img.size}")

        content_tensor = (
            load_image(content_img).to(device).contiguous().requires_grad_(False)
        )
        style_tensor = load_image(style_img).to(device).contiguous().requires_grad_(False)
        pred_tensor = content_tensor.clone().to(device).contiguous().requires_grad_(True)

        content_features = extractor(content_tensor)
        style_features = extractor(style_tensor)
        logger.debug(f"Job {job_id}: features extracted")

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

        with lock:
            jobs[job_id]["status"] = JobStatus.RUNNING

        def closure():
            optimizer.zero_grad()
            pred_features = extractor(pred_tensor)
            loss = total_loss(pred_features)
            loss.backward()
            return loss

        for i in range(steps):
            optimizer.step(closure)
            with torch.no_grad():
                pred_tensor.clamp_(lo, hi)

            progress = round((i + 1) / steps * 100)
            with lock:
                jobs[job_id]["progress"] = progress
            logger.info(f"Job {job_id}: step {i + 1}/{steps} ({progress}%)")

        result = decode_image(pred_tensor)
        logger.info(f"Job {job_id}: transfer complete — output: {len(result) / 1024:.1f}KB")

        with lock:
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["result"] = result

    except Exception as e:
        logger.error(f"Job {job_id}: failed — {e}")
        with lock:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = str(e)


@app.get("/health")
def health_check():
    logger.debug("Health check called")
    return {"status": "ok"}


@app.post("/style-transfer")
def start_style_transfer(payload: ImagePayload):
    job_id = str(uuid.uuid4())
    with lock:
        jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0,
            "result": None,
            "error": None,
        }

    logger.info(f"Job {job_id}: created")
    thread = threading.Thread(target=run_style_transfer_job, args=(job_id, payload))
    thread.start()

    return {"job_id": job_id}


@app.get("/style-transfer/{job_id}")
def get_job_status(job_id: str):
    with lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == JobStatus.COMPLETED:
        return {
            "status": JobStatus.COMPLETED,
            "image": job["result"],
        }
    elif job["status"] == JobStatus.FAILED:
        return {
            "status": JobStatus.FAILED,
            "error": job["error"],
        }
    elif job["status"] == JobStatus.RUNNING:
        return {
            "status": JobStatus.RUNNING,
            "progress": job["progress"],
        }
    else:
        return {
            "status": JobStatus.PENDING,
            "progress": 0,
        }


if __name__ == "__main__":
    logger.info("API is up!")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
