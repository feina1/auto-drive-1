"""Single-frame RGB + speed dataset and DINOv2 waypoint model."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset


TRAIN_ROOT = Path(__file__).resolve().parents[1] / "records" / "train"
DINOV2_MODEL = "facebook/dinov2-small"
MODEL_CACHE = Path(__file__).resolve().parent / ".cache" / "huggingface"


def _load_frames(session_dir):
    """Read the per-frame JSON files written by record.SessionRecorder."""
    frames = []
    for path in sorted((session_dir / "frames").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            frame = json.load(handle)
        frames.append(frame)
    return frames


def _to_ego(points, position, heading):
    """World XY -> ego XY in metres: x forward, y left; heading in radians."""
    delta = np.asarray(points, dtype=np.float64) - position
    c, s = np.cos(heading), np.sin(heading)
    return (delta @ np.array([[c, -s], [s, c]])).astype(np.float32)


class FollowPathDataset(Dataset):
    """Imitate future recorded positions within each session.

    Returns float32 tensors: front_rgb [3, H, W] in [0, 1], speed [1]
    in m/s, and waypoints [N, 2] in the current ego frame (metres).
    Only front_rgb and speed are model inputs; waypoints is the target.
    ImageNet/DINO preprocessing can be supplied as a PIL-image transform.
    Without a transform, images keep their recorded resolution.

    At the recorder's 10 Hz, defaults target 0.5, 1.0, ..., 3.0 seconds.
    Incomplete tails and windows with missing frame IDs are excluded.
    """

    def __init__(self, root=TRAIN_ROOT, num_waypoints=6, frame_stride=5, transform=None):
        if not isinstance(num_waypoints, int) or num_waypoints <= 0:
            raise ValueError("num_waypoints must be a positive integer")
        if not isinstance(frame_stride, int) or frame_stride <= 0:
            raise ValueError("frame_stride must be a positive integer")
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Recording directory not found: {root}")

        self.transform = transform
        self.samples = []
        horizon = num_waypoints * frame_stride
        offsets = np.arange(1, num_waypoints + 1) * frame_stride
        for session_dir in sorted(root.iterdir()):
            if not session_dir.is_dir():
                continue
            frames = _load_frames(session_dir)
            if len(frames) <= horizon:
                continue
            positions = np.array(
                [[f["position"]["x"], f["position"]["y"]] for f in frames],
                dtype=np.float64,
            )
            for i in range(len(frames) - horizon):
                frame = frames[i]
                if any(
                    frames[i + j]["frame_id"] != frame["frame_id"] + j
                    for j in range(1, horizon + 1)
                ):
                    continue
                image_path = session_dir / frame["image"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"Front RGB image not found: {image_path}")
                waypoints = _to_ego(
                    positions[i + offsets], positions[i], frame["heading"]
                )
                speed = float(frame["speed_kmh"]) / 3.6
                if not np.isfinite(speed) or not np.isfinite(waypoints).all():
                    raise ValueError(
                        f"Non-finite speed/waypoints: {session_dir}, frame {frame['frame_id']}"
                    )
                self.samples.append((image_path, speed, waypoints))

        if not self.samples:
            raise ValueError(
                f"No usable samples in {root}; each session needs at least "
                f"{horizon + 1} consecutive frames."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, speed, waypoints = self.samples[index]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if self.transform is not None:
                rgb = self.transform(rgb)
            else:
                rgb = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0)
                rgb = rgb.permute(2, 0, 1).contiguous()
        return {
            "front_rgb": rgb,
            "speed": torch.tensor([speed], dtype=torch.float32),
            "waypoints": torch.tensor(waypoints, dtype=torch.float32),
        }


class Model(nn.Module):
    """Frozen DINOv2 CLS feature + speed -> trainable MLP -> ego waypoints.

    Inputs: raw RGB float tensors [B, 3, H, W] in [0, 1] and speed [B, 1]
    in m/s. Use FollowPathDataset without a normalization transform: resize
    and ImageNet normalization happen here. Output: [B, N, 2] in metres.

    model_name may be a Hugging Face ID or a local save_pretrained directory.
    Public ViT-S/14 weights need no login and are cached under model/.cache.
    encoder is optional for offline tests with a DINOv2 model built from config;
    normal construction always loads pretrained weights.
    """

    def __init__(self, num_waypoints=6, model_name=DINOV2_MODEL, *, encoder=None,
                 cache_dir=MODEL_CACHE, local_files_only=False):
        super().__init__()
        if not isinstance(num_waypoints, int) or num_waypoints <= 0:
            raise ValueError("num_waypoints must be a positive integer")
        if encoder is None:
            from transformers import AutoModel

            load_options = dict(cache_dir=str(cache_dir), token=False, use_safetensors=True)
            try:
                encoder = AutoModel.from_pretrained(model_name, local_files_only=True, **load_options)
            except OSError:
                if local_files_only:
                    raise
                encoder = AutoModel.from_pretrained(model_name, **load_options)
        if encoder.config.model_type != "dinov2":
            raise ValueError("Model requires a DINOv2 ViT encoder")
        self.encoder = encoder.requires_grad_(False)
        self.encoder.eval()
        self.num_waypoints = num_waypoints
        self.head = nn.Sequential(
            nn.Linear(encoder.config.hidden_size + 1, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints * 2),
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Keep the frozen backbone deterministic during head training.
        return self

    def forward(self, front_rgb, speed):
        if front_rgb.ndim != 4 or front_rgb.shape[1] != 3:
            raise ValueError("front_rgb must have shape [B, 3, H, W]")
        if speed.shape != (front_rgb.shape[0], 1):
            raise ValueError("speed must have shape [B, 1] in m/s")
        if not front_rgb.is_floating_point():
            raise ValueError("front_rgb must be floating point RGB in [0, 1]")
        with torch.no_grad():
            pixels = F.interpolate(
                front_rgb, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
            )
            pixels = (pixels - self.image_mean) / self.image_std
            features = self.encoder(pixel_values=pixels, return_dict=True).last_hidden_state[:, 0]
        # A fixed scale keeps typical driving speeds near unit magnitude.
        fused = torch.cat((features, speed.to(dtype=features.dtype) / 30.0), dim=1)
        return self.head(fused).reshape(-1, self.num_waypoints, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--num-waypoints", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--check-model", action="store_true", help="Load pretrained DINOv2 and run one forward pass")
    parser.add_argument("--model-name", default=DINOV2_MODEL, help="Hugging Face model ID or local model directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached weights without network access")
    args = parser.parse_args()
    dataset = FollowPathDataset(args.root, args.num_waypoints, args.frame_stride)
    sample = dataset[0]
    print(f"Samples: {len(dataset)}")
    for key, value in sample.items():
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"Speed (m/s): {sample['speed'].item():.3f}")
    print(f"Ego waypoints (m):\n{sample['waypoints']}")
    if args.check_model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(args.num_waypoints, args.model_name,
                      local_files_only=args.local_files_only).to(device).eval()
        with torch.no_grad():
            prediction = model(
                sample["front_rgb"].unsqueeze(0).to(device),
                sample["speed"].unsqueeze(0).to(device),
            )
        print(f"Prediction: shape={tuple(prediction.shape)}, device={device}")
        print("Waypoint head is untrained; this checks tensor flow only.")
