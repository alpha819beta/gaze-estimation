"""Shared gaze-estimation runtime: model load, preprocessing, decoding, face detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 448


def resolve_dataset_config(dataset: str) -> Dict[str, int]:
    if dataset not in data_config:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {list(data_config.keys())}"
        )
    return dict(data_config[dataset])


def default_weight_path(model: str, weights_dir: Union[str, Path] = "weights") -> Path:
    return Path(weights_dir) / f"{model}.pt"


def load_gaze_model(
    arch: str,
    weight_path: Union[str, Path],
    bins: int,
    device: torch.device,
) -> torch.nn.Module:
    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weight_path}. "
            f"Download with: sh download.sh {arch}  (or place file under weights/)"
        )

    model = get_model(arch, bins, inference_mode=True)
    try:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded gaze model %s from %s", arch, weight_path)
    return model


def clamp_bbox(
    bbox: List[float], frame_width: int, frame_height: int
) -> Optional[Tuple[int, int, int, int]]:
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, y_min, x_max, y_max


def crop_face(frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    clamped = clamp_bbox(bbox, w, h)
    if clamped is None:
        return None
    x_min, y_min, x_max, y_max = clamped
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return crop


class FacePreprocessor:
    def __init__(self) -> None:
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, bgr_face: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        return self._transform(rgb).unsqueeze(0)


def decode_gaze(
    pitch_logits: torch.Tensor,
    yaw_logits: torch.Tensor,
    idx_tensor: torch.Tensor,
    binwidth: int,
    angle: int,
) -> Tuple[float, float]:
    """Return pitch and yaw in radians."""
    pitch_prob = F.softmax(pitch_logits, dim=1)
    yaw_prob = F.softmax(yaw_logits, dim=1)
    pitch_deg = torch.sum(pitch_prob * idx_tensor, dim=1) * binwidth - angle
    yaw_deg = torch.sum(yaw_prob * idx_tensor, dim=1) * binwidth - angle
    return float(np.radians(pitch_deg.cpu().item())), float(
        np.radians(yaw_deg.cpu().item())
    )


class GazeSmoother:
    """Per-face EMA smoothing keyed by bbox center."""

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("smooth alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Dict[Tuple[int, int], Tuple[float, float]] = {}

    @staticmethod
    def _key(bbox: List[float]) -> Tuple[int, int]:
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        return ((x_min + x_max) // 2, (y_min + y_max) // 2)

    def apply(self, bbox: List[float], pitch: float, yaw: float) -> Tuple[float, float]:
        key = self._key(bbox)
        if key not in self._state:
            self._state[key] = (pitch, yaw)
            return pitch, yaw
        prev_p, prev_y = self._state[key]
        pitch = self.alpha * pitch + (1.0 - self.alpha) * prev_p
        yaw = self.alpha * yaw + (1.0 - self.alpha) * prev_y
        self._state[key] = (pitch, yaw)
        return pitch, yaw


def create_face_detector(
    backend: str,
    *,
    min_confidence: float = 0.5,
    model_selection: int = 1,
) -> Any:
    backend = backend.lower()
    if backend == "retinaface":
        from uniface import RetinaFace

        return RetinaFace()
    if backend == "mediapipe":
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is required for --detector mediapipe. "
                "Install with: pip install mediapipe"
            ) from exc
        return MediaPipeFaceDetector(
            min_confidence=min_confidence,
            model_selection=model_selection,
            mp_module=mp,
        )
    raise ValueError(f"Unknown detector: {backend}. Use retinaface or mediapipe.")


class MediaPipeFaceDetector:
    """RetinaFace-compatible wrapper around MediaPipe Face Detection."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        model_selection: int = 1,
        mp_module: Any = None,
    ) -> None:
        import mediapipe as mp

        mp = mp_module or mp
        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=model_selection,
        )

    def detect(self, frame: np.ndarray) -> List[Dict[str, List[float]]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        if not result.detections:
            return []

        h, w = frame.shape[:2]
        faces: List[Dict[str, List[float]]] = []
        for det in result.detections:
            box = det.location_data.relative_bounding_box
            x_min = int(box.xmin * w)
            y_min = int(box.ymin * h)
            x_max = int((box.xmin + box.width) * w)
            y_max = int((box.ymin + box.height) * h)
            if clamp_bbox([x_min, y_min, x_max, y_max], w, h) is None:
                continue
            faces.append({"bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]})
        return faces

    def close(self) -> None:
        self._detector.close()


def open_video_source(source: str) -> cv2.VideoCapture:
    if source.isdigit() or source == "0":
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"Cannot open video source: {source}")
    return cap
