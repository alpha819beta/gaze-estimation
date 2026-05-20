#!/usr/bin/env python3
"""Gaze estimation on webcam or video (PyTorch)."""

import argparse
import logging
import warnings

import cv2
import numpy as np
import torch

from config import data_config
from utils.gaze_runtime import (
    FacePreprocessor,
    GazeSmoother,
    create_face_detector,
    crop_face,
    decode_gaze,
    default_weight_path,
    load_gaze_model,
    open_video_source,
    resolve_dataset_config,
)
from utils.helpers import draw_bbox_gaze

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
        help="Architecture: resnet18/34/50, mobilenetv2, mobileone_s0",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="",
        help="Path to .pt weights (default: weights/<model>.pt)",
    )
    parser.add_argument("--view", action="store_true", help="Show live preview window")
    parser.add_argument(
        "--source",
        type=str,
        default="assets/in_video.mp4",
        help="Video path or camera index (e.g. 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Save annotated video to this path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaze360",
        choices=list(data_config.keys()),
        help="Dataset config for angle bins",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=["retinaface", "mediapipe"],
        help="Face detector backend",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        metavar="ALPHA",
        help="Gaze EMA smoothing 0=off, 0.2-0.5 recommended",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe min detection confidence (ignored for retinaface)",
    )
    args = parser.parse_args()

    cfg = resolve_dataset_config(args.dataset)
    args.bins = cfg["bins"]
    args.binwidth = cfg["binwidth"]
    args.angle = cfg["angle"]

    if not args.weight:
        args.weight = str(default_weight_path(args.model))

    return args


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)
    preprocessor = FacePreprocessor()
    smoother = GazeSmoother(params.smooth) if params.smooth > 0 else None

    face_detector = create_face_detector(
        params.detector,
        min_confidence=params.detection_confidence,
    )
    gaze_model = load_gaze_model(params.model, params.weight, params.bins, device)

    cap = open_video_source(params.source)
    writer = None
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(params.output, fourcc, fps, (width, height))

    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.info("End of stream or read failure")
                    break

                faces = face_detector.detect(frame)
                for face in faces:
                    bbox = face["bbox"]
                    crop = crop_face(frame, bbox)
                    if crop is None:
                        continue

                    batch = preprocessor(crop).to(device)
                    pitch_logits, yaw_logits = gaze_model(batch)
                    pitch, yaw = decode_gaze(
                        pitch_logits, yaw_logits, idx_tensor, params.binwidth, params.angle
                    )
                    if smoother is not None:
                        pitch, yaw = smoother.apply(bbox, pitch, yaw)

                    draw_bbox_gaze(frame, bbox, np.array(pitch), np.array(yaw))

                if writer is not None:
                    writer.write(frame)
                if params.view:
                    cv2.imshow("Gaze estimation", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if hasattr(face_detector, "close"):
            face_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    if not args.view and not args.output:
        raise SystemExit("Provide at least one of --view or --output")
    main(args)
