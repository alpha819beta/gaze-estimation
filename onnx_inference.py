# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse
import logging
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from uniface import RetinaFace

from config import data_config
from utils.gaze_runtime import crop_face, open_video_source, resolve_dataset_config
from utils.helpers import draw_bbox_gaze

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class GazeEstimationONNX:
    """Gaze estimation using ONNX Runtime (logits to radians)."""

    def __init__(
        self,
        model_path: str,
        dataset: str = "gaze360",
        session: ort.InferenceSession = None,
    ) -> None:
        cfg = resolve_dataset_config(dataset)
        self._bins = cfg["bins"]
        self._binwidth = cfg["binwidth"]
        self._angle_offset = cfg["angle"]
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        self.session = session
        if self.session is None:
            if not model_path:
                raise ValueError("model_path is required when session is not provided")
            self.session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = tuple(input_cfg.shape[2:][::-1])

        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        if len(self.output_names) != 2:
            raise ValueError(f"Expected 2 outputs, got {len(self.output_names)}")

    def preprocess_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        processed = []
        for image in face_images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.input_size)
            image = image.astype(np.float32) / 255.0
            mean = np.array(self.input_mean, dtype=np.float32)
            std = np.array(self.input_std, dtype=np.float32)
            image = (image - mean) / std
            processed.append(np.transpose(image, (2, 0, 1)))
        return np.stack(processed, axis=0).astype(np.float32)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def decode(self, pitch_logits: np.ndarray, yaw_logits: np.ndarray) -> List[Tuple[float, float]]:
        pitch_probs = self.softmax(pitch_logits)
        yaw_probs = self.softmax(yaw_logits)
        pitch = np.sum(pitch_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        return list(zip(np.radians(pitch), np.radians(yaw)))

    def estimate_batch(self, face_images: List[np.ndarray]) -> List[Tuple[float, float]]:
        if not face_images:
            return []
        batch = self.preprocess_batch(face_images)
        outputs = self.session.run(self.output_names, {self.input_name: batch})
        return self.decode(outputs[0], outputs[1])


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze Estimation ONNX Inference")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video path or camera index (e.g., 0 for webcam)",
    )
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--output", type=str, default="", help="Save annotated video to this path")
    parser.add_argument("--view", action="store_true", help="Show live preview window")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaze360",
        choices=list(data_config.keys()),
        help="Dataset bin config (must match how the ONNX model was trained)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        dest="batch_size",
        help="Frames read per loop (more = higher throughput, more latency)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.view and not args.output:
        raise SystemExit("Provide at least one of --view or --output")

    cap = open_video_source(args.source)
    engine = GazeEstimationONNX(model_path=args.model, dataset=args.dataset)
    detector = RetinaFace()

    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    stop = False
    try:
        while cap.isOpened() and not stop:
            batch_frames = []
            batch_faces_data = []

            for _ in range(args.batch_size):
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx = len(batch_frames)
                batch_frames.append(frame)
                faces = detector.detect(frame)
                if faces:
                    batch_faces_data.append((frame_idx, faces))

            if not batch_frames:
                break

            all_face_crops = []
            all_face_info = []

            for frame_idx, faces in batch_faces_data:
                for face in faces:
                    bbox = face["bbox"]
                    crop = crop_face(batch_frames[frame_idx], bbox)
                    if crop is None:
                        continue
                    all_face_crops.append(crop)
                    all_face_info.append((frame_idx, bbox))

            if all_face_crops:
                logger.info(
                    "Batch: %d frames, %d faces",
                    len(batch_frames),
                    len(all_face_crops),
                )
                results = engine.estimate_batch(all_face_crops)
                for (frame_idx, bbox), (pitch, yaw) in zip(all_face_info, results):
                    draw_bbox_gaze(batch_frames[frame_idx], bbox, pitch, yaw)

            for frame in batch_frames:
                if writer is not None:
                    writer.write(frame)
                if args.view:
                    cv2.imshow("Gaze estimation (ONNX)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop = True
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
