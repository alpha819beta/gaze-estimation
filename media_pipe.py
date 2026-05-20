import cv2
import logging
import argparse
import warnings
import numpy as np
import time

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe not found. Install with: pip install mediapipe")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation with MediaPipe face detection")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="resnet34.pt",
        help="Path to gaze estimation model weights",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=False,
        help="Display the inference results",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="assets/in_video.mp4",
        help="Path to source video file or camera index",
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaze360",
        help="Dataset name to get dataset related configs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for gaze estimation (number of faces processed simultaneously)",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for face detection (0.0-1.0)",
    )
    parser.add_argument(
        "--model-selection",
        type=int,
        default=1,
        choices=[0, 1],
        help="MediaPipe model: 0 for short-range (within 2m), 1 for full-range (within 5m)",
    )
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def pre_process_batch(images):
    """Process a batch of face images"""
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    processed_images = []
    for image in images:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb)
            processed_images.append(image_tensor)
        except Exception as e:
            logging.warning(f"Failed to preprocess image: {e}")
            continue
    
    if processed_images:
        return torch.stack(processed_images)
    return None


def extract_mediapipe_faces(frame, detection_result):
    """Extract face crops and bboxes from MediaPipe detection results
    
    Args:
        frame: Input frame (BGR)
        detection_result: MediaPipe detection result
        
    Returns:
        face_crops: List of face crop images
        bboxes: List of bounding boxes in format [x_min, y_min, x_max, y_max]
    """
    face_crops = []
    bboxes = []
    
    if detection_result.detections:
        h, w, _ = frame.shape
        
        for detection in detection_result.detections:
            # Get bounding box (MediaPipe returns normalized coordinates)
            bbox_data = detection.location_data.relative_bounding_box
            
            # Convert to absolute pixel coordinates
            x_min = int(bbox_data.xmin * w)
            y_min = int(bbox_data.ymin * h)
            bbox_width = int(bbox_data.width * w)
            bbox_height = int(bbox_data.height * h)
            
            x_max = x_min + bbox_width
            y_max = y_min + bbox_height
            
            # Clip to frame boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            # Validate bbox
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Extract face crop
            face_crop = frame[y_min:y_max, x_min:x_max]
            
            if face_crop.size == 0:
                continue
            
            face_crops.append(face_crop)
            bboxes.append([x_min, y_min, x_max, y_max])
    
    return face_crops, bboxes


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Batch size: {params.batch_size}")
    logging.info(f"Face detector: MediaPipe (model_selection={params.model_selection})")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=params.model_selection,
        min_detection_confidence=params.detection_confidence
    )
    logging.info(f"MediaPipe face detection initialized (confidence threshold: {params.detection_confidence})")

    # Load gaze estimation model
    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.error(f"Exception occurred while loading pre-trained weights of gaze estimation model. Exception: {e}")
        raise FileNotFoundError(f"Model weights not found at {params.weight}") from e

    gaze_detector.to(device)
    gaze_detector.eval()

    # Open video source
    video_source = params.source
    if video_source.isdigit() or video_source == "0":
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open video source")

    # Setup video writer
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(params.output, fourcc, fps, (width, height))

    # Statistics
    total_frames = 0
    total_faces = 0
    total_batches = 0
    start_time = time.time()
    
    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Failed to obtain frame or EOF")
                break
            
            total_frames += 1
            
            # Face detection with MediaPipe (expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = face_detector.process(frame_rgb)
            
            # Extract faces from detection results
            face_crops, bboxes = extract_mediapipe_faces(frame, detection_result)
            total_faces += len(face_crops)
            
            if not face_crops:
                # No faces detected, write frame and continue
                if params.output:
                    out.write(frame)
                if params.view:
                    cv2.imshow("Demo", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            
            # Process faces in batches
            results = []
            
            for i in range(0, len(face_crops), params.batch_size):
                batch_crops = face_crops[i:i + params.batch_size]
                batch_bboxes = bboxes[i:i + params.batch_size]
                
                # Preprocess batch
                image_batch = pre_process_batch(batch_crops)
                if image_batch is None:
                    continue
                
                image_batch = image_batch.to(device)
                
                # Batch inference
                pitch_batch, yaw_batch = gaze_detector(image_batch)
                
                pitch_predicted = F.softmax(pitch_batch, dim=1)
                yaw_predicted = F.softmax(yaw_batch, dim=1)
                
                # Mapping from binned to angles
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                
                # Degrees to Radians
                pitch_predicted = np.radians(pitch_predicted.cpu().numpy())
                yaw_predicted = np.radians(yaw_predicted.cpu().numpy())
                
                # Store results
                for j, bbox in enumerate(batch_bboxes):
                    results.append((bbox, pitch_predicted[j], yaw_predicted[j]))
                
                total_batches += 1
            
            # Draw results on frame
            for bbox, pitch, yaw in results:
                draw_bbox_gaze(frame, bbox, pitch, yaw)
            
            if params.output:
                out.write(frame)

            if params.view:
                cv2.imshow("Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()
    face_detector.close()
    
    elapsed_time = time.time() - start_time
    
    logging.info(f"\n{'='*50}")
    logging.info(f"Processing Statistics:")
    logging.info(f"{'='*50}")
    logging.info(f"Total frames processed: {total_frames}")
    logging.info(f"Total faces detected: {total_faces}")
    logging.info(f"Total GPU batches: {total_batches}")
    logging.info(f"Processing time: {elapsed_time:.2f}s")
    logging.info(f"FPS: {total_frames/elapsed_time:.2f}")
    if total_frames > 0:
        logging.info(f"Average faces per frame: {total_faces/total_frames:.2f}")
    if total_batches > 0:
        logging.info(f"Average batch size: {total_faces/total_batches:.2f}")
    logging.info(f"{'='*50}")


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --output must be provided.")

    main(args)