import cv2
import logging
import argparse
import warnings
import numpy as np
import time

import torch
import torch.nn.functional as F

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

try:
    from mediapipe.python.solutions import face_detection as mp_face_detection
except ImportError:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-optimized gaze estimation")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name")
    parser.add_argument("--weight", type=str, default="resnet34.pt", help="Path to weights")
    parser.add_argument("--view", action="store_true", default=False, help="Display results")
    parser.add_argument("--source", type=str, default="assets/in_video.mp4", help="Video source")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return args


def preprocess_gpu_batch(images, device):
    """Fast GPU-based preprocessing
    
    Converts list of numpy images to normalized tensor batch on GPU
    """
    # Convert to tensors and stack
    tensors = []
    for img in images:
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize using cv2 (faster than PIL)
        img_resized = cv2.resize(img_rgb, (448, 448), interpolation=cv2.INTER_LINEAR)
        # Convert to tensor (HWC -> CHW)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        tensors.append(tensor)
    
    # Stack into batch
    batch = torch.stack(tensors).to(device)
    
    # Normalize on GPU
    batch = batch / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    
    return batch


def extract_mediapipe_faces(frame, detection_result):
    """Extract face crops and bboxes from MediaPipe detection results"""
    face_crops = []
    bboxes = []
    
    if detection_result.detections:
        h, w, _ = frame.shape
        
        for detection in detection_result.detections:
            bbox_data = detection.location_data.relative_bounding_box
            
            x_min = int(bbox_data.xmin * w)
            y_min = int(bbox_data.ymin * h)
            bbox_width = int(bbox_data.width * w)
            bbox_height = int(bbox_data.height * h)
            
            x_max = x_min + bbox_width
            y_max = y_min + bbox_height
            
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
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
    logging.info(f"Using GPU preprocessing")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    # Initialize MediaPipe
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    # Load gaze model
    gaze_detector = get_model(params.model, params.bins, inference_mode=True)
    state_dict = torch.load(params.weight, map_location=device)
    gaze_detector.load_state_dict(state_dict)
    gaze_detector.to(device)
    gaze_detector.eval()
    logging.info("Models loaded")

    # Open video
    video_source = params.source
    if video_source.isdigit() or video_source == "0":
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open video source")

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
                break
            
            total_frames += 1
            
            # Detect faces
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = face_detector.process(frame_rgb)
            face_crops, bboxes = extract_mediapipe_faces(frame, detection_result)
            
            total_faces += len(face_crops)
            
            if not face_crops:
                if params.output:
                    out.write(frame)
                if params.view:
                    cv2.imshow("Demo", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            
            # Process in batches with GPU preprocessing
            results = []
            
            for i in range(0, len(face_crops), params.batch_size):
                batch_crops = face_crops[i:i + params.batch_size]
                batch_bboxes = bboxes[i:i + params.batch_size]
                
                # GPU-accelerated preprocessing
                image_batch = preprocess_gpu_batch(batch_crops, device)
                
                # Inference
                pitch_batch, yaw_batch = gaze_detector(image_batch)
                
                pitch_predicted = F.softmax(pitch_batch, dim=1)
                yaw_predicted = F.softmax(yaw_batch, dim=1)
                
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                
                pitch_predicted = np.radians(pitch_predicted.cpu().numpy())
                yaw_predicted = np.radians(yaw_predicted.cpu().numpy())
                
                for j, bbox in enumerate(batch_bboxes):
                    results.append((bbox, pitch_predicted[j], yaw_predicted[j]))
                
                total_batches += 1
            
            # Draw results
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
    logging.info(f"Total frames: {total_frames}")
    logging.info(f"Total faces: {total_faces}")
    logging.info(f"Total GPU batches: {total_batches}")
    logging.info(f"Processing time: {elapsed_time:.2f}s")
    logging.info(f"FPS: {total_frames/elapsed_time:.2f}")
    if total_frames > 0:
        logging.info(f"Avg faces/frame: {total_faces/total_frames:.2f}")
    if total_batches > 0:
        logging.info(f"Avg batch size: {total_faces/total_batches:.2f}")
    logging.info(f"{'='*50}")


if __name__ == "__main__":
    args = parse_args()
    if not args.view and not args.output:
        raise Exception("At least one of --view or --output must be provided.")
    main(args)