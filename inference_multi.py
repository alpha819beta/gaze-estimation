import cv2
import logging
import argparse
import warnings
import numpy as np
from collections import deque
import time

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

from uniface import RetinaFace

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-frame batch gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name")
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
        default=16,
        help="Maximum batch size for gaze estimation inference",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames to accumulate before processing",
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


class FrameBuffer:
    """Buffer to accumulate frames and their detections"""
    def __init__(self, max_size):
        self.frames = deque(maxlen=max_size)
        self.detections = deque(maxlen=max_size)
        
    def add(self, frame, faces):
        self.frames.append(frame.copy())
        self.detections.append(faces)
    
    def get_all_faces(self):
        """Extract all face crops and metadata from buffered frames"""
        face_crops = []
        metadata = []
        
        for frame_idx, (frame, faces) in enumerate(zip(self.frames, self.detections)):
            for face in faces:
                bbox = face["bbox"]
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                
                # Validate bbox
                if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0:
                    continue
                if x_max > frame.shape[1] or y_max > frame.shape[0]:
                    x_max = min(x_max, frame.shape[1])
                    y_max = min(y_max, frame.shape[0])
                
                face_crop = frame[y_min:y_max, x_min:x_max]
                
                if face_crop.size == 0:
                    continue
                
                face_crops.append(face_crop)
                metadata.append((frame_idx, bbox))
        
        return face_crops, metadata
    
    def clear(self):
        self.frames.clear()
        self.detections.clear()
    
    def __len__(self):
        return len(self.frames)


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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image_rgb)
        processed_images.append(image_tensor)
    
    if processed_images:
        return torch.stack(processed_images)
    return None


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Batch size: {params.batch_size}, Frame buffer: {params.num_frames}")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = RetinaFace()

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occurred while loading pre-trained weights. Exception: {e}")
        raise FileNotFoundError(f"Model weights not found at {params.weight}") from e

    gaze_detector.to(device)
    gaze_detector.eval()

    video_source = params.source
    if video_source.isdigit() or video_source == "0":
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(params.output, fourcc, fps, (width, height))

    if not cap.isOpened():
        raise IOError("Cannot open video source")

    # Create frame buffer
    frame_buffer = FrameBuffer(params.num_frames)
    
    # Statistics
    total_frames = 0
    total_faces = 0
    total_batches = 0
    start_time = time.time()
    
    with torch.no_grad():
        while True:
            # Read and accumulate frames
            for _ in range(params.num_frames):
                success, frame = cap.read()
                if not success:
                    break
                
                total_frames += 1
                
                # Detect faces
                faces = face_detector.detect(frame)
                frame_buffer.add(frame, faces)
            
            if len(frame_buffer) == 0:
                logging.info("No more frames to process")
                break
            
            # Extract all face crops from buffered frames
            face_crops, metadata = frame_buffer.get_all_faces()
            total_faces += len(face_crops)
            
            if not face_crops:
                frame_buffer.clear()
                continue
            
            # Store results for each frame
            frame_results = [[] for _ in range(len(frame_buffer.frames))]
            
            # Process all faces in batches
            for i in range(0, len(face_crops), params.batch_size):
                batch_crops = face_crops[i:i + params.batch_size]
                batch_meta = metadata[i:i + params.batch_size]
                
                # Preprocess batch
                image_batch = pre_process_batch(batch_crops)
                if image_batch is None:
                    continue
                
                image_batch = image_batch.to(device)
                
                # Batch inference
                pitch_batch, yaw_batch = gaze_detector(image_batch)
                
                pitch_predicted = F.softmax(pitch_batch, dim=1)
                yaw_predicted = F.softmax(yaw_batch, dim=1)
                
                # Convert from binned to angles
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                
                # Degrees to Radians
                pitch_predicted = np.radians(pitch_predicted.cpu().numpy())
                yaw_predicted = np.radians(yaw_predicted.cpu().numpy())
                
                # Store results by frame
                for j, (frame_idx, bbox) in enumerate(batch_meta):
                    frame_results[frame_idx].append((bbox, pitch_predicted[j], yaw_predicted[j]))
                
                total_batches += 1
            
            # Draw and output frames
            for frame_idx, (frame, results) in enumerate(zip(frame_buffer.frames, frame_results)):
                for bbox, pitch, yaw in results:
                    draw_bbox_gaze(frame, bbox, pitch, yaw)
                
                if params.output:
                    out.write(frame)
                
                if params.view:
                    cv2.imshow("Demo", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        if params.output:
                            out.release()
                        cv2.destroyAllWindows()
                        return
            
            # Clear buffer for next batch of frames
            frame_buffer.clear()
    
    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()
    
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