import cv2
import logging
import argparse
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

from uniface import RetinaFace

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="resnet34.pt",
        help="Path to gaze esimation model weights",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=True,
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
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing multiple faces at once (higher = better GPU utilization, more lag)",
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


def pre_process(image):
    """Preprocess single image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = transform(image)
    return image


def pre_process_batch(images):
    """Preprocess batch of images. RECOMMENDED for GPU efficiency."""
    processed = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        image = transform(image)
        processed.append(image)
    
    # Stack into batch
    image_batch = torch.stack(processed, dim=0)
    return image_batch


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = RetinaFace()  # third-party face detection library

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")
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
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            # Read batch_size frames
            batch_frames = []
            batch_faces_data = []  # List of (frame_idx, faces) tuples
            
            for _ in range(params.batch_size):
                success, frame = cap.read()
                
                if not success:
                    break
                
                frame_idx = len(batch_frames)
                batch_frames.append(frame)
                
                # Detect faces in this frame
                faces = face_detector.detect(frame)
                if faces:
                    batch_faces_data.append((frame_idx, faces))
            
            if not batch_frames:
                # EOF reached
                logging.info("Failed to obtain frame or EOF")
                break
            
            # Extract all valid face crops from all frames in batch
            all_face_crops = []
            all_face_info = []  # List of (frame_idx, bbox)
            
            for frame_idx, faces in batch_faces_data:
                for face in faces:
                    bbox = face["bbox"]
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    face_crop = batch_frames[frame_idx][y_min:y_max, x_min:x_max]
                    
                    if face_crop.size == 0:
                        continue
                    
                    all_face_crops.append(face_crop)
                    all_face_info.append((frame_idx, bbox))
            
            # Batch process all faces from all frames
            if all_face_crops:
                logging.info(f"Batch: {len(batch_frames)} frames, {len(all_face_crops)} faces total")
                
                images = pre_process_batch(all_face_crops)
                images = images.to(device)
                
                pitch, yaw = gaze_detector(images)
                pitch_predicted, yaw_predicted = (
                    F.softmax(pitch, dim=1),
                    F.softmax(yaw, dim=1),
                )
                
                # Mapping from binned to angles
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                
                # Degrees to Radians
                pitch_predicted = np.radians(pitch_predicted.cpu().numpy())
                yaw_predicted = np.radians(yaw_predicted.cpu().numpy())
                
                # Draw results on all faces
                for (frame_idx, bbox), pitch_val, yaw_val in zip(all_face_info, pitch_predicted, yaw_predicted):
                    draw_bbox_gaze(batch_frames[frame_idx], bbox, pitch_val, yaw_val)
            
            # Write and display all processed frames
            for frame in batch_frames:
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


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --ouput must be provided.")

    main(args)
