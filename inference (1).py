import cv2
import json
import logging
import argparse
import warnings
import numpy as np
from google.cloud import storage
import torch
import torch.nn.functional as F
from torchvision import transforms
import tempfile
import os
from pathlib import Path
from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

from uniface import RetinaFace

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_gcs_path(gcs_path):
    """
    Parse GCS path and extract bucket name and blob path.
    Supports formats:
    - gs://bucket-name/path/to/file.mp4
    - bucket-name/path/to/file.mp4
    """
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]  # Remove gs:// prefix
    
    parts = gcs_path.split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    
    return bucket_name, blob_path


def download_from_gcs(bucket_name, blob_path, local_path, project=None):
    """Download file from GCS to local path"""
    logging.info(f"Downloading gs://{bucket_name}/{blob_path} to {local_path}")
    
    if project:
        storage_client = storage.Client(project=project)
    else:
        storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    
    logging.info(f"Download complete: {local_path}")


def upload_to_gcs(local_path, bucket_name, blob_path, project=None):
    """Upload file from local path to GCS"""
    logging.info(f"Uploading {local_path} to gs://{bucket_name}/{blob_path}")
    
    if project:
        storage_client = storage.Client(project=project)
    else:
        storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    
    logging.info(f"Upload complete: gs://{bucket_name}/{blob_path}")


def get_output_paths(input_gcs_path):
    """
    Convert input path to output paths.
    Input: paper-recordings/66ecefjker454/234324235.mp4
    Outputs:
    - paper-eye-engagement/66ecefjker454/234324235/gaze_result.mp4
    - paper-eye-engagement/66ecefjker454/234324235/gaze_result.json
    - paper-eye-engagement/66ecefjker454/234324235/gaze_result.txt
    """
    # Remove gs:// if present
    if input_gcs_path.startswith("gs://"):
        input_gcs_path = input_gcs_path[5:]
    
    # Parse the input path: bucket/folder1/folder2/filename.mp4
    parts = input_gcs_path.split("/")
    
    # Assuming format: paper-recordings/folder1/filename.mp4
    if len(parts) < 3:
        raise ValueError(f"Invalid input path format: {input_gcs_path}")
    
    input_bucket = parts[0]
    folder1 = parts[1]  # e.g., 66ecefjker454
    filename_with_ext = parts[2]  # e.g., 234324235.mp4
    filename_without_ext = Path(filename_with_ext).stem  # e.g., 234324235
    
    # Create output paths
    output_bucket = "paper-dev-program-cycle-gaze-engagement"
    output_base_path = f"{folder1}/{filename_without_ext}"
    
    return {
        "bucket": output_bucket,
        "video": f"{output_base_path}/gaze_result.mp4",
        "json": f"{output_base_path}/gaze_result.json",
        "txt": f"{output_base_path}/gaze_result.txt"
    }


def read_file_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    return content


def write_file_to_gcs(bucket_name, blob_name, content):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content)
    print(f"File {blob_name} uploaded to {bucket_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="resnet34.pt",
        help="Path to gaze estimation model weights",
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
        "--angles-output",
        type=str,
        default="gaze_angles.txt",
        help="Path to save frame-by-frame angles"
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="gaze_statistics.json",
        help="Path to save statistics JSON"
    )
    parser.add_argument(
        "--gcs_project1",
        type=str,
        help="GCP project ID for input bucket (optional if using default credentials)"
    )
    parser.add_argument(
        "--gcs_project2",
        type=str,
        help="GCP project ID for output bucket (optional if using default credentials)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum face detection confidence threshold (0.0 - 1.0), default 0.5"
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def get_angle_bin(angle_degrees):
    """Classify angle into one of 7 bins."""
    if angle_degrees < -75:
        return 0
    elif angle_degrees < -45:
        return 1
    elif angle_degrees < -15:
        return 2
    elif angle_degrees < 15:
        return 3
    elif angle_degrees < 45:
        return 4
    elif angle_degrees < 75:
        return 5
    else:
        return 6


def get_bin_label(bin_index):
    """Get label for bin index"""
    bin_labels = [
        "[-90, -75)",
        "[-75, -45)",
        "[-45, -15)",
        "[-15, 15)",
        "[15, 45)",
        "[45, 75]",
        "[75, 95]"
    ]
    return bin_labels[bin_index]


temp_dir = tempfile.mkdtemp()


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = RetinaFace()
    input_bucket, input_blob = parse_gcs_path(params.source)
    print(f"Processing GCS path: {params.source}")
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

    local_video_path = os.path.join(temp_dir, "input_video.mp4")
    download_from_gcs(input_bucket, input_blob, local_video_path, params.gcs_project1)

    # Get output paths
    output_paths = get_output_paths(params.source)

    # Local paths for output files
    local_output_video = os.path.join(temp_dir, "gaze_result.mp4")
    local_output_json = os.path.join(temp_dir, "gaze_result.json")
    local_output_txt = os.path.join(temp_dir, "gaze_result.txt")

    # Open video
    cap = cv2.VideoCapture(local_video_path)

    if not cap.isOpened():
        raise IOError("Cannot open video file")

    # Get video FPS and calculate skip rate
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 3
    skip_frames = int(video_fps / target_fps)
    print(f"Video FPS: {video_fps}")
    print(f"Processing every {skip_frames} frames ({target_fps} FPS)")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # mp4v is broadly supported across OpenCV builds
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(local_output_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # Disable view mode automatically if no display is available (e.g. headless server)
    view = bool(os.environ.get("DISPLAY"))
    if not view:
        logging.info("No display found, running in headless mode.")

    # Precompute the right boundary of the left 30% region for face detection
    left_region_width = int(width * 0.3)

    # Initialize statistics tracking
    frame_count = 0
    processed_frames = 0
    no_speaker_detected = False
    angle_frequency = np.zeros((7, 7), dtype=int)  # 7x7 grid for pitch and yaw bins

    # Open file for writing angles
    angles_file = open(local_output_txt, 'w')
    angles_file.write("Frame,Pitch(degrees),Yaw(degrees)\n")

    with torch.no_grad():
        while True:
            success, frame = cap.read()

            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            frame_count += 1

            if frame_count % skip_frames != 0:
                continue

            # Run face detection only on the left 30% of the frame
            left_region = frame[:, :left_region_width]
            faces = face_detector.detect(left_region)

            # Track processed frame attempts for the no-speaker early-exit check
            processed_frames += 1

            if processed_frames <= 250 and len(faces) == 0:
                if processed_frames == 250 and angle_frequency.sum() == 0:
                    logging.info("No speaker detected in first 250 processed frames. Returning empty.")
                    no_speaker_detected = True
                    break
                continue

            if len(faces) == 0:
                continue

            # Filter faces by confidence threshold
            faces = [f for f in faces if f.get("confidence", f.get("score", 0)) >= 0.9]

            if len(faces) == 0:
                continue

            # Select the topmost face (smallest y_min)
            best_face = min(faces, key=lambda x: x["bbox"][1])

            bbox = best_face["bbox"]
            x_min, y_min, x_max, y_max = map(int, bbox[:4])

            # Ensure bbox is within the left region boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(left_region_width, x_max)
            y_max = min(height, y_max)

            # Check if bbox is valid
            if x_min >= x_max or y_min >= y_max:
                out.write(frame)
                continue

            image = left_region[y_min:y_max, x_min:x_max]

            # Check if crop is empty
            if image.size == 0:
                out.write(frame)
                continue

            # Move tensor to the same device as the model
            image = pre_process(image).to(device)

            pitch, yaw = gaze_detector(image)

            pitch_predicted, yaw_predicted = (
                F.softmax(pitch, dim=1),
                F.softmax(yaw, dim=1),
            )

            # Mapping from binned (softmax) to continuous angles
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * 4 - 180

            # Degrees to Radians for visualization
            pitch_radians = np.radians(pitch_predicted.cpu())
            yaw_radians = np.radians(yaw_predicted.cpu())

            # Get degrees for statistics
            pitch_degrees = pitch_predicted.cpu().item()
            yaw_degrees = yaw_predicted.cpu().item()

            # Write to angles file
            angles_file.write(f"{frame_count},{pitch_degrees:.2f},{yaw_degrees:.2f}\n")

            # Update frequency statistics
            pitch_bin = get_angle_bin(pitch_degrees)
            yaw_bin = get_angle_bin(yaw_degrees)
            angle_frequency[pitch_bin, yaw_bin] += 1

            # Draw box and gaze direction on the full frame
            # (bbox coords are relative to left_region, which is fine for draw_bbox_gaze)
            draw_bbox_gaze(frame, bbox, pitch_radians, yaw_radians)

            # if params.output:
            #     out.write(frame)

            # if view:
            #     cv2.imshow("Demo", frame)
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         break

    # Close angles file
    angles_file.close()

    # If no speaker was detected, upload empty outputs and exit early
    if no_speaker_detected:
        empty_stats = {"no_speaker": True, "total_frames_processed": 0}
        with open(local_output_json, 'w') as json_file:
            json.dump(empty_stats, json_file, indent=2)
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("No speaker detected. Empty results uploaded.")
        return empty_stats

    # --- Engagement score calculation ---
    def neighbors(indices, shape, distance):
        """Return all indices exactly 'distance' steps away (Chebyshev shell)."""
        i, j = indices
        r, c = shape
        idxs = []
        for di in range(-distance, distance + 1):
            for dj in range(-distance, distance + 1):
                if max(abs(di), abs(dj)) == distance:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < r and 0 <= nj < c:
                        idxs.append((ni, nj))
        return idxs
    frames_with_face = int(angle_frequency.sum())
    heatmap_percent = (angle_frequency / frames_with_face * 100) if frames_with_face > 0 else angle_frequency.astype(float)
    total_percentage = heatmap_percent.sum()
    max_idx = np.unravel_index(np.argmax(heatmap_percent), heatmap_percent.shape)

    score_array = np.zeros_like(heatmap_percent)
    visited = set()
    for step in range(4):
        weight = 1 / (2 ** step)
        idxs = [max_idx] if step == 0 else neighbors(max_idx, heatmap_percent.shape, step)
        for idx in idxs:
            if idx not in visited:
                score_array[idx] = heatmap_percent[idx] * weight
                visited.add(idx)

    engagement_score = float(score_array.sum() / total_percentage) if total_percentage > 0 else 0.0

    # Generate statistics JSON
    statistics = {
        "total_frames_processed": processed_frames,
        "total_frames_in_video": frame_count,
        "target_fps": target_fps,
        "engagement_score": round(engagement_score, 4),
        "sections": {}
    }

    # Calculate percentages and create section labels
    for pitch_idx in range(7):
        for yaw_idx in range(7):
            section_name = f"Pitch:{get_bin_label(pitch_idx)}_Yaw:{get_bin_label(yaw_idx)}"
            count = int(angle_frequency[pitch_idx, yaw_idx])
            percentage = (count / frames_with_face * 100) if frames_with_face > 0 else 0

            statistics["sections"][section_name] = {
                "count": count,
                "percentage": round(percentage, 2)
            }

    # Save statistics to JSON
    # with open(local_output_json, 'w') as json_file:
    #     json.dump(statistics, json_file, indent=2)

    cap.release()
    # if params.output:
    #     out.release()
    cv2.destroyAllWindows()


    print(f"\nProcessing complete!")
    print(f"Total frames processed: {processed_frames}/{frame_count}")
    print(f"Engagement score: {engagement_score:.4f}")
    return statistics


if __name__ == "__main__":
    args = parse_args()
    print(main(args))