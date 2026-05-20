# Gaze project — gaps addressed (local fork)

This document lists what was missing in the upstream **MobileGaze** checkout and what this repo adds.

## What was lacking

| Gap | Impact |
|-----|--------|
| **Many duplicate inference scripts** (`inference (1).py`, `inference_new.py`, `updated_inference.py`, …) | Hard to maintain; unclear entry point |
| **No shared runtime** | Same preprocess/decode logic copy-pasted |
| **No face-detector choice in main CLI** | MediaPipe only in side scripts |
| **No input validation** | Empty face crops and bad bboxes could crash inference |
| **VideoWriter before `isOpened()` check** | Could fail silently on bad sources |
| **No temporal smoothing** | Jittery gaze arrows on video |
| **Weights path confusion** | `.gitignore` ignores `*.pt`; no hint where to put files |
| **CUDA-only `requirements.txt`** | Breaks install on CPU-only machines |
| **No local setup guide** | Only upstream README |

## What we changed

1. **`utils/gaze_runtime.py`** — model load, preprocess, gaze decode, bbox clamp, optional EMA smoother, RetinaFace / MediaPipe factory.
2. **`inference.py`** — single recommended entry point with `--detector`, `--smooth`, safer video I/O, default `weights/<model>.pt`.
3. **`weights/README.md`** — how to download checkpoints.
4. **`requirements.txt`** — portable PyTorch pins + optional CUDA extra file.
5. **`.gitignore`** — ignore generated videos/archives, keep `weights/.gitkeep`.

## Recommended usage

```bash
cd gaze-estimation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# GPU: pip install -r requirements-cuda.txt

sh download.sh resnet34   # or place weights/resnet34.pt manually

python inference.py --model resnet34 --view --source 0
python inference.py --model resnet34 --detector mediapipe --smooth 0.35 --view --source assets/in_video.mp4 --output output.mp4
```

## Legacy scripts (not maintained)

Keep for reference only; prefer **`inference.py`**:

- `inference_multi.py` — batched multi-frame processing
- `media_pipe.py` — MediaPipe-only variant
- `updated_inference.py` — GCS / cloud workflow
- `inference (1).py`, `inference_new.py`, `onnx_inference_new.py` — experiments

## Still not in scope (future work)

- Unit tests and CI
- PR/issue migration (GitHub metadata)
- Multi-person identity tracking across frames
- Unified ONNX CLI using `gaze_runtime` decoding
