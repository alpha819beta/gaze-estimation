# Model weights

Weight files (`*.pt`, `*.onnx`) are **not** committed (see root `.gitignore`).

## Download (Linux / Git Bash)

```bash
sh download.sh resnet34
```

Available names: `resnet18`, `resnet34`, `resnet50`, `mobilenetv2`, `mobileone_s0`.

## Manual download

See the [releases page](https://github.com/yakhyo/gaze-estimation/releases/tag/weights) and place files here, e.g.:

- `weights/resnet34.pt`
- `weights/resnet34_gaze.onnx`

## Run inference

```bash
python inference.py --model resnet34 --weight weights/resnet34.pt --view --source 0
```

If `--weight` is omitted, `inference.py` defaults to `weights/<model>.pt`.
