#!/usr/bin/env python3
"""
export_yolo26n.py — Download YOLO26n and export to TFLite (INT8) and CoreML.

Blueprint Part 1 (Model Export):
  - Uses Ultralytics YOLO26 (released January 2026).
  - Exports INT8-quantised TFLite for Android (NNAPI / GPU delegate).
  - Exports CoreML INT8 for iOS (Apple Neural Engine).
  - Also exports ONNX FP16 as a universal fallback for debugging.
  - Verifies each exported model with a sample inference.

Usage:
    pip install ultralytics pillow numpy
    python scripts/export_yolo26n.py [--output-dir ./models] [--verify]

Requirements:
    ultralytics>=8.3.0   (YOLO26 support)
    pillow>=10.0.0
    numpy>=1.26.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "ERROR: ultralytics is not installed.\n"
        "       Run: pip install ultralytics"
    )

try:
    from PIL import Image
except ImportError:
    sys.exit(
        "ERROR: Pillow is not installed.\n"
        "       Run: pip install pillow"
    )


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "yolo26n.pt"          # Ultralytics will auto-download on first use
INPUT_SIZE = 640                    # YOLO26n default input resolution
CONFIDENCE_THRESHOLD = 0.5         # Filter outputs below this confidence


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_or_download_model(output_dir: Path) -> YOLO:
    """Load YOLO26n from cache or download it."""
    pt_path = output_dir / MODEL_NAME
    if pt_path.exists():
        print(f"[INFO] Loading cached model from {pt_path}")
        return YOLO(str(pt_path))

    print("[INFO] Downloading YOLO26n weights…")
    model = YOLO(MODEL_NAME)           # triggers auto-download to Ultralytics cache
    # Copy to our output dir for repeatability
    import shutil
    cached = Path(model.ckpt_path)
    shutil.copy(cached, pt_path)
    print(f"[INFO] Model saved to {pt_path}")
    return model


def make_sample_image(size: int = INPUT_SIZE) -> np.ndarray:
    """Create a synthetic RGB test image (random noise)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def verify_pytorch(model: YOLO, sample: np.ndarray) -> None:
    """Run a PyTorch inference pass and print detections."""
    print("\n[VERIFY] Running PyTorch inference…")
    start = time.perf_counter()
    results = model.predict(source=sample, imgsz=INPUT_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000

    detections = results[0].boxes
    print(f"  → {len(detections)} detection(s) in {elapsed:.1f} ms")
    if len(detections):
        for box in detections[:5]:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"     class={cls}  conf={conf:.3f}  box={[f'{v:.1f}' for v in xyxy]}")


def export_tflite_int8(model: YOLO, output_dir: Path, sample: np.ndarray) -> Path:
    """
    Export YOLO26n to INT8-quantised TFLite for Android.

    INT8 quantisation reduces model size by ~4× with minimal accuracy loss
    and enables use of NNAPI / GPU delegate on Android (blueprint §Model).
    """
    print("\n[EXPORT] TFLite INT8…")
    tflite_path = model.export(
        format="tflite",
        imgsz=INPUT_SIZE,
        int8=True,
        data=None,          # uses built-in calibration dataset
        simplify=True,
        opset=17,
    )
    dest = output_dir / "yolo26n_int8.tflite"
    Path(tflite_path).rename(dest)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  → Saved: {dest}  ({size_mb:.2f} MB)")
    return dest


def verify_tflite(tflite_path: Path, sample: np.ndarray) -> None:
    """Run a TFLite inference pass and print detections."""
    try:
        import tensorflow as tf
    except ImportError:
        print("  [SKIP] TFLite verification skipped — tensorflow not installed.")
        return

    print(f"\n[VERIFY] TFLite: {tflite_path.name}…")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # Prepare input tensor (NHWC).
    input_data = sample[np.newaxis, ...]     # (1, H, W, 3)
    if in_details[0]["dtype"] == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(in_details[0]["index"], input_data)

    start = time.perf_counter()
    interpreter.invoke()
    elapsed = (time.perf_counter() - start) * 1000

    # YOLO26n (NMS-free) outputs: boxes, scores, classes
    boxes   = interpreter.get_tensor(out_details[0]["index"])
    scores  = interpreter.get_tensor(out_details[1]["index"])
    classes = interpreter.get_tensor(out_details[2]["index"])

    mask = scores[0] >= CONFIDENCE_THRESHOLD
    num  = int(mask.sum())
    print(f"  → {num} detection(s) above threshold in {elapsed:.1f} ms")


def export_coreml(model: YOLO, output_dir: Path) -> Path:
    """
    Export YOLO26n to CoreML for iOS (Apple Neural Engine).

    The resulting .mlpackage runs on the ANE via CoreML and is suitable for
    direct use in an Xcode project (blueprint §Model).
    """
    print("\n[EXPORT] CoreML INT8…")
    coreml_path = model.export(
        format="coreml",
        imgsz=INPUT_SIZE,
        int8=True,
        simplify=True,
        nms=False,          # YOLO26n is already NMS-free; don't add NMS layer
    )
    dest = output_dir / "yolo26n_int8.mlpackage"
    import shutil
    shutil.move(coreml_path, str(dest))
    size_mb = sum(
        f.stat().st_size for f in dest.rglob("*") if f.is_file()
    ) / 1024 / 1024
    print(f"  → Saved: {dest}  ({size_mb:.2f} MB)")
    return dest


def export_onnx(model: YOLO, output_dir: Path) -> Path:
    """
    Export YOLO26n to ONNX FP16 for cross-platform debugging.

    The ONNX format is the universal fallback for integration testing and
    enables profiling with tools like Netron (blueprint §Model).
    """
    print("\n[EXPORT] ONNX FP16…")
    onnx_path = model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        half=True,          # FP16
        simplify=True,
        opset=17,
        dynamic=False,
    )
    dest = output_dir / "yolo26n_fp16.onnx"
    Path(onnx_path).rename(dest)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  → Saved: {dest}  ({size_mb:.2f} MB)")
    return dest


def verify_onnx(onnx_path: Path, sample: np.ndarray) -> None:
    """Run an ONNX Runtime inference pass and print detections."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] ONNX verification skipped — onnxruntime not installed.")
        return

    print(f"\n[VERIFY] ONNX: {onnx_path.name}…")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # ONNX models expect NCHW float32
    input_data = (
        sample.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
    )

    start = time.perf_counter()
    outputs = session.run(None, {input_name: input_data})
    elapsed = (time.perf_counter() - start) * 1000

    scores = outputs[1][0]
    num    = int((scores >= CONFIDENCE_THRESHOLD).sum())
    print(f"  → {num} detection(s) above threshold in {elapsed:.1f} ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO26n to TFLite (INT8), CoreML, and ONNX (FP16).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to write exported models.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run a sample inference on each exported model.",
    )
    parser.add_argument(
        "--skip-coreml",
        action="store_true",
        default=False,
        help="Skip CoreML export (requires macOS with Xcode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  YOLO26n Export Script")
    print(f"  Output dir : {output_dir.resolve()}")
    print(f"  Input size : {INPUT_SIZE}×{INPUT_SIZE}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_or_download_model(output_dir)

    # ── Sample image for verification ─────────────────────────────────────────
    sample = make_sample_image(INPUT_SIZE)

    # ── PyTorch baseline ──────────────────────────────────────────────────────
    if args.verify:
        verify_pytorch(model, sample)

    # ── TFLite INT8 ───────────────────────────────────────────────────────────
    tflite_path = export_tflite_int8(model, output_dir, sample)
    if args.verify:
        verify_tflite(tflite_path, sample)

    # ── CoreML INT8 ───────────────────────────────────────────────────────────
    if not args.skip_coreml:
        try:
            coreml_path = export_coreml(model, output_dir)
            print(f"  CoreML model ready at {coreml_path}")
        except Exception as exc:
            print(f"  [WARN] CoreML export failed: {exc}")
            print("         CoreML export requires macOS with Xcode installed.")
    else:
        print("\n[SKIP] CoreML export (--skip-coreml flag set).")

    # ── ONNX FP16 ─────────────────────────────────────────────────────────────
    onnx_path = export_onnx(model, output_dir)
    if args.verify:
        verify_onnx(onnx_path, sample)

    print("\n" + "=" * 60)
    print("  Export complete.")
    print("  Copy yolo26n_int8.tflite → DetectionApp/assets/yolo26n.tflite")
    print("  Copy yolo26n_int8.mlpackage → your Xcode project resources")
    print("=" * 60)


if __name__ == "__main__":
    main()
