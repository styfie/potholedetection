"""Inference layer shared by image, video, and browser camera."""
from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parent

MODELS = {
    ("YOLO26", "Nano (n)"): ("YOLO26n", "yolo26n.pt"),
    ("YOLO26", "Small (s)"): ("YOLO26s", "yolo26s.pt"),
    ("YOLO26", "Medium (m)"): ("YOLO26m", "yolo26m.pt"),
    ("YOLO26", "Large (l)"): ("YOLO26l", "yolo26l.pt"),
    ("RT-DETR", "Large (L)"): ("RT-DETR-L", "rtdetr-l.pt"),
}
SCENARIOS = {"pure", "domain_adapt", "combined"}
VIDEO_TARGET_FPS = 10.0
VIDEO_IMGSZ = 640
VIDEO_IOU = 0.50
INFERENCE_LOCK = threading.RLock()


def device():
    return 0 if torch.cuda.is_available() else "cpu"


def device_label():
    return "CUDA" if torch.cuda.is_available() else "CPU"


@st.cache_resource(show_spinner=False, max_entries=15)
def load_model(model_family: str, variant: str, scenario: str = "pure"):
    if scenario not in SCENARIOS:
        raise ValueError(f"Training scenario tidak dikenal: {scenario}")
    model_name, filename = MODELS[(model_family, variant)]
    checkpoint = ROOT / "models" / scenario / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint {model_name} untuk skenario '{scenario}' tidak ditemukan: "
            f"{checkpoint}"
        )
    model = RTDETR(str(checkpoint)) if model_family == "RT-DETR" else YOLO(str(checkpoint))
    return model


def predict(image_bgr, model, confidence: float, imgsz: int | None = None):
    kwargs = dict(
        source=image_bgr, conf=confidence, iou=VIDEO_IOU,
        device=device(), verbose=False,
    )
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    with INFERENCE_LOCK:
        results = model.predict(**kwargs)
    if not results:
        raise RuntimeError("Model tidak mengembalikan hasil deteksi.")
    result = results[0]
    count = len(result.boxes) if result.boxes is not None else 0
    ms = result.speed.get("inference") if result.speed else None
    return result.plot(), count, ms


def decode_image(data: bytes):
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Gambar tidak dapat dibaca.")
    return frame


def process_video(video_bytes: bytes, suffix: str, model, confidence: float,
                  progress_callback=None):
    """Sample at <=10 FPS, encode browser-compatible MP4, preserve source duration."""
    if suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise ValueError("Format video tidak didukung.")
    with tempfile.TemporaryDirectory(prefix="pothole_") as folder:
        folder = Path(folder)
        source = folder / ("input" + suffix.lower())
        raw = folder / "raw.mp4"
        output = folder / "result.mp4"
        source.write_bytes(video_bytes)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError("Video tidak dapat dibuka.")
        writer = None
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise ValueError("Resolusi video tidak valid.")
            target_fps = min(fps, VIDEO_TARGET_FPS)
            interval = max(1, round(fps / target_fps))
            sampled_fps = fps / interval
            writer = cv2.VideoWriter(
                str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                sampled_fps, (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("Tidak dapat membuat video sementara.")
            start = time.perf_counter()
            processed = detections = frames_with_detections = 0
            inference_sum = 0.0
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % interval == 0:
                    annotated, count, ms = predict(frame, model, confidence, VIDEO_IMGSZ)
                    writer.write(annotated)
                    processed += 1
                    detections += count
                    frames_with_detections += int(count > 0)
                    inference_sum += float(ms or 0.0)
                    if progress_callback and processed % 5 == 0 and frames > 0:
                        progress_callback(min(frame_index / frames, 0.99))
                frame_index += 1
            if processed == 0:
                raise ValueError("Tidak ada frame video yang dapat diproses.")
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(output),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError("FFmpeg gagal mengonversi video: " + run.stderr[-500:])
        elapsed = (time.perf_counter() - start) * 1000
        return output.read_bytes(), {
            "detections": detections,
            "frames_with_detections": frames_with_detections,
            "processed_frames": processed,
            "source_frames": frames,
            "source_fps": fps,
            "output_fps": sampled_fps,
            "frame_interval": interval,
            "inference_time_ms": inference_sum / processed,
            "processing_time_ms": elapsed,
            "device": device_label(),
        }
