"""Pothole Detection System — self-contained Streamlit interface.

Keep inference.py and model checkpoints in the same locations as before.
No external style.css is required.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import av
import cv2
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from inference import MODELS, decode_image, device_label, load_model, predict, process_video

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Pothole Detection System", page_icon="🛣️", layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------------------
# Per-image evaluation settings
# ---------------------------------------------------------------------------
METRIC_NAMES = ["Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]
# Folder label dicari di dalam repo dulu, lalu di folder induknya (untuk jalan lokal lama).
LABEL_DIR_CANDIDATES = [ROOT / "selected-data-display", ROOT.parent / "selected-data-display"]
LABEL_DIR = next((p for p in LABEL_DIR_CANDIDATES if p.is_dir()), LABEL_DIR_CANDIDATES[0])
IOU_MATCH_THRESHOLD = 0.50
AP_CONFIDENCE_FLOOR = 0.001


def find_label_file(image_name: str):
    """Cari .txt berdasarkan nama dasar gambar; None jika tidak ada."""
    stem = Path(image_name).stem
    exact = LABEL_DIR / f"{stem}.txt"
    if exact.is_file():
        return exact
    if LABEL_DIR.is_dir():  # fallback tanpa membedakan huruf besar/kecil (Linux case-sensitive)
        for candidate in LABEL_DIR.glob("*.txt"):
            if candidate.stem.lower() == stem.lower():
                return candidate
    return None


def load_yolo_labels(image_name: str, image_width: int, image_height: int):
    """Load label YOLO. Mengembalikan None jika file label tidak ada."""
    label_path = find_label_file(image_name)
    if label_path is None:
        return None

    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Format label tidak valid pada baris {line_number}: {line!r}")
        class_id, x_center, y_center, width, height = map(float, parts[:5])
        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height
        boxes.append({"class_id": int(class_id), "xyxy": [x1, y1, x2, y2]})
    return boxes


def box_iou_xyxy(box, boxes):
    """IoU between one xyxy box and a list/array of xyxy boxes."""
    import numpy as np

    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return np.zeros((0,), dtype=float)
    box = np.asarray(box, dtype=float)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def evaluate_single_image(result, ground_truth, confidence_threshold: float):
    """
    Calculate image-level P/R/F1 and single-image AP.
    TP/FP/FN use the selected confidence threshold and IoU=0.50.
    AP uses detections down to AP_CONFIDENCE_FLOOR and confidence ranking.
    """
    import numpy as np

    pred_boxes = result.boxes
    if pred_boxes is None or len(pred_boxes) == 0:
        predictions = []
    else:
        xyxy = pred_boxes.xyxy.detach().cpu().numpy()
        confs = pred_boxes.conf.detach().cpu().numpy()
        classes = pred_boxes.cls.detach().cpu().numpy().astype(int)
        predictions = [
            {"xyxy": box.tolist(), "confidence": float(conf), "class_id": int(cls)}
            for box, conf, cls in zip(xyxy, confs, classes)
        ]

    gt_by_class = {}
    for gt in ground_truth:
        gt_by_class.setdefault(gt["class_id"], []).append(gt["xyxy"])

    # Thresholded counts for Precision, Recall, F1, TP, FP and FN.
    thresholded = [p for p in predictions if p["confidence"] >= confidence_threshold]
    matched_gt = {class_id: set() for class_id in gt_by_class}
    tp = fp = 0
    for pred in sorted(thresholded, key=lambda item: item["confidence"], reverse=True):
        class_id = pred["class_id"]
        candidates = gt_by_class.get(class_id, [])
        ious = box_iou_xyxy(pred["xyxy"], candidates)
        order = np.argsort(ious)[::-1]
        match = next(
            (int(idx) for idx in order
             if ious[idx] >= IOU_MATCH_THRESHOLD and int(idx) not in matched_gt.get(class_id, set())),
            None,
        )
        if match is None:
            fp += 1
        else:
            tp += 1
            matched_gt.setdefault(class_id, set()).add(match)

    gt_count = len(ground_truth)
    fn = gt_count - tp
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if gt_count == 0 else 0.0)
    recall = tp / gt_count if gt_count else (1.0 if tp + fp == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    def average_precision(iou_threshold):
        """All-points interpolated AP for this single image and class."""
        aps = []
        class_ids = sorted(set(gt_by_class) | {p["class_id"] for p in predictions})
        for class_id in class_ids:
            class_gt = gt_by_class.get(class_id, [])
            n_gt = len(class_gt)
            class_preds = [
                p for p in predictions
                if p["class_id"] == class_id and p["confidence"] >= AP_CONFIDENCE_FLOOR
            ]
            class_preds.sort(key=lambda item: item["confidence"], reverse=True)
            if n_gt == 0:
                # A class with predictions but no GT is not assigned AP.
                continue
            used = set()
            tp_flags, fp_flags = [], []
            for pred in class_preds:
                ious = box_iou_xyxy(pred["xyxy"], class_gt)
                order = np.argsort(ious)[::-1]
                match = next(
                    (int(idx) for idx in order
                     if ious[idx] >= iou_threshold and int(idx) not in used),
                    None,
                )
                if match is None:
                    tp_flags.append(0)
                    fp_flags.append(1)
                else:
                    used.add(match)
                    tp_flags.append(1)
                    fp_flags.append(0)

            if not tp_flags:
                aps.append(0.0)
                continue
            cum_tp = np.cumsum(tp_flags)
            cum_fp = np.cumsum(fp_flags)
            recalls = cum_tp / n_gt
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([0.0], precisions, [0.0]))
            for idx in range(len(mpre) - 1, 0, -1):
                mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
            changes = np.where(mrec[1:] != mrec[:-1])[0]
            aps.append(float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1])))
        return float(np.mean(aps)) if aps else 0.0

    ap50 = average_precision(0.50)
    ap5095 = float(np.mean([average_precision(t) for t in np.arange(0.50, 0.951, 0.05)]))
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "map50": ap50, "map5095": ap5095,
        "gt": gt_count, "tp": tp, "fn": fn, "fp": fp,
    }

# Styling lives in this file so an outdated/misplaced style.css cannot override it.
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {--ink:#1d2230;--text:#252936;--muted:#77776f;--yellow:#f0df7b;--pale:#fff9dc;--box:#f0f0ed;}
html,body,[class*="stApp"]{font-family:Inter,Arial,sans-serif!important;color:var(--text)}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#f7f6f1!important}
.block-container{max-width:100%!important;padding:.6rem 1.3rem 2rem!important}
header[data-testid="stHeader"]{background:transparent!important}
.app-header{display:flex;align-items:center;gap:15px;min-height:92px;margin:-10px -21px 8px;padding:8px 26px;background:linear-gradient(105deg,#ffffc5 0%,#f7edaa 52%,#f0df7b 100%);border-bottom:1px solid #e5d46e}
.brand-logo{width:90px;height:74px;object-fit:contain}.brand-fallback{font-size:45px}
.brand-title{color:var(--ink);font-size:30px;font-weight:800;letter-spacing:-.7px;line-height:1.1}
.brand-subtitle{margin-top:7px;color:#5f604f;font-size:12px;font-weight:500}
.section-heading{font-size:19px;font-weight:800;color:var(--ink);margin:0 0 15px;letter-spacing:-.35px}
/* Keyed st.container elements are the panels themselves, regardless of the wrapper test-id. */
.st-key-settings_panel,.st-key-workspace_panel,.st-key-performance_panel{background-color:#fff!important;background-image:none!important;border:1px solid #deded8!important;border-radius:12px!important;box-shadow:0 3px 13px rgba(35,35,30,.045)!important}
.st-key-settings_panel>div,.st-key-workspace_panel>div,.st-key-performance_panel>div{background-color:transparent!important;background-image:none!important}
.stRadio label,.stNumberInput label,.stFileUploader label{font-size:12px!important;font-weight:750!important;color:var(--ink)!important}
[data-testid="stRadio"] [role="radiogroup"]{gap:7px}
[data-testid="stRadio"] label:has(input:checked){background:var(--pale);border-radius:8px}
/* Border the number input control itself, not its entire labeled field. */
.st-key-settings_panel [data-testid="stNumberInput"] [data-baseweb="input"],
.st-key-settings_panel [data-testid="stNumberInput"] [data-baseweb="base-input"]{background:var(--pale)!important;border:1px solid #a9a38e!important;border-radius:8px!important;box-shadow:none!important;overflow:hidden}
.st-key-settings_panel [data-testid="stNumberInput"] input,
.st-key-settings_panel [data-testid="stNumberInput"] button{background:var(--pale)!important;color:var(--ink)!important}
.st-key-settings_panel [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within{border-color:#b6a23c!important;box-shadow:0 0 0 2px rgba(240,223,123,.25)!important}
.stButton button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#ffffc5,#f0df7b)!important;color:var(--ink)!important;border:1px solid #d7c45e!important;border-radius:9px!important;font-size:12px!important;font-weight:800!important;min-height:46px}
.stButton button[kind="primary"]:hover{background:linear-gradient(135deg,#fffdb6,#e8d66f)!important;color:var(--ink)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--box)!important;background-image:none!important;border:1px dashed #d8d6cd!important;border-radius:11px!important;min-height:0!important;padding:7px 12px!important}
[data-testid="stFileUploaderDropzone"] button{background:#ffffc5!important;color:var(--ink)!important;border:1px solid #d8c862!important}
.empty-result{min-height:160px;display:flex;align-items:center;justify-content:center;text-align:center;border:1px dashed #dedcd2;border-radius:11px;background:var(--box)!important;color:#85847d;font-size:12px;padding:25px}
.stat-box{background:var(--box);border:1px solid #e9e6d9;border-radius:9px;padding:10px 8px;min-height:70px}
.stat-label{font-size:10px;font-weight:700;color:var(--muted)}.stat-value{margin-top:7px;font-size:12px;font-weight:800;color:var(--ink);overflow-wrap:anywhere}
.camera-intro{min-height:95px;text-align:center;padding:16px;background:var(--box);border:1px dashed #d8d6cd;border-radius:9px}.camera-intro strong{font-size:15px}.camera-intro p{font-size:11px;color:#7c7f87;margin:8px 0}
/* Compact native uploader: keep Streamlit file selection and removal functional. */
.st-key-workspace_panel [data-testid="stFileUploaderDropzone"]{min-height:0!important;padding:7px 12px!important}
.st-key-workspace_panel [data-testid="stFileUploaderDropzoneInstructions"]{display:none!important}
.st-key-workspace_panel [data-testid="stFileUploaderDropzone"] button{min-height:32px!important;padding:4px 12px!important}
.st-key-workspace_panel [data-testid="stFileUploaderFile"]{padding:5px 10px!important;margin-top:4px!important;background:#f0f0ed!important;border-radius:8px!important}
.st-key-workspace_panel [data-testid="stFileUploader"]{margin-bottom:2px!important}
/* Keep the settings panel visually balanced against the right workspace. */
.st-key-settings_panel{min-height:620px!important}
/* Comfortable bottom breathing room for the result box. */
.st-key-workspace_panel{padding-bottom:24px!important;box-sizing:border-box!important}
.st-key-performance_panel{margin-top:14px!important;padding-bottom:10px!important;box-sizing:border-box!important}

/* Model Performance Comparison panel */
.perf-subheading{font-size:12px;font-weight:800;color:var(--ink);margin:2px 0 8px}
.perf-model{font-size:12px;font-weight:700;color:#5f604f;margin-bottom:8px;overflow-wrap:anywhere}
.perf-row{display:flex;justify-content:space-between;align-items:center;background:var(--box);border:1px solid #e9e6d9;border-radius:8px;padding:7px 10px;margin-bottom:5px;font-size:12px;color:var(--ink)}
.perf-row .perf-name{font-weight:650}.perf-row .perf-value{font-weight:800;font-variant-numeric:tabular-nums}
.perf-empty{font-size:11px;color:#85847d;background:var(--box);border:1px dashed #dedcd2;border-radius:8px;padding:12px;text-align:center}

/* Full-width model and variant controls: actual Streamlit buttons, not radio labels. */
.st-key-settings_panel [data-testid="stButton"] button {
  width:100%!important; min-height:46px!important; display:flex!important;
  justify-content:flex-start!important; text-align:left!important;
  padding:10px 14px!important; border:1px solid #dedcd5!important;
  border-radius:9px!important; background:#fff!important; color:#1d2230!important;
  font-size:13px!important; font-weight:650!important;
  transition:background-color .18s ease,border-color .18s ease,box-shadow .18s ease!important;
}
.st-key-settings_panel [data-testid="stButton"] button:hover {
  background:#fffbe8!important; border-color:#d9c75d!important;
  box-shadow:0 2px 7px rgba(0,0,0,.035)!important;
}
.st-key-settings_panel [data-testid="stButton"] button[kind="primary"] {
  background:#fff7b8!important; border-color:#e3cb55!important; color:#1d2230!important;
}
.st-key-settings_panel [data-testid="stButton"] button[kind="primary"]:hover {
  background:#f9ed9d!important;
}
/* Keep the actual Run Detection button visually distinct. */
.st-key-settings_panel .st-key-run_detection button[kind="primary"] {
  justify-content:center!important; background:linear-gradient(135deg,#ffffc5,#f0df7b)!important;
  border-color:#d7c45e!important; font-weight:800!important; min-height:46px!important;
}
/* Three real buttons in equal Streamlit columns, guaranteed to fill the row. */
.st-key-workspace_panel .st-key-source_image button,
.st-key-workspace_panel .st-key-source_video button,
.st-key-workspace_panel .st-key-source_camera button {
  width:100%!important; min-height:49px!important; border-radius:0!important;
  background:#faf9f5!important; border:1px solid #e4dfcf!important;
  color:#555!important; font-size:13px!important; font-weight:650!important;
  transition:background-color .18s ease,border-color .18s ease!important;
}
.st-key-workspace_panel .st-key-source_image button:hover,
.st-key-workspace_panel .st-key-source_video button:hover,
.st-key-workspace_panel .st-key-source_camera button:hover {
  background:#fffbe8!important; border-color:#d9c75d!important;
}
.st-key-workspace_panel .st-key-source_image button[kind="primary"],
.st-key-workspace_panel .st-key-source_video button[kind="primary"],
.st-key-workspace_panel .st-key-source_camera button[kind="primary"] {
  background:#fff4a8!important; border-color:#e3cb55!important; color:#1d2230!important;
}
.st-key-workspace_panel .st-key-source_image button{border-radius:10px 0 0 10px!important}
.st-key-workspace_panel .st-key-source_camera button{border-radius:0 10px 10px 0!important}
/* Remove column gutters only for the three source buttons. */
.st-key-source_selector [data-testid="stHorizontalBlock"] {gap:0!important}
.st-key-source_selector [data-testid="stColumn"] {min-width:0!important}
.st-key-source_selector [data-testid="stButton"] {width:100%!important}
/* Dedicated nested media surfaces: gray backdrop, centered 70%-width media. */
.st-key-input_preview,.st-key-output_preview{background:#f0f0ed!important;background-image:none!important;border:1px solid #dedcd5!important;border-radius:12px!important;padding:16px!important;margin-top:10px!important;margin-bottom:18px!important;box-sizing:border-box!important}
.st-key-input_preview>div,.st-key-output_preview>div{background:transparent!important}
.st-key-input_preview img,.st-key-output_preview img,.st-key-input_preview video,.st-key-output_preview video{border-radius:7px!important}
.st-key-workspace_panel{padding-bottom:24px!important}

/* Preview size is controlled by dedicated columns in Python. */
@media(max-width:850px){.st-key-settings_panel{min-height:0!important}}
@media(max-width:850px){.brand-title{font-size:22px}.brand-subtitle{font-size:10px}.app-header{padding:10px 14px}.brand-logo{width:60px;height:55px}.block-container{padding:.7rem!important}}
"""
st.html(f"<style>{CSS}</style>")


def reset_result() -> None:
    st.session_state.result = None


def header() -> None:
    logo = ROOT / "assets" / "pothole-sign.png"
    if logo.is_file():
        encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
        logo_html = f'<img class="brand-logo" src="data:image/png;base64,{encoded}" alt="Pothole sign">'
    else:
        logo_html = '<span class="brand-fallback">⚠</span>'
    st.markdown(
        f'<div class="app-header">{logo_html}<div>'
        '<div class="brand-title">Pothole Detection System</div>'
        '<div class="brand-subtitle">Detect potholes from images, videos, and live camera input using YOLO26 and RT-DETR.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )



def show_result(result: dict | None) -> None:
    st.markdown('<div class="section-heading">Detection Result</div>', unsafe_allow_html=True)
    if result is None:
        st.markdown('<div class="empty-result">Detection results will appear here after running the model.</div>', unsafe_allow_html=True)
        return
    with st.container(border=False, key="output_preview"):
        preview_left, preview_center, preview_right = st.columns([1.5, 7, 1.5], gap="small")
        with preview_center:
            if result["kind"] == "image":
                st.image(result["content"], use_container_width=True)
            else:
                st.video(result["content"])
    if result["kind"] == "video":
        st.download_button("Download Result Video", result["content"], file_name="pothole_detection_result.mp4", mime="video/mp4", use_container_width=True)

    stats = result["stats"]
    if result["kind"] == "image" and stats.get("evaluation"):
        evaluation = stats["evaluation"]
        values = [
            ("GT Potholes", evaluation["gt"]),
            ("Detected (TP)", evaluation["tp"]),
            ("Missed (FN)", evaluation["fn"]),
            ("False Detected (FP)", evaluation["fp"]),
            ("Model", result["model"]),
            ("Confidence", f'{result["confidence"]:.2f}'),
            ("Inference Time", f'{stats["inference_time_ms"]:.1f} ms' if stats.get("inference_time_ms") is not None else "-"),
            ("Device", stats["device"]),
        ]
    else:
        values = [
            ("Potholes Detected", stats["detections"]),
            ("Model", result["model"]),
            ("Confidence", f'{result["confidence"]:.2f}'),
            ("Inference Time", f'{stats["inference_time_ms"]:.1f} ms' if stats.get("inference_time_ms") is not None else "-"),
            ("Device", stats["device"]),
        ]
    for column, (label, value) in zip(st.columns(len(values), gap="small"), values):
        with column:
            st.markdown(
                f'<div class="stat-box"><div class="stat-label">{html.escape(label)}</div>'
                f'<div class="stat-value">{html.escape(str(value))}</div></div>',
                unsafe_allow_html=True,
            )



def history_chart(history: list[dict]) -> alt.Chart:
    """Grouped bar chart of per-image metrics from runs in this session."""
    if not history:
        frame = pd.DataFrame({"Metric": METRIC_NAMES, "Score": [0.0] * len(METRIC_NAMES), "Run": ["No runs yet"] * len(METRIC_NAMES)})
    else:
        rows = [
            {"Metric": metric, "Score": float(entry["metrics"][metric]), "Run": entry["label"]}
            for entry in history for metric in METRIC_NAMES
        ]
        frame = pd.DataFrame(rows)
    run_order = list(dict.fromkeys(frame["Run"].tolist()))
    base = alt.Chart(frame).encode(
        y=alt.Y("Metric:N", sort=METRIC_NAMES, title=None),
        yOffset=alt.YOffset("Run:N", sort=run_order),
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score", axis=alt.Axis(format=".1f")),
    )
    bars = base.mark_bar().encode(
        color=alt.Color("Run:N", sort=run_order, legend=alt.Legend(orient="bottom", title="Run", columns=1, labelLimit=260)),
        tooltip=["Run:N", "Metric:N", alt.Tooltip("Score:Q", format=".4f")],
    )
    labels = base.mark_text(align="left", dx=3, fontSize=10).encode(text=alt.Text("Score:Q", format=".2f"))
    return (bars + labels).properties(height=max(230, len(METRIC_NAMES) * (len(run_order) * 17 + 14)))


def performance_panel(family: str, variant: str, scenario: str, result: dict | None) -> None:
    with st.container(border=True, key="performance_panel"):
        st.markdown('<div class="section-heading">Image Performance Metrics</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="perf-model">{html.escape(family)} · {html.escape(variant)}</div>',
            unsafe_allow_html=True,
        )
        evaluation = None
        if result and result.get("kind") == "image":
            evaluation = result.get("stats", {}).get("evaluation")

        # Tanpa label: tidak ada metrik dan tidak ada bar chart.
        if evaluation is None:
            if result and result.get("kind") == "image":
                st.caption("Label tidak ditemukan untuk gambar ini, sehingga metrik tidak ditampilkan.")
            else:
                st.caption("Run detection on an image to fill in these metrics.")
            return

        metric_keys = [
            ("Precision", "precision"), ("Recall", "recall"), ("F1-Score", "f1"),
            ("mAP50", "map50"), ("mAP50-95", "map5095"),
        ]
        for name, key in metric_keys:
            st.markdown(
                f'<div class="perf-row"><span class="perf-name">{name}</span>'
                f'<span class="perf-value">{evaluation[key]:.4f}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="perf-subheading" style="margin-top:16px">Per-Image Run History</div>', unsafe_allow_html=True)
        st.altair_chart(history_chart(st.session_state.history), use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []
if "scenario" not in st.session_state:
    st.session_state.scenario = ""
if "model_family" not in st.session_state:
    st.session_state.model_family = "YOLO26"
if "model_variant" not in st.session_state:
    st.session_state.model_variant = "Nano (n)"
if "source" not in st.session_state:
    st.session_state.source = "Image"

def select_model(family: str, variant: str) -> None:
    if (st.session_state.model_family, st.session_state.model_variant) != (family, variant):
        st.session_state.model_family = family
        st.session_state.model_variant = variant
        reset_result()


def select_source(source_name: str) -> None:
    if st.session_state.source != source_name:
        st.session_state.source = source_name
        reset_result()


header()
left, right = st.columns([250, 920], gap="small")

with left:
    with st.container(border=True, key="settings_panel"):
        st.markdown('<div class="section-heading">☷ &nbsp; Model Settings</div>', unsafe_allow_html=True)

        st.markdown('**Model Architecture**')
        model_options = [
            ("YOLO26", "Nano (n)", "YOLO26n"),
            ("YOLO26", "Small (s)", "YOLO26s"),
            ("YOLO26", "Medium (m)", "YOLO26m"),
            ("YOLO26", "Large (l)", "YOLO26l"),
            ("RT-DETR", "Large (L)", "RT-DETR-L"),
        ]
        for index, (family_option, variant_option, display_name) in enumerate(model_options):
            selected = (
                st.session_state.model_family == family_option
                and st.session_state.model_variant == variant_option
            )
            st.button(
                display_name,
                key=f"pick_architecture_{index}",
                use_container_width=True,
                type="primary" if selected else "secondary",
                on_click=select_model, args=(family_option, variant_option),
            )
        scenario = st.session_state.scenario
        model_family = st.session_state.model_family
        variant = st.session_state.model_variant
        confidence = st.number_input("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f", key="confidence", on_change=reset_result)
        st.caption("Value between 0.00 and 1.00")
        run_placeholder = st.empty()

with right:
    with st.container(border=True, key="workspace_panel"):
        st.markdown('<div class="section-heading">⇧ &nbsp; Input Source</div>', unsafe_allow_html=True)
        with st.container(key="source_selector"):
            source_columns = st.columns(3, gap=None)
            for col, item, symbol in zip(source_columns, ("Image", "Video", "Camera"), ("", "", "")):
                with col:
                    st.button(
                        f"{symbol}  {item}", key=f"source_{item.lower()}",
                        use_container_width=True,
                        type="primary" if st.session_state.source == item else "secondary",
                        on_click=select_source, args=(item,),
                    )
        source = st.session_state.source
        upload = None
        if source == "Image":
            upload = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], help="Drag and drop an image here, or choose a file from your computer.", key="image_upload", on_change=reset_result)
            if upload is not None:
                with st.container(border=False, key="input_preview"):
                    preview_left, preview_center, preview_right = st.columns([1.5, 7, 1.5], gap="small")
                    with preview_center:
                        st.image(upload, use_container_width=True)
        elif source == "Video":
            upload = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv", "webm"], help="MP4, MOV, AVI, MKV, WEBM", key="video_upload", on_change=reset_result)
            if upload is not None:
                with st.container(border=False, key="input_preview"):
                    preview_left, preview_center, preview_right = st.columns([1.5, 7, 1.5], gap="small")
                    with preview_center:
                        st.video(upload)
        else:
            st.markdown('<div class="camera-intro"><strong>Browser Camera</strong><p>Open your browser camera for real-time pothole detection.</p></div>', unsafe_allow_html=True)
            st.caption("Camera permission will be requested by your browser")
            st.caption("Press START to open the camera; STOP to close it. Live inference starts with the camera.")
            try:
                live_model = load_model(model_family, variant, st.session_state.scenario)
            except Exception as exc:
                st.warning(str(exc))
                live_model = None
            if live_model is not None:
                def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
                    image = frame.to_ndarray(format="bgr24")
                    annotated, _, _ = predict(image, live_model, float(confidence), imgsz=640)
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

                webrtc_streamer(
                    key=f"camera_{st.session_state.scenario}_{model_family}_{variant}_{confidence}",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                )
                st.caption("Camera: bounding boxes are shown on the live video. CPU hosting may have low FPS.")
        if source != "Camera":
            show_result(st.session_state.result)

with left:
    # New panel under Model Settings.
    performance_panel(model_family, variant, st.session_state.scenario, st.session_state.result)

    if source == "Camera":
        run_placeholder.info("For Camera, use START / STOP in the camera panel.")
    else:
        run = run_placeholder.button("▶  Run Detection", key="run_detection", use_container_width=True, type="primary")
        if run:
            if upload is None:
                st.warning(f"Please select an {source.lower()} first.")
            else:
                try:
                    model_label, _ = MODELS[(model_family, variant)]
                    model_label = f"{st.session_state.scenario.replace('_', ' ').title()} · {model_label}"
                    with st.spinner(f"Loading {model_label}..."):
                        model = load_model(model_family, variant, st.session_state.scenario)
                    if source == "Image":
                        with st.spinner("Detecting this image..."):
                            frame = decode_image(upload.getvalue())
                            annotated, count, ms = predict(frame, model, float(confidence))
                            height, width = frame.shape[:2]
                            ground_truth = load_yolo_labels(upload.name, width, height)

                            # Metrik hanya dihitung jika file label pasangan ditemukan.
                            evaluation = None
                            if ground_truth is not None:
                                # Re-run at a low confidence floor so AP can rank detections
                                # across confidence levels; the displayed annotation remains
                                # based on the user-selected confidence threshold.
                                ap_results = model.predict(
                                    source=frame, conf=AP_CONFIDENCE_FLOOR, iou=0.50,
                                    device=0 if __import__("torch").cuda.is_available() else "cpu",
                                    verbose=False,
                                )
                                evaluation = evaluate_single_image(ap_results[0], ground_truth, float(confidence))

                            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            stats = {
                                "detections": count, "inference_time_ms": ms,
                                "device": device_label(), "evaluation": evaluation,
                            }
                            st.session_state.result = {
                                "kind": "image", "content": rgb, "model": model_label,
                                "confidence": float(confidence), "stats": stats,
                                "image_name": upload.name,
                            }
                            if evaluation is not None:
                                run_number = len(st.session_state.history) + 1
                                st.session_state.history.append({
                                    "label": f"#{run_number} {model_label} · {Path(upload.name).stem}",
                                    "metrics": {
                                        "Precision": evaluation["precision"],
                                        "Recall": evaluation["recall"],
                                        "F1-Score": evaluation["f1"],
                                        "mAP50": evaluation["map50"],
                                        "mAP50-95": evaluation["map5095"],
                                    },
                                })
                                # Keep the chart readable during longer sessions.
                                st.session_state.history = st.session_state.history[-10:]
                    else:
                        progress = st.progress(0.0, text="Processing video...")
                        with st.spinner("Processing video frames and encoding MP4..."):
                            data, stats = process_video(
                                upload.getvalue(), Path(upload.name).suffix, model,
                                float(confidence), progress_callback=lambda value: progress.progress(value),
                            )
                            st.session_state.result = {
                                "kind": "video", "content": data, "model": model_label,
                                "confidence": float(confidence), "stats": stats,
                            }
                        progress.empty()

                    st.rerun()
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")
