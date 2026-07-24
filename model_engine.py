"""
model_engine.py

This module is NOT new business logic. It is the exact inference and
business-logic code from `PPE_Detection_BusinessLogic.ipynb`
(predict_filtered, compute_iou, match_attribute, analyze_detections,
draw_annotations, process_video, and the image-processing logic from
cell 30 / reportpt logic from cell 36), copied verbatim so it can be
imported from app.py. Jupyter notebooks cannot be imported as Python
modules directly, so this file exists purely as a loader for that code.

Do not "improve" the detection thresholds or matching logic here without
also updating the source notebook — they must stay in sync.
"""

import os
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"Model\best.pt"

# ---------------------------------------------------------------------------
# 2. Load the model
# ---------------------------------------------------------------------------

_model = None


def get_model():
    """Cached YOLO model loader (best.pt)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"'{MODEL_PATH}' was not found in the project directory. "
                "Place best.pt next to app.py before running."
            )
        _model = YOLO(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# 2.1 Detection configuration  (verbatim from notebook cell 21)
# ---------------------------------------------------------------------------

IMGSZ = 1280
GLOBAL_CONF = 0.10
USE_AUGMENT = True

CLASS_CONF_THRESH = {
    "Person": 0.20,
    "Hardhat": 0.35,
    "Safety Vest": 0.35,
    "NO-Hardhat": 0.30,
    "NO-Safety Vest": 0.30,
}


def predict_filtered(frame):
    model = get_model()
    results = model.predict(
        frame, conf=GLOBAL_CONF, imgsz=IMGSZ, augment=USE_AUGMENT, verbose=False
    )
    r = results[0]
    if len(r.boxes) == 0:
        return r

    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    keep_mask = np.array(
        [
            confs[i] >= CLASS_CONF_THRESH.get(model.names[cls_ids[i]], GLOBAL_CONF)
            for i in range(len(cls_ids))
        ]
    )
    return r[keep_mask]


# ---------------------------------------------------------------------------
# 3. Business logic  (verbatim from notebook cell 25)
# ---------------------------------------------------------------------------


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxA_area + boxB_area - inter_area
    if union == 0:
        return 0
    return inter_area / union


def match_attribute(person_box, attr_boxes, min_iou=0.02):
    best_iou = 0
    best_idx = None
    for i, abox in enumerate(attr_boxes):
        iou = compute_iou(person_box, abox)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    if best_idx is not None and best_iou >= min_iou:
        return best_idx
    return None


def analyze_detections(result):
    model = get_model()
    CLASS_NAMES = model.names
    NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.empty((0, 4))
    cls_ids = (
        boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.empty((0,), dtype=int)
    )

    person_boxes = xyxy[cls_ids == NAME_TO_ID["Person"]]
    hardhat_boxes = xyxy[cls_ids == NAME_TO_ID["Hardhat"]]
    no_hardhat_boxes = xyxy[cls_ids == NAME_TO_ID["NO-Hardhat"]]
    vest_boxes = xyxy[cls_ids == NAME_TO_ID["Safety Vest"]]
    no_vest_boxes = xyxy[cls_ids == NAME_TO_ID["NO-Safety Vest"]]

    people = []
    for pbox in person_boxes:
        hh_idx = match_attribute(pbox, hardhat_boxes)
        nhh_idx = match_attribute(pbox, no_hardhat_boxes)
        if hh_idx is not None and nhh_idx is None:
            hardhat_status = "Hardhat"
        elif nhh_idx is not None and hh_idx is None:
            hardhat_status = "NO-Hardhat"
        elif hh_idx is not None and nhh_idx is not None:
            hardhat_status = (
                "Hardhat"
                if compute_iou(pbox, hardhat_boxes[hh_idx])
                >= compute_iou(pbox, no_hardhat_boxes[nhh_idx])
                else "NO-Hardhat"
            )
        else:
            hardhat_status = "Unknown"

        v_idx = match_attribute(pbox, vest_boxes)
        nv_idx = match_attribute(pbox, no_vest_boxes)
        if v_idx is not None and nv_idx is None:
            vest_status = "Safety Vest"
        elif nv_idx is not None and v_idx is None:
            vest_status = "NO-Safety Vest"
        elif v_idx is not None and nv_idx is not None:
            vest_status = (
                "Safety Vest"
                if compute_iou(pbox, vest_boxes[v_idx])
                >= compute_iou(pbox, no_vest_boxes[nv_idx])
                else "NO-Safety Vest"
            )
        else:
            vest_status = "Unknown"

        compliant = (hardhat_status == "Hardhat") and (vest_status == "Safety Vest")
        violation = (hardhat_status == "NO-Hardhat") or (
            vest_status == "NO-Safety Vest"
        )

        people.append(
            {
                "box": pbox,
                "hardhat_status": hardhat_status,
                "vest_status": vest_status,
                "compliant": compliant,
                "violation": violation,
            }
        )

    total_workers = len(people)
    violations = sum(1 for p in people if p["violation"])
    compliant_count = sum(1 for p in people if p["compliant"])
    helmet_violations = sum(1 for p in people if p["hardhat_status"] == "NO-Hardhat")
    vest_violations = sum(1 for p in people if p["vest_status"] == "NO-Safety Vest")
    unknown_count = sum(
        1
        for p in people
        if p["hardhat_status"] == "Unknown" or p["vest_status"] == "Unknown"
    )
    compliance_rate = (
        (compliant_count / total_workers * 100) if total_workers > 0 else 0.0
    )

    stats = {
        "total_workers": total_workers,
        "compliant_workers": compliant_count,
        "violations": violations,
        "helmet_violations": helmet_violations,
        "vest_violations": vest_violations,
        "unknown_count": unknown_count,
        "compliance_rate": round(compliance_rate, 1),
    }

    return people, stats


# ---------------------------------------------------------------------------
# 4. Drawing annotations  (verbatim from notebook cell 27)
# ---------------------------------------------------------------------------

COLOR_OK = (0, 200, 0)
COLOR_BAD = (0, 0, 255)
COLOR_UNKNOWN = (0, 165, 255)


def draw_annotations(frame, people, stats):
    frame = frame.copy()
    for p in people:
        x1, y1, x2, y2 = map(int, p["box"])
        if p["violation"]:
            color = COLOR_BAD
        elif p["compliant"]:
            color = COLOR_OK
        else:
            color = COLOR_UNKNOWN

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{p["hardhat_status"]} | {p["vest_status"]}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    overlay_text = (
        f'Workers: {stats["total_workers"]}  '
        f'Violations: {stats["violations"]}  '
        f'Compliance: {stats["compliance_rate"]}%'
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(
        frame,
        overlay_text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


# ---------------------------------------------------------------------------
# 5. Process an image  (same pipeline as notebook cell 30, wrapped as a
#    reusable function — the notebook ran this inline rather than as a
#    named function, so it is wrapped here without changing what it does)
# ---------------------------------------------------------------------------


def process_image(frame, output_path="outputs/annotated_image.jpg"):
    result = predict_filtered(frame)
    people, stats = analyze_detections(result)
    annotated = draw_annotations(frame, people, stats)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)

    return annotated, people, stats, output_path


# ---------------------------------------------------------------------------
# 6. Process a video  (verbatim from notebook cell 33)
# ---------------------------------------------------------------------------


def process_video(
    video_path,
    output_path="outputs/annotated_video.mp4",
    frame_skip=1,
    progress_callback=None,
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    all_stats = []
    last_people, last_stats = [], {
        "total_workers": 0,
        "compliant_workers": 0,
        "violations": 0,
        "helmet_violations": 0,
        "vest_violations": 0,
        "unknown_count": 0,
        "compliance_rate": 0.0,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            r = predict_filtered(frame)
            people, stats = analyze_detections(r)
            last_people, last_stats = people, stats
            all_stats.append(stats)
        else:
            people, stats = last_people, last_stats

        annotated = draw_annotations(frame, people, stats)
        writer.write(annotated)

        frame_idx += 1
        if progress_callback is not None and total_frames:
            progress_callback(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()

    if all_stats:
        avg_workers = float(np.mean([s["total_workers"] for s in all_stats]))
        max_workers = max(s["total_workers"] for s in all_stats)
        total_violation_frames = sum(1 for s in all_stats if s["violations"] > 0)
        frames_with_people = [s for s in all_stats if s["total_workers"] > 0]
        avg_compliance = (
            float(np.mean([s["compliance_rate"] for s in frames_with_people]))
            if frames_with_people
            else 0.0
        )
        total_helmet_violations = sum(s["helmet_violations"] for s in all_stats)
        total_vest_violations = sum(s["vest_violations"] for s in all_stats)
        total_violations = sum(s["violations"] for s in all_stats)
        total_compliant = sum(s["compliant_workers"] for s in all_stats)

        summary = {
            "processed_frames": len(all_stats),
            "avg_workers_per_frame": round(avg_workers, 1),
            "max_workers_in_frame": max_workers,
            "frames_with_violations": total_violation_frames,
            "avg_compliance_rate": round(avg_compliance, 1),
            # aggregate, dashboard-friendly fields (same underlying numbers,
            # summed rather than averaged, so the KPI cards on the
            # dashboard page have a consistent shape with the image path)
            "total_workers": max_workers,
            "compliant_workers": total_compliant,
            "violations": total_violations,
            "helmet_violations": total_helmet_violations,
            "vest_violations": total_vest_violations,
            "compliance_rate": round(avg_compliance, 1),
        }
    else:
        summary = {
            "processed_frames": 0,
            "avg_workers_per_frame": 0,
            "max_workers_in_frame": 0,
            "frames_with_violations": 0,
            "avg_compliance_rate": 0.0,
            "total_workers": 0,
            "compliant_workers": 0,
            "violations": 0,
            "helmet_violations": 0,
            "vest_violations": 0,
            "compliance_rate": 0.0,
        }

    return output_path, summary


# ---------------------------------------------------------------------------
# 8. Generate report  (verbatim logic from notebook cell 36, generalized
#    to accept either an image-stats dict or a video-summary dict)
# ---------------------------------------------------------------------------


def generate_report(
    stats, mode, model_path=MODEL_PATH, output_path="outputs/report.txt"
):
    lines = []
    lines.append("PPE DETECTION REPORT")
    lines.append("=" * 40)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {model_path}")
    lines.append("")

    if mode == "image":
        lines.append("IMAGE RESULTS")
        lines.append("-" * 40)
        lines.append(f"Total workers detected: {stats['total_workers']}")
        lines.append(f"Compliant workers: {stats['compliant_workers']}")
        lines.append(f"Violations: {stats['violations']}")
        lines.append(f"Helmet violations: {stats.get('helmet_violations', 0)}")
        lines.append(f"Vest violations: {stats.get('vest_violations', 0)}")
        lines.append(f"Compliance rate: {stats['compliance_rate']}%")
        lines.append("")
    else:
        lines.append("VIDEO RESULTS")
        lines.append("-" * 40)
        lines.append(f"Frames processed: {stats['processed_frames']}")
        lines.append(f"Average workers per frame: {stats['avg_workers_per_frame']}")
        lines.append(
            f"Maximum workers in a single frame: {stats['max_workers_in_frame']}"
        )
        lines.append(
            f"Frames containing at least one violation: {stats['frames_with_violations']}"
        )
        lines.append(f"Average compliance rate: {stats['avg_compliance_rate']}%")
        lines.append("")

    lines.append("CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Image size: {IMGSZ}")
    lines.append(f"Augmentation enabled: {USE_AUGMENT}")
    for cls_name, thresh in CLASS_CONF_THRESH.items():
        lines.append(f"Confidence threshold ({cls_name}): {thresh}")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)

    return output_path, report_text
