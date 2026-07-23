"""
app.py — AI-Powered PPE Detection & Construction Site Safety Monitoring System

Frontend only. All inference and business logic lives in model_engine.py,
which is a verbatim, importable copy of PPE_Detection_BusinessLogic.ipynb.
This file does not reimplement, alter, or mock any detection logic.

Run with:  streamlit run app.py
Requires, in the same folder: best.pt, model_engine.py
Optional, used opportunistically if present: outputs/results.png,
outputs/confusion_matrix.png, filtered_dataset/{train,valid,test}/labels
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model_engine as engine

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SiteGuard AI | Construction Safety Platform",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["Person", "Hardhat", "Safety Vest", "NO-Hardhat", "NO-Safety Vest"]

METRICS = {
    "Precision": 79.6,
    "Recall": 63.1,
    "mAP@50": 70.2,
    "mAP@50-95": 44.3,
}

# =============================================================================
# THEME / CSS
# =============================================================================

PRIMARY = "#F5A623"  # safety orange/yellow
PRIMARY_DARK = "#D4881A"
DARK = "#1C1F26"
DARK_2 = "#2A2E37"
LIGHT = "#F7F7F5"
GREEN = "#1DB954"
RED = "#E5484D"
YELLOW = "#F5C518"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Barlow+Condensed:wght@600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {DARK} 0%, {DARK_2} 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: {LIGHT} !important;
        }}
        section[data-testid="stSidebar"] .stRadio > div {{
            gap: 0.4rem;
        }}
        section[data-testid="stSidebar"] label {{
            background: rgba(255,255,255,0.04);
            padding: 10px 14px;
            border-radius: 10px;
            width: 100%;
            transition: all 0.15s ease;
        }}
        section[data-testid="stSidebar"] label:hover {{
            background: rgba(245,166,35,0.15);
        }}

        .hero {{
            background: linear-gradient(120deg, {DARK} 0%, #33261A 55%, {PRIMARY_DARK} 130%);
            border-radius: 20px;
            padding: 56px 48px;
            color: white;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(245,166,35,0.25);
        }}
        .hero::before {{
            content: "";
            position: absolute;
            top: -60px; right: -60px;
            width: 260px; height: 260px;
            background: radial-gradient(circle, rgba(245,166,35,0.35), transparent 70%);
        }}
        .hero-tag {{
            display: inline-block;
            background: rgba(245,166,35,0.18);
            border: 1px solid {PRIMARY};
            color: {PRIMARY};
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 18px;
        }}
        .hero h1 {{
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 3.1rem;
            font-weight: 700;
            line-height: 1.08;
            margin: 0 0 14px 0;
        }}
        .hero p {{
            font-size: 1.08rem;
            color: #D8D8D8;
            max-width: 640px;
            line-height: 1.55;
        }}

        .card {{
            background: white;
            border: 1px solid #ECECEC;
            border-radius: 16px;
            padding: 24px 26px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.035);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            height: 100%;
        }}
        .card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 24px rgba(0,0,0,0.08);
        }}
        .card h4 {{
            margin: 0 0 8px 0;
            font-size: 1.02rem;
            color: {DARK};
        }}
        .card p {{
            margin: 0;
            color: #6B7280;
            font-size: 0.92rem;
            line-height: 1.5;
        }}

        .section-title {{
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 1.9rem;
            font-weight: 700;
            color: {DARK};
            margin: 34px 0 4px 0;
            border-left: 5px solid {PRIMARY};
            padding-left: 12px;
        }}
        .section-sub {{
            color: #8A8F98;
            margin: 0 0 18px 14px;
            font-size: 0.95rem;
        }}

        .kpi {{
            background: {DARK};
            border-radius: 16px;
            padding: 20px 22px;
            color: white;
            border-top: 4px solid {PRIMARY};
            height: 100%;
        }}
        .kpi .kpi-label {{
            font-size: 0.78rem;
            color: #A9AFBC;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .kpi .kpi-value {{
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
        }}

        .badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1rem;
        }}
        .badge-low {{ background: rgba(29,185,84,0.15); color: {GREEN}; border: 1px solid {GREEN}; }}
        .badge-med {{ background: rgba(245,197,24,0.15); color: #B8860B; border: 1px solid {YELLOW}; }}
        .badge-high {{ background: rgba(229,72,77,0.15); color: {RED}; border: 1px solid {RED}; }}

        .alert-card {{
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
            font-size: 0.93rem;
            border-left: 4px solid;
        }}
        .alert-danger {{ background: #FDECEC; border-color: {RED}; color: #7A1F22; }}
        .alert-warn {{ background: #FFF6E0; border-color: {YELLOW}; color: #6B5200; }}
        .alert-ok {{ background: #E9F9EE; border-color: {GREEN}; color: #145C2C; }}

        .footer-bar {{
            margin-top: 50px;
            padding: 26px 10px 10px 10px;
            border-top: 1px solid #EAEAEA;
            color: #9CA3AF;
            font-size: 0.85rem;
            text-align: center;
        }}

        div.stButton > button {{
            background: {PRIMARY};
            color: {DARK};
            border: none;
            border-radius: 10px;
            padding: 10px 22px;
            font-weight: 700;
            transition: all 0.15s ease;
        }}
        div.stButton > button:hover {{
            background: {PRIMARY_DARK};
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "detection_done" not in st.session_state:
    st.session_state.detection_done = False
if "stats" not in st.session_state:
    st.session_state.stats = None
if "people" not in st.session_state:
    st.session_state.people = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "annotated_image_path" not in st.session_state:
    st.session_state.annotated_image_path = None
if "annotated_video_path" not in st.session_state:
    st.session_state.annotated_video_path = None

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 6px 0 18px 0;">
            <div style="font-size:2.1rem;">🦺</div>
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.4rem; font-weight:700; color:white;">
                SiteGuard AI
            </div>
            <div style="font-size:0.75rem; color:#9AA0AC; letter-spacing:0.05em; text-transform:uppercase;">
                Construction Safety Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["🏗️ Project Overview", "🤖 AI Analytics", "📊 Safety Monitoring Dashboard"],
        label_visibility="collapsed",
    )
    st.markdown(
        "<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True
    )
    model_status = (
        "🟢 Model ready"
        if os.path.exists(engine.MODEL_PATH)
        else "🔴 best.pt not found"
    )
    st.markdown(
        f"<div style='font-size:0.82rem; color:#9AA0AC;'>{model_status}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.75rem; color:#6B7280; margin-top:18px;'>YOLO11 · 5-class PPE detector</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE 1 — PROJECT OVERVIEW
# =============================================================================

if page == "🏗️ Project Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">AI Computer Vision · Construction Safety</div>
            <h1>Real-Time PPE Compliance<br>Monitoring for Construction Sites</h1>
            <p>SiteGuard AI uses a custom-trained YOLO11 model to detect workers and their
            protective equipment in images and video, flag violations instantly, and turn
            raw detections into safety analytics your team can act on.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">About the Project</div>', unsafe_allow_html=True
    )
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.write(
            "Construction sites are among the highest-risk workplaces for injury, and most "
            "of that risk traces back to missing or misused personal protective equipment — "
            "hardhats and hi-vis vests in particular. SiteGuard AI applies a custom-trained "
            "YOLO11 object detector to camera footage or uploaded media, identifies every "
            "worker in frame, and matches each one to their hardhat and vest status using "
            "IoU-based spatial matching between person and equipment boxes."
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <h4>🎯 Detected Classes</h4>
                <p>Person · Hardhat · Safety Vest · NO-Hardhat · NO-Safety Vest</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Business Problem</div>', unsafe_allow_html=True
    )
    cols = st.columns(3)
    problems = [
        (
            "⚠️",
            "Manual Monitoring Doesn't Scale",
            "Site supervisors can't watch every worker on every camera feed at once.",
        ),
        (
            "📉",
            "Violations Go Unnoticed",
            "PPE non-compliance is often only caught after an incident, not before.",
        ),
        (
            "🕒",
            "Slow Reporting",
            "Paper-based safety walks produce data too late to prevent the next injury.",
        ),
    ]
    for col, (icon, title, desc) in zip(cols, problems):
        with col:
            st.markdown(
                f'<div class="card"><h4>{icon} {title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Objectives</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    objectives = [
        (
            "🔍",
            "Detect",
            "Identify every person and PPE item in an image or video frame.",
        ),
        (
            "🧠",
            "Classify",
            "Match each worker to hardhat and vest status via IoU matching.",
        ),
        (
            "📊",
            "Quantify",
            "Turn detections into compliance rate, violation counts, and risk level.",
        ),
        (
            "📄",
            "Report",
            "Generate downloadable reports for safety records and audits.",
        ),
    ]
    for col, (icon, title, desc) in zip(cols, objectives):
        with col:
            st.markdown(
                f'<div class="card"><h4>{icon} {title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Business Value & KPIs</div>', unsafe_allow_html=True
    )
    cols = st.columns(4)
    kpis = [
        ("Faster Detection", "Seconds vs. manual walkthroughs"),
        ("Consistent Coverage", "Every uploaded frame checked equally"),
        ("Audit-Ready Reports", "JSON / CSV / TXT exports on demand"),
        ("Actionable Alerts", "Auto-generated safety recommendations"),
    ]
    for col, (label, desc) in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div style="font-size:0.85rem; color:#C9CDD6; margin-top:6px;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)
    steps = [
        "Upload Image/Video",
        "YOLO11 Inference",
        "IoU Attribute Matching",
        "Compliance Scoring",
        "Dashboard + Report",
    ]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="width:44px;height:44px;border-radius:50%;background:{PRIMARY};
                        color:{DARK}; font-weight:700; display:flex; align-items:center;
                        justify-content:center; margin:0 auto 8px auto;">{i+1}</div>
                    <div style="font-size:0.85rem; color:#4B5563;">{step}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Key Features</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    features = [
        (
            "🖼️",
            "Image & Video Support",
            "Run the same detection pipeline on a single photo or a full video.",
        ),
        (
            "📈",
            "Interactive Analytics",
            "Plotly-powered charts for compliance, violations, and PPE distribution.",
        ),
        (
            "🚦",
            "Automated Risk Scoring",
            "Compliance rate is translated into a Low / Medium / High risk badge.",
        ),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f'<div class="card"><h4>{icon} {title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Project Pipeline</div>', unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="card">
            <p><b>Data</b> → Roboflow construction-safety dataset, remapped to a focused 5-class
            schema (Person, Hardhat, Safety Vest, NO-Hardhat, NO-Safety Vest).<br><br>
            <b>Model</b> → YOLO11, trained on the filtered dataset, exported as <code>best.pt</code>.<br><br>
            <b>Business Logic</b> → Per-class confidence thresholds, IoU-based person↔PPE
            matching, compliance/violation classification.<br><br>
            <b>Application</b> → This Streamlit interface, calling the notebook's detection and
            analysis functions directly — no logic is duplicated or reimplemented here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE 2 — AI ANALYTICS
# =============================================================================

elif page == "🤖 AI Analytics":

    st.markdown(
        """
        <div class="hero" style="padding:36px 44px;">
            <div class="hero-tag">Model Intelligence</div>
            <h1 style="font-size:2.3rem;">AI Analytics</h1>
            <p>Architecture, training configuration, and evaluation results for the YOLO11
            PPE detector powering this platform.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Evaluation Metrics</div>', unsafe_allow_html=True
    )
    cols = st.columns(4)
    for col, (label, val) in zip(cols, METRICS.items()):
        with col:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{val}%</div></div>',
                unsafe_allow_html=True,
            )

    fig_bar = go.Figure(
        go.Bar(
            x=list(METRICS.keys()),
            y=list(METRICS.values()),
            marker_color=[PRIMARY, PRIMARY_DARK, GREEN, DARK],
            text=[f"{v}%" for v in METRICS.values()],
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        title="Model Evaluation Metrics",
        yaxis_title="Score (%)",
        template="plotly_white",
        height=380,
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=list(METRICS.values()) + [list(METRICS.values())[0]],
            theta=list(METRICS.keys()) + [list(METRICS.keys())[0]],
            fill="toself",
            line_color=PRIMARY,
        )
    )
    fig_radar.update_layout(
        title="Metric Profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown(
        '<div class="section-title">Dataset & Classes</div>', unsafe_allow_html=True
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"""
            <div class="card">
                <h4>📦 Dataset</h4>
                <p>Roboflow "Construction Site Safety" dataset, remapped from its original
                label set to 5 focused classes and re-exported to <code>filtered_dataset/</code>
                with a matching <code>data.yaml</code>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <h4>🧬 Architecture</h4>
                <p>YOLO11 (Ultralytics), single-stage anchor-free detector, fine-tuned on the
                filtered 5-class dataset and exported as <code>best.pt</code>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    class_df = pd.DataFrame({"Class ID": range(5), "Class Name": CLASS_NAMES})
    st.dataframe(class_df, use_container_width=True, hide_index=True)

    # ---- Live dataset distribution, computed from actual label files if present ----
    st.markdown(
        '<div class="section-title">Dataset Class Distribution</div>',
        unsafe_allow_html=True,
    )
    label_dirs = [
        Path("filtered_dataset/train/labels"),
        Path("filtered_dataset/valid/labels"),
        Path("filtered_dataset/test/labels"),
    ]
    counts = {name: 0 for name in CLASS_NAMES}
    found_any = False
    for d in label_dirs:
        if d.exists():
            found_any = True
            for txt_file in d.glob("*.txt"):
                for line in txt_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    cls_id = int(line.split()[0])
                    if 0 <= cls_id < len(CLASS_NAMES):
                        counts[CLASS_NAMES[cls_id]] += 1

    if found_any and sum(counts.values()) > 0:
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = go.Figure(
                go.Pie(
                    labels=list(counts.keys()), values=list(counts.values()), hole=0.0
                )
            )
            fig_pie.update_layout(
                title="Class Distribution", template="plotly_white", height=380
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_donut = go.Figure(
                go.Pie(
                    labels=list(counts.keys()), values=list(counts.values()), hole=0.55
                )
            )
            fig_donut.update_layout(
                title="Class Distribution (Donut)", template="plotly_white", height=380
            )
            st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info(
            "Class distribution charts will populate automatically once `filtered_dataset/` "
            "is present in the project directory (train/valid/test label folders)."
        )

    st.markdown(
        '<div class="section-title">Training Artifacts</div>', unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    artifact_paths = {
        "Results": "outputs/results.png",
        "Confusion Matrix": "outputs/confusion_matrix.png",
        "Sample Detection": "outputs/annotated_image.jpg",
    }
    for col, (label, path) in zip([c1, c2, c3], artifact_paths.items()):
        with col:
            st.markdown(f"**{label}**")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.markdown(
                    f'<div class="card" style="text-align:center; color:#9CA3AF;">'
                    f"Not found at <code>{path}</code></div>",
                    unsafe_allow_html=True,
                )

# =============================================================================
# PAGE 3 — SAFETY MONITORING DASHBOARD
# =============================================================================

elif page == "📊 Safety Monitoring Dashboard":

    st.markdown(
        """
        <div class="hero" style="padding:36px 44px;">
            <div class="hero-tag">Live Monitoring</div>
            <h1 style="font-size:2.3rem;">Safety Monitoring Dashboard</h1>
            <p>Upload a site photo or video clip to run the YOLO11 PPE detector and generate
            a compliance report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    media_type = st.radio("Select media type", ["Image", "Video"], horizontal=True)
    uploaded = st.file_uploader(
        "Upload file",
        type=["jpg", "jpeg", "png"] if media_type == "Image" else ["mp4", "mov", "avi"],
    )

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        tmp_path = os.path.join(tempfile.gettempdir(), f"siteguard_upload{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.markdown("#### Original " + media_type)
        if media_type == "Image":
            st.image(tmp_path, use_container_width=True)
        else:
            st.video(tmp_path)

        run = st.button("▶ Run Detection", type="primary")

        if run:
            if not os.path.exists(engine.MODEL_PATH):
                st.error(
                    "`best.pt` was not found in the project directory. Place your trained "
                    "model file next to app.py and reload."
                )
            else:
                with st.spinner(
                    "Running YOLO11 inference and applying business logic..."
                ):
                    if media_type == "Image":
                        frame = cv2.imread(tmp_path)
                        annotated, people, stats, out_path = engine.process_image(frame)
                        st.session_state.annotated_image_path = out_path
                        st.session_state.annotated_video_path = None
                        st.session_state.people = people
                        st.session_state.stats = stats
                        st.session_state.mode = "image"
                    else:
                        progress_bar = st.progress(
                            0.0, text="Processing video frames..."
                        )

                        def _cb(frac):
                            progress_bar.progress(
                                frac,
                                text=f"Processing video frames... {int(frac*100)}%",
                            )

                        out_path, summary = engine.process_video(
                            tmp_path,
                            output_path=os.path.join("outputs", "annotated_video.mp4"),
                            frame_skip=1,
                            progress_callback=_cb,
                        )
                        progress_bar.empty()
                        st.session_state.annotated_video_path = out_path
                        st.session_state.annotated_image_path = None
                        st.session_state.people = None
                        st.session_state.stats = summary
                        st.session_state.mode = "video"

                st.session_state.detection_done = True

    # -------------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------------
    if st.session_state.detection_done and st.session_state.stats:
        stats = st.session_state.stats
        mode = st.session_state.mode

        st.markdown(
            '<div class="section-title">Detection Result</div>', unsafe_allow_html=True
        )
        if mode == "image" and st.session_state.annotated_image_path:
            st.image(
                st.session_state.annotated_image_path,
                caption="Annotated Image",
                use_container_width=True,
            )
        elif (
            mode == "video"
            and st.session_state.annotated_video_path
            and os.path.exists(st.session_state.annotated_video_path)
        ):
            st.video(st.session_state.annotated_video_path)

        # ---- KPI Cards ----
        st.markdown(
            '<div class="section-title">AI Dashboard</div>', unsafe_allow_html=True
        )
        kpi_defs = [
            ("Total Workers", stats.get("total_workers", 0)),
            ("Compliant Workers", stats.get("compliant_workers", 0)),
            ("Violations", stats.get("violations", 0)),
            ("Helmet Violations", stats.get("helmet_violations", 0)),
            ("Vest Violations", stats.get("vest_violations", 0)),
        ]
        cols = st.columns(len(kpi_defs))
        for col, (label, val) in zip(cols, kpi_defs):
            with col:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        # ---- Compliance progress bar ----
        st.markdown("<br>", unsafe_allow_html=True)
        compliance_rate = float(stats.get("compliance_rate", 0.0))
        st.markdown(f"**Compliance Rate: {compliance_rate}%**")
        st.progress(min(max(compliance_rate / 100, 0.0), 1.0))

        # ---- Risk level badge ----
        if compliance_rate >= 80:
            risk_label, risk_class = "🟢 Low Risk", "badge-low"
        elif compliance_rate >= 50:
            risk_label, risk_class = "🟡 Medium Risk", "badge-med"
        else:
            risk_label, risk_class = "🔴 High Risk", "badge-high"
        st.markdown(
            f'<span class="badge {risk_class}">{risk_label}</span>',
            unsafe_allow_html=True,
        )

        # ---- Charts ----
        st.markdown(
            '<div class="section-title">Compliance & Violation Charts</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure(
                go.Pie(
                    labels=["Compliant", "Non-Compliant"],
                    values=[
                        stats.get("compliant_workers", 0),
                        max(
                            stats.get("total_workers", 0)
                            - stats.get("compliant_workers", 0),
                            0,
                        ),
                    ],
                    marker_colors=[GREEN, RED],
                    hole=0.5,
                )
            )
            fig.update_layout(
                title="Safe vs Unsafe Workers", template="plotly_white", height=360
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure(
                go.Bar(
                    x=["Helmet Violations", "Vest Violations"],
                    y=[
                        stats.get("helmet_violations", 0),
                        stats.get("vest_violations", 0),
                    ],
                    marker_color=[PRIMARY, RED],
                )
            )
            fig.update_layout(
                title="Violation Breakdown", template="plotly_white", height=360
            )
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            total = max(stats.get("total_workers", 0), 1)
            fig = go.Figure(
                go.Bar(
                    x=["Compliance Rate"],
                    y=[compliance_rate],
                    marker_color=GREEN,
                    text=[f"{compliance_rate}%"],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Compliance Distribution",
                template="plotly_white",
                height=340,
                yaxis_range=[0, 100],
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            risk_score = 100 - compliance_rate
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={"text": "Risk Distribution"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": DARK},
                        "steps": [
                            {"range": [0, 20], "color": GREEN},
                            {"range": [20, 50], "color": YELLOW},
                            {"range": [50, 100], "color": RED},
                        ],
                    },
                )
            )
            fig.update_layout(template="plotly_white", height=340)
            st.plotly_chart(fig, use_container_width=True)

        # ---- AI Recommendations (rule-based on real stats, not mocked) ----
        st.markdown(
            '<div class="section-title">AI Recommendations</div>',
            unsafe_allow_html=True,
        )
        recs = []
        if stats.get("helmet_violations", 0) > 0:
            recs.append(
                (
                    "alert-danger",
                    f"⚠️ {stats['helmet_violations']} worker(s) detected without a hardhat — increase helmet compliance enforcement.",
                )
            )
        if stats.get("vest_violations", 0) > 0:
            recs.append(
                (
                    "alert-danger",
                    f"⚠️ {stats['vest_violations']} worker(s) detected without a safety vest — enforce hi-vis vest policy.",
                )
            )
        if compliance_rate < 50 and stats.get("total_workers", 0) > 0:
            recs.append(
                (
                    "alert-danger",
                    "🚨 Compliance rate is below 50% — schedule an immediate safety inspection.",
                )
            )
        elif compliance_rate < 80 and stats.get("total_workers", 0) > 0:
            recs.append(
                (
                    "alert-warn",
                    "🟡 Compliance rate is moderate — provide refresher PPE training for this crew/shift.",
                )
            )
        if stats.get("violations", 0) >= 3:
            recs.append(
                (
                    "alert-warn",
                    "🔁 Multiple violations detected in this capture — investigate for repeated non-compliance patterns.",
                )
            )
        if stats.get("unknown_count", 0) if "unknown_count" in stats else 0:
            pass
        if not recs and stats.get("total_workers", 0) > 0:
            recs.append(
                (
                    "alert-ok",
                    "✅ All detected workers are fully compliant — no action required.",
                )
            )
        if stats.get("total_workers", 0) == 0:
            recs.append(
                (
                    "alert-warn",
                    "ℹ️ No workers were detected in this capture — verify camera framing or upload a clearer image/video.",
                )
            )

        for css_class, text in recs:
            st.markdown(
                f'<div class="alert-card {css_class}">{text}</div>',
                unsafe_allow_html=True,
            )

        # ---- Reports / Downloads ----
        st.markdown('<div class="section-title">Reports</div>', unsafe_allow_html=True)

        report_path, report_text = engine.generate_report(stats, mode)

        json_bytes = json.dumps(stats, indent=2, default=str).encode("utf-8")
        csv_df = pd.DataFrame([stats])
        csv_bytes = csv_df.to_csv(index=False).encode("utf-8")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button(
                "⬇ Summary Report (TXT)", report_text, file_name="summary_report.txt"
            )
        with d2:
            st.download_button(
                "⬇ JSON Report",
                json_bytes,
                file_name="detection_report.json",
                mime="application/json",
            )
        with d3:
            st.download_button(
                "⬇ CSV Report",
                csv_bytes,
                file_name="detection_report.csv",
                mime="text/csv",
            )
        with d4:
            if (
                mode == "image"
                and st.session_state.annotated_image_path
                and os.path.exists(st.session_state.annotated_image_path)
            ):
                with open(st.session_state.annotated_image_path, "rb") as f:
                    st.download_button(
                        "⬇ Annotated Image",
                        f.read(),
                        file_name="annotated_image.jpg",
                        mime="image/jpeg",
                    )
            elif (
                mode == "video"
                and st.session_state.annotated_video_path
                and os.path.exists(st.session_state.annotated_video_path)
            ):
                with open(st.session_state.annotated_video_path, "rb") as f:
                    st.download_button(
                        "⬇ Annotated Video",
                        f.read(),
                        file_name="annotated_video.mp4",
                        mime="video/mp4",
                    )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    f"""
    <div class="footer-bar">
        SiteGuard AI · Construction Safety Monitoring Platform · Powered by YOLO11 ·
        Generated {datetime.now().strftime('%Y')}
    </div>
    """,
    unsafe_allow_html=True,
)
