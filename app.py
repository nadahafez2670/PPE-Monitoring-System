"""
app.py — AI-Powered PPE Detection & Construction Site Safety Monitoring System

Frontend only. All inference and business logic lives in model_engine.py,
which is a verbatim, importable copy of PPE_Detection_BusinessLogic.ipynb.
This file does not reimplement, alter, or mock any detection logic.

Run with:  streamlit run app.py
Requires, in the same folder: best.pt, model_engine.py
Optional, used opportunistically if present:
  results/results.png, results/confusion_matrix.png, results/pr_curve.png
  results/sample_detection.jpg
  filtered_dataset/{train,valid,test}/labels
(outputs/ is used as a fallback location for the same training-artifact
filenames, and is also where the app writes its own run-time outputs —
annotated_image.jpg, annotated_video.mp4, report.txt.)
"""

import json
import os
import tempfile
import base64
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

PER_CLASS_AP50 = {
    "Person": 73.5,
    "Hardhat": 78.2,
    "Safety Vest": 73.7,
    "NO-Hardhat": 57.1,
    "NO-Safety Vest": 68.6,
}

# =============================================================================
# BACKGROUND IMAGE CONFIGURATION
# =============================================================================

# ✏️ عدّلي مسار الصورة من هنا (أدخلي الباث الخاص بك)
BACKGROUND_IMAGE_PATH = r"assets\istockphoto-1420678520-612x612.jpg"


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# تحويل الصورة إلى base64 لاستخدامها داخل الـ CSS إذا كانت موجودة
bg_image_style = ""
if os.path.exists(BACKGROUND_IMAGE_PATH):
    encoded_bg = get_base64_of_bin_file(BACKGROUND_IMAGE_PATH)
    bg_image_style = f"""
        background-image: linear-gradient(180deg, rgba(11, 17, 27, 0.85) 0%, rgba(11, 17, 27, 0.95) 100%), url('data:image/png;base64,{encoded_bg}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    """

# =============================================================================
# THEME / CSS  — dark, premium, industrial (per design reference)
# =============================================================================

BG = "#0B111B"  # main app background
BG_2 = "#111827"  # secondary background (sidebar, hero)
CARD = "#182231"  # glass card fill
BORDER = "#2A3547"  # card border
PRIMARY = "#F59E0B"  # construction orange (primary accent)
PRIMARY_LIGHT = "#FBBF24"  # golden yellow (secondary accent)
GREEN = "#22C55E"  # success
YELLOW = "#FACC15"  # warning
RED = "#EF4444"  # danger
TEXT = "#F8FAFC"  # primary text
TEXT_SECONDARY = "#CBD5E1"  # secondary text
TEXT_MUTED = "#94A3B8"  # muted text

PRIMARY_DARK = "#C77D0A"
DARK = BG_2
DARK_2 = BG
LIGHT = TEXT

# استخدام الخصائص البديلة للـ Background في حال عدم وجود مسار الصورة
default_bg_style = f"""
    background:
        linear-gradient(180deg, rgba(11,17,27,0.97) 0%, rgba(11,17,27,0.99) 100%),
        repeating-linear-gradient(
            135deg,
            rgba(245,158,11,0.05) 0px, rgba(245,158,11,0.05) 26px,
            transparent 26px, transparent 52px
        );
    background-color: {BG};
"""

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Barlow+Condensed:wght@600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}

        /* ---- Custom background dynamic loading ---- */
        .stApp {{
            {bg_image_style if bg_image_style else default_bg_style}
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }}

        h1, h2, h3, h4, h5, p, span, label, div {{
            color: {TEXT};
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {BG_2} 0%, {BG} 100%);
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT} !important;
        }}
        section[data-testid="stSidebar"] .stRadio > div {{
            gap: 0.4rem;
        }}
        section[data-testid="stSidebar"] label {{
            background: rgba(255,255,255,0.03);
            border: 1px solid transparent;
            padding: 10px 14px;
            border-radius: 10px;
            width: 100%;
            transition: all 0.18s ease;
        }}
        section[data-testid="stSidebar"] label:hover {{
            background: rgba(245,158,11,0.12);
            border: 1px solid rgba(245,158,11,0.35);
        }}

        /* ---- Fade-in entrance for the whole page body ---- */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .block-container > div {{
            animation: fadeInUp 0.5s ease-out both;
        }}

        /* ---- Hero banner ---- */
        .hero {{
            background:
                linear-gradient(120deg, rgba(245,158,11,0.10) 0%, transparent 55%),
                linear-gradient(150deg, {BG_2} 0%, #1a1207 60%, #2a1a06 100%);
            border-radius: 22px;
            padding: 56px 48px;
            color: {TEXT};
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
            border: 1px solid {BORDER};
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }}
        .hero::before {{
            content: "";
            position: absolute;
            top: -60px; right: -60px;
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(245,158,11,0.30), transparent 70%);
        }}
        .hero::after {{
            content: "";
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 6px;
            background: repeating-linear-gradient(
                45deg, {PRIMARY} 0px, {PRIMARY} 10px,
                {DARK} 10px, {DARK} 20px
            );
        }}
        .hero-tag {{
            display: inline-block;
            background: rgba(245,158,11,0.14);
            border: 1px solid {PRIMARY};
            color: {PRIMARY_LIGHT};
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
            color: {TEXT};
        }}
        .hero p {{
            font-size: 1.08rem;
            color: {TEXT_SECONDARY};
            max-width: 660px;
            line-height: 1.6;
        }}

        /* ---- Glass cards ---- */
        .card {{
            background: linear-gradient(155deg, {CARD} 0%, {BG_2} 100%);
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 24px 26px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            height: 100%;
        }}
        .card:hover {{
            transform: translateY(-4px) scale(1.01);
            border-color: rgba(245,158,11,0.55);
            box-shadow: 0 12px 30px rgba(245,158,11,0.12), 0 8px 24px rgba(0,0,0,0.35);
        }}
        .card h4 {{
            margin: 0 0 8px 0;
            font-size: 1.04rem;
            color: {TEXT};
        }}
        .card p {{
            margin: 0;
            color: {TEXT_SECONDARY};
            font-size: 0.92rem;
            line-height: 1.6;
        }}

        /* ---- Problem cards: icon chip + text ---- */
        .problem-card {{
            display: flex;
            gap: 16px;
            align-items: flex-start;
            background: linear-gradient(155deg, {CARD} 0%, {BG_2} 100%);
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 22px 24px;
            height: 100%;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .problem-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(245,158,11,0.5);
        }}
        .problem-icon {{
            flex-shrink: 0;
            width: 46px; height: 46px;
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.35rem;
            border: 1px solid;
        }}
        .problem-card h4 {{ margin: 2px 0 6px 0; font-size: 1.02rem; }}
        .problem-card p {{ margin: 0; color: {TEXT_SECONDARY}; font-size: 0.9rem; line-height: 1.55; }}

        /* ---- Solution highlight card ---- */
        .solution-card {{
            display: flex;
            gap: 22px;
            align-items: center;
            background: linear-gradient(120deg, rgba(245,158,11,0.14) 0%, {CARD} 55%);
            border: 1px solid {PRIMARY};
            border-radius: 22px;
            padding: 28px 30px;
            box-shadow: 0 8px 30px rgba(245,158,11,0.10);
        }}
        .solution-icon {{
            flex-shrink: 0;
            width: 64px; height: 64px;
            border-radius: 18px;
            background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 2rem;
            box-shadow: 0 6px 18px rgba(245,158,11,0.35);
        }}
        .solution-card p {{ margin: 0; color: {TEXT_SECONDARY}; font-size: 0.98rem; line-height: 1.65; }}

        /* ---- Objective / colorful feature cards ---- */
        .objective-card {{
            background: linear-gradient(155deg, {CARD} 0%, {BG_2} 100%);
            border: 1px solid {BORDER};
            border-left: 4px solid var(--accent, {PRIMARY});
            border-radius: 18px;
            padding: 20px 22px;
            height: 100%;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .objective-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 26px rgba(0,0,0,0.3);
        }}
        .objective-card h4 {{ margin: 8px 0 6px 0; font-size: 1.0rem; color: var(--accent, {PRIMARY}); }}
        .objective-card p {{ margin: 0; color: {TEXT_SECONDARY}; font-size: 0.88rem; line-height: 1.5; }}

        /* ---- Detected-class icon badges ---- */
        .class-badge {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }}
        .class-badge-circle {{
            width: 64px; height: 64px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.6rem;
            border: 2px solid;
            background: rgba(255,255,255,0.02);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .class-badge:hover .class-badge-circle {{
            transform: scale(1.08);
        }}
        .class-badge-label {{
            font-size: 0.85rem;
            font-weight: 600;
            color: {TEXT_SECONDARY};
        }}

        /* ---- Section titles ---- */
        .section-title {{
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 1.95rem;
            font-weight: 700;
            color: {TEXT};
            margin: 38px 0 4px 0;
            border-left: 5px solid {PRIMARY};
            padding-left: 14px;
        }}
        .section-sub {{
            color: {TEXT_MUTED};
            margin: 0 0 18px 18px;
            font-size: 0.95rem;
        }}

        /* ---- KPI cards ---- */
        .kpi {{
            background: linear-gradient(155deg, {CARD} 0%, {BG_2} 100%);
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 20px 22px;
            color: {TEXT};
            border-top: 4px solid {PRIMARY};
            height: 100%;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            animation: fadeInUp 0.6s ease-out both;
        }}
        .kpi:hover {{
            transform: translateY(-4px) scale(1.015);
            border-color: {PRIMARY};
            box-shadow: 0 10px 28px rgba(245,158,11,0.18);
        }}
        .kpi .kpi-label {{
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .kpi .kpi-value {{
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: {TEXT};
        }}

        /* ---- Risk badges ---- */
        .badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1rem;
        }}
        .badge-low {{ background: rgba(34,197,94,0.14); color: {GREEN}; border: 1px solid {GREEN}; }}
        .badge-med {{ background: rgba(250,204,21,0.14); color: {YELLOW}; border: 1px solid {YELLOW}; }}
        .badge-high {{ background: rgba(239,68,68,0.14); color: {RED}; border: 1px solid {RED}; }}

        /* ---- Alert cards ---- */
        .alert-card {{
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 10px;
            font-size: 0.93rem;
            border: 1px solid;
            border-left: 4px solid;
        }}
        .alert-danger {{ background: rgba(239,68,68,0.08); border-color: {RED}; color: #FCA5A5; }}
        .alert-warn {{ background: rgba(250,204,21,0.08); border-color: {YELLOW}; color: #FDE68A; }}
        .alert-ok {{ background: rgba(34,197,94,0.08); border-color: {GREEN}; color: #86EFAC; }}

        /* ---- Custom animated compliance bar ---- */
        .progress-track {{
            width: 100%;
            height: 14px;
            border-radius: 999px;
            background: {BG_2};
            border: 1px solid {BORDER};
            overflow: hidden;
            margin: 6px 0 4px 0;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 100%);
            width: 0%;
            animation: growFill 1.1s ease-out forwards;
            box-shadow: 0 0 12px rgba(245,158,11,0.55);
        }}
        @keyframes growFill {{
            from {{ width: 0%; }}
            to   {{ width: var(--target-width); }}
        }}

        .footer-bar {{
            margin-top: 50px;
            padding: 26px 10px 10px 10px;
            border-top: 1px solid {BORDER};
            color: {TEXT_MUTED};
            font-size: 0.85rem;
            text-align: center;
        }}

        div.stButton > button {{
            background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 100%);
            color: {BG};
            border: none;
            border-radius: 10px;
            padding: 10px 22px;
            font-weight: 700;
            transition: all 0.18s ease;
        }}
        div.stButton > button:hover {{
            filter: brightness(1.08);
            box-shadow: 0 6px 18px rgba(245,158,11,0.35);
            transform: translateY(-1px);
        }}

        div[data-testid="stFileUploader"], div[data-testid="stDataFrame"] {{
            border-radius: 14px;
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
        "<div style='font-size:0.75rem; color:#94A3B8; margin-top:18px;'>YOLOv8s · 5-class PPE detector</div>",
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
            <p>SiteGuard AI uses a custom-trained YOLOv8s model to detect workers and their
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
            "YOLOv8s object detector to camera footage or uploaded media, identifies every "
            "worker in frame, and matches each one to their hardhat and vest status using "
            "IoU-based spatial matching between person and equipment boxes."
        )
    with c2:
        DETECTED_CLASSES = [
            ("👤", "Person", GREEN),
            ("🪖", "Hardhat", PRIMARY),
            ("🦺", "Safety Vest", GREEN),
            ("🚫", "NO-Hardhat", RED),
            ("🚫", "NO-Safety Vest", RED),
        ]
        badges_html = "".join(f"""
            <div class="class-badge">
                <div class="class-badge-circle" style="border-color:{color}; color:{color};">{icon}</div>
                <div class="class-badge-label">{label}</div>
            </div>
            """ for icon, label, color in DETECTED_CLASSES)
        st.markdown(
            f"""
            <div class="card">
                <h4>🎯 Detected Classes</h4>
                <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:14px; margin-top:14px;">
                    {badges_html}
                </div>
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
            PRIMARY,
        ),
        (
            "🚫",
            "Violations Go Unnoticed",
            "PPE non-compliance is often only caught after an incident, not before.",
            RED,
        ),
        (
            "🕒",
            "Slow Reporting",
            "Paper-based safety walks produce data too late to prevent the next injury.",
            "#60A5FA",
        ),
    ]
    for col, (icon, title, desc, color) in zip(cols, problems):
        with col:
            st.markdown(
                f"""
                <div class="problem-card">
                    <div class="problem-icon" style="border-color:{color}; color:{color}; background:{color}22;">{icon}</div>
                    <div>
                        <h4 style="color:{color};">{title}</h4>
                        <p>{desc}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Solution</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="solution-card">
            <div class="solution-icon">🛡️</div>
            <p>SiteGuard AI replaces manual, after-the-fact safety walks with a continuous,
            automated check: any image or video clip from the site is run through a
            custom-trained <b style="color:{PRIMARY_LIGHT};">YOLOv8s</b> model, every worker is
            matched to their hardhat and vest status via IoU-based spatial matching, and the
            result is turned into a compliance rate, a risk badge, and a downloadable report —
            in seconds, not after an incident has already happened.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Objectives</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    objectives = [
        (
            "🔍",
            "Detect",
            "Identify every person and PPE item in an image or video frame.",
            GREEN,
        ),
        (
            "🧠",
            "Classify",
            "Match each worker to hardhat and vest status via IoU matching.",
            PRIMARY,
        ),
        (
            "📊",
            "Quantify",
            "Turn detections into compliance rate, violation counts, and risk level.",
            "#60A5FA",
        ),
        (
            "📄",
            "Report",
            "Generate downloadable reports for safety records and audits.",
            "#A78BFA",
        ),
    ]
    for col, (icon, title, desc, color) in zip(cols, objectives):
        with col:
            st.markdown(
                f"""
                <div class="objective-card" style="--accent:{color};">
                    <span style="font-size:1.5rem;">{icon}</span>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
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
    for i, (col, (label, desc)) in enumerate(zip(cols, kpis)):
        with col:
            st.markdown(
                f'<div class="kpi" style="animation-delay:{i*0.08}s;"><div class="kpi-label">{label}</div>'
                f'<div style="font-size:0.85rem; color:{TEXT_SECONDARY}; margin-top:6px;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("📤", "Upload Image/Video"),
        ("🧠", "YOLOv8s Inference"),
        ("📐", "IoU Attribute Matching"),
        ("✅", "Compliance Scoring"),
        ("📊", "Dashboard + Report"),
    ]
    cols = st.columns(len(steps))
    for i, (col, (icon, step)) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="width:52px;height:52px;border-radius:50%;background:{PRIMARY};
                        color:{DARK}; font-weight:700; font-size:1.3rem; display:flex; align-items:center;
                        justify-content:center; margin:0 auto 8px auto; box-shadow:0 4px 12px rgba(245,166,35,0.35);">{icon}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:{PRIMARY_LIGHT};">Step {i+1}</div>
                    <div style="font-size:0.82rem; color:{TEXT_SECONDARY};">{step}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if i < len(steps) - 1:
                st.markdown(
                    f"<div style='height:2px; background:{PRIMARY}; opacity:0.25; margin-top:-46px;'></div>",
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
        '<div class="section-title">System Features</div>', unsafe_allow_html=True
    )
    cols = st.columns(3)
    sys_features = [
        (
            "🎨",
            "Color-Coded Boxes",
            "Green for compliant workers, red for violations, yellow for unknown status.",
        ),
        (
            "🧾",
            "Multi-Format Reports",
            "Every run can be exported as TXT, JSON, and CSV for audit trails.",
        ),
        (
            "⚙️",
            "Per-Class Thresholds",
            "Confidence thresholds are tuned per class, not a single global cutoff.",
        ),
    ]
    for col, (icon, title, desc) in zip(cols, sys_features):
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
            <b>Model</b> → YOLOv8s, trained on the filtered dataset, exported as <code>best.pt</code>.<br><br>
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
            <p>Architecture, training configuration, and evaluation results for the YOLOv8s
            PPE detector powering this platform.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Evaluation Metrics</div>', unsafe_allow_html=True
    )
    cols = st.columns(4)
    for i, (col, (label, val)) in enumerate(zip(cols, METRICS.items())):
        with col:
            st.markdown(
                f'<div class="kpi" style="animation-delay:{i*0.08}s;"><div class="kpi-label">{label}</div>'
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#F8FAFC"),
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#F8FAFC"),
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown(
        '<div class="section-title">Per-Class AP@50</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">From the model\'s precision-recall curve, one average '
        "precision score per class at an IoU threshold of 0.5.</div>",
        unsafe_allow_html=True,
    )
    fig_class = go.Figure(
        go.Bar(
            x=list(PER_CLASS_AP50.keys()),
            y=list(PER_CLASS_AP50.values()),
            marker_color=[PRIMARY, GREEN, PRIMARY_DARK, RED, YELLOW],
            text=[f"{v}%" for v in PER_CLASS_AP50.values()],
            textposition="outside",
        )
    )
    fig_class.update_layout(
        yaxis_title="AP@50 (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#F8FAFC"),
        height=380,
        yaxis_range=[0, 100],
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_class, use_container_width=True)

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
                <p>YOLOv8s (Ultralytics), single-stage anchor-free detector, fine-tuned on the
                filtered 5-class dataset and exported as <code>best.pt</code>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Training Configuration</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    train_cfg = [
        ("Image Size", f"{engine.IMGSZ}px"),
        ("Global Confidence", engine.GLOBAL_CONF),
        ("Test-Time Augment", "Enabled" if engine.USE_AUGMENT else "Disabled"),
        ("Matching Strategy", "IoU ≥ 0.02"),
    ]
    for i, (col, (label, val)) in enumerate(zip([c1, c2, c3, c4], train_cfg)):
        with col:
            st.markdown(
                f'<div class="kpi" style="animation-delay:{i*0.08}s;"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" style="font-size:1.5rem;">{val}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    thresh_df = pd.DataFrame(
        {
            "Class": list(engine.CLASS_CONF_THRESH.keys()),
            "Confidence Threshold": list(engine.CLASS_CONF_THRESH.values()),
        }
    )
    st.dataframe(thresh_df, use_container_width=True, hide_index=True)

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
                title="Class Distribution", template="plotly_dark", height=380
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_donut = go.Figure(
                go.Pie(
                    labels=list(counts.keys()), values=list(counts.values()), hole=0.55
                )
            )
            fig_donut.update_layout(
                title="Class Distribution (Donut)", template="plotly_dark", height=380
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
    artifact_candidates = {
        "Training Curves (Results)": [
            r"results\photo_2026-07-24_14-50-34.jpg",
            "outputs/results.png",
        ],
        "Confusion Matrix": [
            r"results\photo_2026-07-24_14-50-26.jpg",
            "outputs/confusion_matrix.png",
        ],
        "Precision-Recall Curve": [
            r"results\photo_2026-07-24_14-50-12.jpg",
            "outputs/pr_curve.png",
        ],
    }
    for col, (label, candidates) in zip([c1, c2, c3], artifact_candidates.items()):
        with col:
            st.markdown(f"**{label}**")
            found_path = next((p for p in candidates if os.path.exists(p)), None)
            if found_path:
                st.image(found_path, use_container_width=True)
            else:
                st.markdown(
                    f'<div class="card" style="text-align:center; color:#9CA3AF;">'
                    f"Not found — place this file at <code>{candidates[0]}</code></div>",
                    unsafe_allow_html=True,
                )

    sample_candidates = [
        "results/sample_detection.jpg",
        "outputs/annotated_image.jpg",
    ]
    sample_path = next((p for p in sample_candidates if os.path.exists(p)), None)
    if sample_path:
        st.markdown("**Sample Detection**")
        st.image(sample_path, use_container_width=True)

# =============================================================================
# PAGE 3 — SAFETY MONITORING DASHBOARD
# =============================================================================

elif page == "📊 Safety Monitoring Dashboard":

    st.markdown(
        """
        <div class="hero" style="padding:36px 44px;">
            <div class="hero-tag">Live Monitoring</div>
            <h1 style="font-size:2.3rem;">Safety Monitoring Dashboard</h1>
            <p>Upload a site photo or video clip to run the YOLOv8s PPE detector and generate
            a compliance report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="display:flex; gap:18px; margin-bottom:18px; flex-wrap:wrap;">
            <div style="flex:1; min-width:160px; background:{DARK}; border-radius:12px;
                padding:12px 16px; color:white;">
                <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
                    background:{GREEN}; margin-right:8px;"></span>Compliant Worker
            </div>
            <div style="flex:1; min-width:160px; background:{DARK}; border-radius:12px;
                padding:12px 16px; color:white;">
                <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
                    background:{RED}; margin-right:8px;"></span>Violation
            </div>
            <div style="flex:1; min-width:160px; background:{DARK}; border-radius:12px;
                padding:12px 16px; color:white;">
                <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
                    background:{YELLOW}; margin-right:8px;"></span>Unknown Status
            </div>
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
                    "Running YOLOv8s inference and applying business logic..."
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
        people = st.session_state.people

        st.markdown(
            '<div class="section-title">Detection Result</div>', unsafe_allow_html=True
        )

        # 🔧 [تعديل عرض الفيديو والصورة هنا]
        if mode == "image" and st.session_state.annotated_image_path:
            st.image(
                st.session_state.annotated_image_path,
                caption="Annotated Image",
                use_container_width=True,
            )
        elif mode == "video" and st.session_state.annotated_video_path:
            video_path = st.session_state.annotated_video_path
            if os.path.exists(video_path):
                # قراءة ملف الفيديو كـ bytes لتفادي مشاكل الـ HTML5 Player وتحديث الصفحة
                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_path, format="video/mp4")
            else:
                st.error("Could not find processed video file.")

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
            ("Unknown Status", stats.get("unknown_count", 0)),
        ]
        cols = st.columns(len(kpi_defs))
        for i, (col, (label, val)) in enumerate(zip(cols, kpi_defs)):
            with col:
                st.markdown(
                    f'<div class="kpi" style="animation-delay:{i*0.06}s;"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        # ---- Compliance progress bar (custom, animated) ----
        st.markdown("<br>", unsafe_allow_html=True)
        compliance_rate = float(stats.get("compliance_rate", 0.0))
        clamped_pct = min(max(compliance_rate, 0.0), 100.0)
        st.markdown(
            f"""
            <div style="font-weight:600; color:{TEXT}; margin-bottom:4px;">
                Compliance Rate: <span style="color:{PRIMARY_LIGHT};">{compliance_rate}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="--target-width:{clamped_pct}%;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                title="Safe vs Unsafe Workers", template="plotly_dark", height=360
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
                title="Violation Breakdown", template="plotly_dark", height=360
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
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="#F8FAFC"),
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
            fig.update_layout(template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)

        # ---- Per-attribute worker status breakdown (image mode only, uses
        # ---- the per-person list returned directly by the business logic) ----
        if mode == "image" and people:
            st.markdown(
                '<div class="section-title">Worker Attribute Status</div>',
                unsafe_allow_html=True,
            )
            c5, c6 = st.columns(2)
            hardhat_counts = pd.Series(
                [p["hardhat_status"] for p in people]
            ).value_counts()
            vest_counts = pd.Series([p["vest_status"] for p in people]).value_counts()
            status_color = {
                "Hardhat": GREEN,
                "Safety Vest": GREEN,
                "NO-Hardhat": RED,
                "NO-Safety Vest": RED,
                "Unknown": YELLOW,
            }
            with c5:
                fig = go.Figure(
                    go.Bar(
                        x=hardhat_counts.index,
                        y=hardhat_counts.values,
                        marker_color=[
                            status_color.get(s, DARK) for s in hardhat_counts.index
                        ],
                    )
                )
                fig.update_layout(
                    title="Hardhat Status", template="plotly_dark", height=320
                )
                st.plotly_chart(fig, use_container_width=True)
            with c6:
                fig = go.Figure(
                    go.Bar(
                        x=vest_counts.index,
                        y=vest_counts.values,
                        marker_color=[
                            status_color.get(s, DARK) for s in vest_counts.index
                        ],
                    )
                )
                fig.update_layout(
                    title="Safety Vest Status", template="plotly_dark", height=320
                )
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
        if stats.get("unknown_count", 0) > 0:
            recs.append(
                (
                    "alert-warn",
                    f"🕵️ {stats['unknown_count']} worker(s) could not be confidently matched to PPE status — "
                    "consider a closer camera angle or better lighting for a clearer read.",
                )
            )
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
        SiteGuard AI · Construction Safety Monitoring Platform · Powered by YOLOv8s ·
        Generated {datetime.now().strftime('%Y')}
    </div>
    """,
    unsafe_allow_html=True,
)
