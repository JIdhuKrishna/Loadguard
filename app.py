from __future__ import annotations

from pathlib import Path
from datetime import datetime
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --------------------------- APP CONFIG ---------------------------
st.set_page_config(
    page_title="LoanGuard AI · Loan Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATASET_PATH = BASE_DIR / "dataset" / "loan_approval_dataset.csv"
FEATURES = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
]
GREEN = "#9BEA52"
DARK = "#101A1B"
MUTED = "#687477"

# --------------------------- DATA / MODEL ---------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_DIR / "loan_model.pkl")
    education_encoder = joblib.load(MODEL_DIR / "education_encoder.pkl")
    employment_encoder = joblib.load(MODEL_DIR / "employment_encoder.pkl")
    target_encoder = joblib.load(MODEL_DIR / "target_encoder.pkl")
    return model, education_encoder, employment_encoder, target_encoder

@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()
    for col in ["education", "self_employed", "loan_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

try:
    model, education_encoder, employment_encoder, target_encoder = load_artifacts()
    dataset = load_dataset()
    MODEL_ERROR = None
except Exception as exc:
    model = education_encoder = employment_encoder = target_encoder = dataset = None
    MODEL_ERROR = str(exc)

# --------------------------- SESSION STATE ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "assessments" not in st.session_state:
    st.session_state.assessments = []
if "stack_index" not in st.session_state:
    st.session_state.stack_index = 0
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# --------------------------- DESIGN SYSTEM ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --lg-green:#9BEA52; --lg-green-dark:#5C9F24; --lg-ink:#11191b; --lg-muted:#697679; --lg-line:#e7ece9; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: #f3f6f4; color:var(--lg-ink); }
header[data-testid="stHeader"] { background:rgba(243,246,244,.92); }
.block-container { max-width:1600px; padding-top:1.2rem; padding-bottom:2.4rem; }
[data-testid="stSidebar"] { background:linear-gradient(165deg,#101819 0%,#0c1214 100%); border-right:1px solid #273031; }
[data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
[data-testid="stSidebar"] * { color:#edf3ef; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#c4cfca; }
[data-testid="stSidebar"] .stRadio > div { gap:.32rem; }
[data-testid="stSidebar"] .stRadio label { padding:.55rem .72rem; border-radius:12px; border:1px solid transparent; }
[data-testid="stSidebar"] .stRadio label:has(input:checked) { background:var(--lg-green); color:#101810!important; font-weight:700; }
[data-testid="stSidebar"] .stRadio label:has(input:checked) * { color:#101810!important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display:none; }
.brand-box { background:#192323; border:1px solid #2d3938; border-radius:16px; padding:16px 14px; margin-bottom:22px; }
.brand-row { display:flex; gap:10px; align-items:center; }
.brand-icon { width:38px;height:38px;border-radius:12px;background:var(--lg-green);display:flex;align-items:center;justify-content:center;color:#10200b;font-size:20px;font-weight:800; }
.brand-title { color:#f8fcf9;font-family:Manrope,sans-serif;font-weight:800;font-size:16px;letter-spacing:-.5px; }
.brand-sub { color:#a8b5b0;font-size:10px;letter-spacing:.25px;margin-top:2px; }
.sidebar-note { margin-top:2rem; padding:14px; border:1px solid #293a33; border-radius:14px; background:linear-gradient(150deg,#14211b,#101718); }
.sidebar-note .note-title {font-weight:700;color:#eff8ee;font-size:12px;margin:5px 0;}
.sidebar-note .note-copy {font-size:11px;color:#9eaea5;line-height:1.5;}
.page-eyebrow { color:#5c9b31; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:1.3px; margin-bottom:6px; }
.page-title { font-family:Manrope,sans-serif; font-weight:800; letter-spacing:-1.25px; font-size:30px; line-height:1.15; color:#151d1d; margin:0 0 5px 0; }
.page-subtitle { color:#748080; font-size:13px; margin-bottom:20px; }
.top-chip { display:inline-flex; align-items:center; gap:7px; background:white; border:1px solid var(--lg-line); padding:8px 12px; border-radius:999px; color:#4e5a59; font-size:11px; }
.section-heading { background:#fff; border:1px solid #e6ece8; border-bottom:0; border-radius:16px 16px 0 0; padding:16px 18px 2px; margin-top:4px; }
.section-heading + div { margin-top:0; }
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.section-heading) { margin-bottom:0; }
.metric-card { background:white; border:1px solid #e6ece8; border-radius:17px; padding:17px 18px; min-height:118px; box-shadow:0 5px 18px rgba(21,38,30,.025); }
.metric-top { display:flex; align-items:center; gap:9px; color:#687472; font-size:11px; font-weight:600; }
.metric-icon { display:inline-flex; width:32px;height:32px;border-radius:11px;align-items:center;justify-content:center;background:#eaffd9;color:#386c1d;font-size:16px; }
.metric-value { font-family:Manrope,sans-serif; font-size:27px; font-weight:800; letter-spacing:-.9px; margin-top:8px; color:#151d1d; }
.metric-foot { color:#899391; font-size:10px; margin-top:1px; }
.card-title { font-family:Manrope,sans-serif; font-size:15px; font-weight:800; letter-spacing:-.35px; margin-bottom:3px; color:#17201e; }
.card-caption { color:#8a9590; font-size:11px; margin-bottom:12px; }
.model-card { background:linear-gradient(145deg,#172421 0%,#0c1414 100%); color:#f6fff6; border:1px solid #2d4236; border-radius:18px; padding:20px; min-height:235px; box-shadow:0 12px 26px rgba(20,52,31,.12); }
.model-card .model-kicker {font-size:10px;color:#b2c8b4;letter-spacing:1px;text-transform:uppercase;}
.model-card .model-name {font-family:Manrope,sans-serif;font-weight:800;font-size:21px;margin:10px 0 8px;}
.model-card .model-status {display:inline-block;background:#baff7a;color:#18350b;padding:5px 10px;border-radius:99px;font-size:10px;font-weight:800;}
.model-card .model-copy {font-size:11px;color:#bdcbc1;line-height:1.55;margin-top:14px;}
.model-stat-row {display:flex;justify-content:space-between;border-top:1px solid #304039;margin-top:16px;padding-top:12px;gap:8px;}
.model-stat {font-size:10px;color:#a8b9ae;}.model-stat strong {display:block;font-size:14px;color:#f7fff5;margin-bottom:2px;}
.stack-wrap { position:relative; min-height:230px; width:100%; overflow:hidden; padding:6px 4px 12px; box-sizing:border-box; }
.stack-card { position:absolute; top:8px; width:min(190px, 72%); height:205px; box-sizing:border-box; padding:14px; border:1px solid #dce8dc; border-radius:17px; background:linear-gradient(160deg,#ffffff 0%,#f3faef 100%); box-shadow:0 10px 24px rgba(22,42,26,.12); transition:transform .35s ease,opacity .35s ease; }
.stack-card .stack-id {font-family:Manrope,sans-serif;font-size:12px;font-weight:800;color:#18221d;margin:8px 0 10px;}
.stack-card .stack-label {font-size:9px;color:#849087;margin-top:8px;}
.stack-card .stack-value {font-size:12px;font-weight:800;color:#1b241e;margin-top:2px;}
.stack-status {display:inline-block;border-radius:99px;padding:5px 9px;font-size:9px;font-weight:800;margin-top:10px;}
.status-approved {background:#d9ffb9;color:#2e6816;}.status-rejected {background:#ffe0df;color:#a9302a;}.status-pending {background:#fff0bd;color:#8a6507;}
.stack-empty { border:1px dashed #d5e0d7; border-radius:16px; background:#fafcf9; padding:25px 18px; text-align:center; color:#7b8980; font-size:12px; }
div.stButton > button { border-radius:12px; min-height:40px; font-weight:700; border:1px solid #e0e8e1; }
div.stButton > button[kind="primary"] { background:var(--lg-green); color:#14200f; border:1px solid var(--lg-green); }
div.stButton > button[kind="primary"]:hover { background:#8bdd40; border-color:#8bdd40; color:#14200f; }
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div, .stDateInput input { border-radius:11px!important; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e6ece8; padding:13px 15px; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#78847f!important; }
[data-testid="stMetricValue"] { color:#17201d!important; font-family:Manrope,sans-serif; }
hr { border-color:#e5ebe7; }
.small-muted { color:#89958e;font-size:11px; }
div[data-testid="stForm"] { border:0; padding:0; }
.stProgress > div > div { background:var(--lg-green); }
@media (max-width: 900px) { .page-title{font-size:25px;} .metric-value{font-size:23px;} .stack-card{width:min(165px, 72%);} }
@media (max-width: 560px) { .block-container { padding-left: .8rem; padding-right: .8rem; } .stack-wrap { min-height:220px; } .stack-card { width: min(155px, 74%); height:198px; padding:12px; } .metric-value { font-size:21px; } }
@media (prefers-reduced-motion: reduce) { .stack-card{transition:none;} *{scroll-behavior:auto!important;} }
</style>
""", unsafe_allow_html=True)

# --------------------------- HELPERS ---------------------------
def money(value, decimals=0):
    if value is None or pd.isna(value):
        return "—"
    return f"₹{float(value):,.{decimals}f}"

def page_header(eyebrow, title, subtitle, right=None):
    left, right_col = st.columns([4, 1.35], vertical_alignment="center")
    with left:
        st.markdown(f'<div class="page-eyebrow">{eyebrow}</div><div class="page-title">{title}</div><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with right_col:
        if right:
            st.markdown(f'<div style="text-align:right;padding-bottom:18px">{right}</div>', unsafe_allow_html=True)

def metric_card(label, value, foot, icon="◈"):
    st.markdown(f'''<div class="metric-card"><div class="metric-top"><span class="metric-icon">{icon}</span><span>{label}</span></div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>''', unsafe_allow_html=True)

def model_is_loaded():
    return MODEL_ERROR is None and model is not None

def make_input(dependents, education, self_employed, income, loan_amount, loan_term, cibil, residential, commercial, luxury, bank_assets):
    education_value = education_encoder.transform([education])[0]
    employment_value = employment_encoder.transform([self_employed])[0]
    row = {
        "no_of_dependents": int(dependents),
        "education": education_value,
        "self_employed": employment_value,
        "income_annum": float(income),
        "loan_amount": float(loan_amount),
        "loan_term": int(loan_term),
        "cibil_score": int(cibil),
        "residential_assets_value": float(residential),
        "commercial_assets_value": float(commercial),
        "luxury_assets_value": float(luxury),
        "bank_asset_value": float(bank_assets),
    }
    input_df = pd.DataFrame([row])
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[list(model.feature_names_in_)]
    else:
        input_df = input_df[FEATURES]
    return input_df

def run_prediction(values):
    input_df = make_input(**values)
    encoded_prediction = model.predict(input_df)[0]
    result = str(target_encoder.inverse_transform([encoded_prediction])[0]).strip()
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        class_names = [str(c).strip() for c in target_encoder.classes_]
        if result in class_names:
            confidence = float(probs[class_names.index(result)]) * 100
    total_assets = sum(values[k] for k in ["residential", "commercial", "luxury", "bank_assets"])
    income = float(values["income"])
    loan_amount = float(values["loan_amount"])
    return {
        "id": f"LN-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}",
        "created_at": datetime.now().strftime("%d %b %Y · %I:%M %p"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": result,
        "confidence": confidence,
        "cibil": int(values["cibil"]),
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": int(values["loan_term"]),
        "dependents": int(values["dependents"]),
        "education": values["education"],
        "self_employed": values["self_employed"],
        "residential": float(values["residential"]),
        "commercial": float(values["commercial"]),
        "luxury": float(values["luxury"]),
        "bank_assets": float(values["bank_assets"]),
        "total_assets": float(total_assets),
        "loan_to_income": loan_amount / income if income > 0 else None,
        "asset_to_loan": total_assets / loan_amount if loan_amount > 0 else None,
    }

def status_class(status):
    s = str(status).lower()
    if "approv" in s:
        return "status-approved"
    if "reject" in s:
        return "status-rejected"
    return "status-pending"

def render_card_stack(items, prefix="overview"):
    if not items:
        st.markdown('<div class="stack-empty"><div style="font-size:24px;margin-bottom:8px">▤</div><strong>No assessments in this session yet</strong><br><span>Run a real model prediction from New Assessment. The result will appear here.</span></div>', unsafe_allow_html=True)
        return
    items = list(reversed(items))  # newest first
    st.session_state.stack_index = min(st.session_state.stack_index, max(0, len(items)-1))
    idx = st.session_state.stack_index
    active = items[idx]
    visible = items[idx:idx+4]
    html = ['<div class="stack-wrap">']
    for pos, item in enumerate(reversed(visible)):
        offset = len(visible) - pos - 1
        left = 8 + offset * 30
        scale = 1 - offset * .035
        opacity = 1 - offset * .12
        z = 10 - offset
        html.append(f'''<div class="stack-card" style="left:{left}px;z-index:{z};transform:scale({scale});opacity:{opacity};">
          <div style="display:flex;justify-content:space-between;align-items:center"><span class="metric-icon">♙</span><span style="font-size:10px;color:#9aa59d">•••</span></div>
          <div class="stack-label">{item.get('created_at','Assessment')}</div><div class="stack-id">{item.get('id','Application')}</div>
          <div class="stack-label">Loan amount</div><div class="stack-value">{money(item.get('loan_amount'))}</div>
          <div style="display:flex;gap:12px;margin-top:8px"><div><div class="stack-label">CIBIL</div><div class="stack-value">{item.get('cibil','—')}</div></div><div><div class="stack-label">Income</div><div class="stack-value">{money(item.get('income'))}</div></div></div>
          <span class="stack-status {status_class(item.get('status','Pending'))}">{item.get('status','Pending')}</span>
        </div>''')
    html.append('</div>')
    st.markdown(''.join(html), unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("← Previous", key=f"{prefix}_prev", disabled=idx <= 0, use_container_width=True):
            st.session_state.stack_index = max(0, idx-1)
            st.rerun()
    with c2:
        st.markdown(f'<div style="text-align:center;padding-top:10px;color:#87928a;font-size:11px">Assessment {idx+1} of {len(items)} · newest first</div>', unsafe_allow_html=True)
    with c3:
        if st.button("Next →", key=f"{prefix}_next", disabled=idx >= len(items)-1, use_container_width=True):
            st.session_state.stack_index = min(len(items)-1, idx+1)
            st.rerun()
    with st.expander("View selected assessment details"):
        d1, d2, d3 = st.columns(3)
        d1.write(f"**Application ID**\n\n{active.get('id')}")
        d2.write(f"**Prediction**\n\n{active.get('status')}")
        d3.write(f"**Model confidence**\n\n{active.get('confidence'):.1f}%" if active.get('confidence') is not None else "**Model confidence**\n\nNot available")

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown('''<div class="brand-box"><div class="brand-row"><div class="brand-icon">⬡</div><div><div class="brand-title">LoanGuard AI</div><div class="brand-sub">LOAN RISK ASSESSMENT</div></div></div></div>''', unsafe_allow_html=True)
    pages = ["Overview", "New Assessment", "Prediction Result", "Applications", "Risk Analytics", "Model Performance", "Settings"]
    icons = {"Overview":"⌂", "New Assessment":"＋", "Prediction Result":"✓", "Applications":"▤", "Risk Analytics":"▥", "Model Performance":"⌁", "Settings":"⚙"}
    selected = st.radio("Navigation", pages, index=pages.index(st.session_state.page) if st.session_state.page in pages else 0, format_func=lambda p: f"{icons[p]}    {p}", label_visibility="collapsed", key=f"nav_radio_{st.session_state.page}")
    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()
    st.markdown('''<div class="sidebar-note"><div style="color:#9bea52;font-size:11px;font-weight:800">● AI POWERED</div><div class="note-title">Credit Intelligence</div><div class="note-copy">Decision support based on your trained machine learning model.</div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="small-muted" style="padding:18px 4px 0">LoanGuard AI · Local application</div>', unsafe_allow_html=True)

# Model error state is explicit; do not render misleading analytics if files failed.
if MODEL_ERROR:
    st.error("The model or dataset could not be loaded. The application is not able to make predictions.")
    st.code(MODEL_ERROR)
    st.info("Verify that the model files exist in ./model and the dataset exists in ./dataset.")
    st.stop()

# Dataset-derived stats. The dataset is training/reference data, not live applications.
clean_status = dataset["loan_status"].astype(str).str.strip()
total_records = int(len(dataset))
approved_count = int((clean_status.str.lower() == "approved").sum())
rejected_count = int((clean_status.str.lower() == "rejected").sum())
approved_pct = approved_count / total_records * 100 if total_records else 0
rejected_pct = rejected_count / total_records * 100 if total_records else 0
avg_cibil = float(pd.to_numeric(dataset["cibil_score"], errors="coerce").mean())
avg_income = float(pd.to_numeric(dataset["income_annum"], errors="coerce").mean())
avg_loan = float(pd.to_numeric(dataset["loan_amount"], errors="coerce").mean())
avg_term = float(pd.to_numeric(dataset["loan_term"], errors="coerce").mean())

# --------------------------- OVERVIEW ---------------------------
if st.session_state.page == "Overview":
    page_header("WORKSPACE / OVERVIEW", "Welcome back, Jidhu!", "Your loan assessment workspace and dataset insights at a glance.", '<span class="top-chip">● Model ready</span>')
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1: metric_card("Dataset records", f"{total_records:,}", "Reference dataset · not live applications", "▤")
    with k2: metric_card("Approved in dataset", f"{approved_pct:.1f}%", f"{approved_count:,} records labelled approved", "✓")
    with k3: metric_card("Rejected in dataset", f"{rejected_pct:.1f}%", f"{rejected_count:,} records labelled rejected", "×")
    with k4: metric_card("Average CIBIL score", f"{avg_cibil:.0f}", "Dataset average", "◉")

    left, middle, right = st.columns([1.15, 1.2, .9], gap="medium")
    with left:
        st.markdown('''<div class="section-heading"><div class="card-title">Loan status distribution</div><div class="card-caption">Actual labels in the supplied dataset</div></div>''', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(labels=["Approved", "Rejected"], values=[approved_count, rejected_count], hole=.68, marker=dict(colors=["#9BEA52", "#ff7772"], line=dict(color="#ffffff", width=4)), textinfo="none", hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>")])
        fig.update_layout(height=220, margin=dict(l=0,r=0,t=0,b=0), showlegend=True, legend=dict(orientation="h", y=-.08, x=.08, font=dict(size=10)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", annotations=[dict(text=f"{total_records:,}<br><span style='font-size:10px'>records</span>", x=.5,y=.5,showarrow=False,font=dict(size=20,color="#18201c"))])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False}, key="overview_status")
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
    with middle:
        st.markdown('''<div class="section-heading"><div class="card-title">CIBIL score distribution</div><div class="card-caption">Distribution across dataset records</div></div>''', unsafe_allow_html=True)
        cibil_series = pd.to_numeric(dataset["cibil_score"], errors="coerce").dropna()
        fig2 = px.histogram(x=cibil_series, nbins=18, color_discrete_sequence=["#83d83d"])
        fig2.update_layout(height=220, margin=dict(l=0,r=0,t=5,b=0), bargap=.12, xaxis_title="CIBIL score", yaxis_title="Records", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#6f7b75",size=10))
        fig2.update_xaxes(showgrid=False, zeroline=False); fig2.update_yaxes(gridcolor="#edf1ed", zeroline=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False}, key="overview_cibil")
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
    with right:
        st.markdown(f'''<div class="model-card"><div class="model-kicker">Inference engine</div><div class="model-name">Random Forest</div><span class="model-status">● Model loaded</span><div class="model-copy">Trained model artifacts loaded successfully. Ready to assess applicant inputs.</div><div class="model-stat-row"><div class="model-stat"><strong>{getattr(model,'n_features_in_',len(FEATURES))}</strong>Input features</div><div class="model-stat"><strong>{getattr(model,'n_estimators','—')}</strong>Trees</div><div class="model-stat"><strong>Ready</strong>Status</div></div></div>''', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('''<div class="section-heading"><div class="card-title">Dataset insights</div><div class="card-caption">Reference-data averages</div></div>''', unsafe_allow_html=True)
        st.write("**Average annual income**")
        st.markdown(f"<div style='font-size:20px;font-weight:800'>{money(avg_income)}</div>", unsafe_allow_html=True)
        st.write("**Average loan amount**")
        st.markdown(f"<div style='font-size:20px;font-weight:800'>{money(avg_loan)}</div>", unsafe_allow_html=True)
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c_left, c_right = st.columns([1.45, 1], gap="medium")
    with c_left:
        st.markdown('''<div class="section-heading"><div class="card-title">Recent applications</div><div class="card-caption">Real predictions created in this browser session · newest first</div></div>''', unsafe_allow_html=True)
        render_card_stack(st.session_state.assessments, "overview_stack")
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
    with c_right:
        st.markdown('''<div class="section-heading"><div class="card-title">Financial profile</div><div class="card-caption">Calculated from the provided dataset</div></div>''', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a: st.metric("Average loan term", f"{avg_term:.1f} years")
        with b: st.metric("Self-employed", f"{(dataset['self_employed'].str.lower()=='yes').mean()*100:.1f}%")
        st.caption("These values describe the supplied reference dataset. They are not statistics from submitted applications.")
        if st.button("＋  Start new assessment", type="primary", use_container_width=True):
            st.session_state.page = "New Assessment"; st.rerun()
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.

# --------------------------- NEW ASSESSMENT ---------------------------
elif st.session_state.page == "New Assessment":
    page_header("ASSESSMENT WORKFLOW", "New loan assessment", "Enter applicant information, review the details, and run the actual trained Random Forest model.", '<span class="top-chip">● Real inference</span>')
    st.progress(1.0, text="Applicant details · complete the form to run the assessment")
    st.caption("Enter the applicant’s actual values. The starting values are editable form defaults and are not a prediction.")
    st.markdown('''<div class="section-heading"><div class="card-title">01 · Applicant profile</div><div class="card-caption">Enter the applicant’s profile information.</div></div>''', unsafe_allow_html=True)
    with st.form("loan_assessment_form", clear_on_submit=False):
        c1,c2,c3 = st.columns(3)
        with c1: dependents = st.number_input("Number of dependents", min_value=0, max_value=20, value=2, step=1)
        with c2: self_employed = st.selectbox("Self-employed", list(employment_encoder.classes_))
        with c3: education = st.selectbox("Education", list(education_encoder.classes_))
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
        st.markdown('''<div class="section-heading"><div class="card-title">02 · Loan & credit details</div><div class="card-caption">Financial values should match the units used in the training dataset.</div></div>''', unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        with a: income = st.number_input("Annual income (₹)", min_value=1, value=500000, step=50000)
        with b: loan_amount = st.number_input("Loan amount (₹)", min_value=1, value=1000000, step=50000)
        with c: loan_term = st.number_input("Loan term (years)", min_value=1, max_value=50, value=10, step=1)
        with d: cibil = st.number_input("CIBIL score", min_value=300, max_value=900, value=720, step=1)
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
        st.markdown('''<div class="section-heading"><div class="card-title">03 · Asset valuation</div><div class="card-caption">Declare the available asset values.</div></div>''', unsafe_allow_html=True)
        e1,e2,e3,e4 = st.columns(4)
        with e1: residential = st.number_input("Residential assets (₹)", min_value=0, value=1000000, step=50000)
        with e2: commercial = st.number_input("Commercial assets (₹)", min_value=0, value=500000, step=50000)
        with e3: luxury = st.number_input("Luxury assets (₹)", min_value=0, value=500000, step=50000)
        with e4: bank_assets = st.number_input("Bank assets (₹)", min_value=0, value=500000, step=50000)
        total_assets = residential + commercial + luxury + bank_assets
        st.markdown(f"<div style='background:#f4faef;border:1px solid #e0efcf;border-radius:12px;padding:12px 14px;margin:10px 0 16px'><span style='font-size:11px;color:#6f806b'>TOTAL DECLARED ASSETS</span><div style='font-size:22px;font-weight:800;color:#263b1d'>{money(total_assets)}</div></div>", unsafe_allow_html=True)
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
        submitted = st.form_submit_button("Run real model prediction  →", type="primary", use_container_width=True)

    if submitted:
        if income <= 0 or loan_amount <= 0:
            st.error("Annual income and loan amount must be greater than zero.")
        else:
            values = {"dependents":dependents,"education":education,"self_employed":self_employed,"income":income,"loan_amount":loan_amount,"loan_term":loan_term,"cibil":cibil,"residential":residential,"commercial":commercial,"luxury":luxury,"bank_assets":bank_assets}
            try:
                with st.spinner("Running inference with the saved Random Forest model…"):
                    record = run_prediction(values)
                st.session_state.assessments.append(record)
                st.session_state.latest_result = record
                st.session_state.stack_index = 0
                st.session_state.page = "Prediction Result"
                st.rerun()
            except Exception as exc:
                st.error("The model prediction failed. No result has been saved.")
                st.caption(f"Technical detail: {type(exc).__name__}. Check the terminal logs for diagnostics.")

# --------------------------- PREDICTION RESULT ---------------------------
elif st.session_state.page == "Prediction Result":
    record = st.session_state.latest_result
    if not record:
        page_header("ASSESSMENT RESULT", "No prediction yet", "Run a real assessment before viewing a result.")
        st.info("No prediction has been created in this session.")
        if st.button("Start assessment", type="primary"):
            st.session_state.page = "New Assessment"; st.rerun()
    else:
        page_header("ASSESSMENT RESULT", "Loan status prediction", "Model-generated outcome based on the submitted applicant information.", f'<span class="top-chip">{record["id"]}</span>')
        is_approved = "approv" in record["status"].lower()
        bg = "#edffdf" if is_approved else "#fff0ef"
        fg = "#2d6818" if is_approved else "#ad322e"
        symbol = "✓" if is_approved else "×"
        st.markdown(f'''<div style="background:{bg};border:1px solid {'#c9efad' if is_approved else '#ffd2cf'};border-radius:20px;padding:24px;margin-bottom:16px"><div style="font-size:11px;text-transform:uppercase;letter-spacing:1.1px;color:{fg};font-weight:800">Random Forest prediction</div><div style="font-family:Manrope,sans-serif;font-size:32px;font-weight:800;color:{fg};margin:7px 0">{symbol} {record['status'].upper()}</div><div style="font-size:12px;color:{fg}">Application ID: {record['id']} · {record['created_at']}</div></div>''', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("CIBIL score", f"{record['cibil']}/900")
        with m2: st.metric("Loan-to-income ratio", f"{record['loan_to_income']:.2f}x" if record['loan_to_income'] is not None else "Not available")
        with m3: st.metric("Asset-to-loan ratio", f"{record['asset_to_loan']:.2f}x" if record['asset_to_loan'] is not None else "Not available")
        with m4: st.metric("Total declared assets", money(record['total_assets']))
        left,right = st.columns([1.15,1], gap="medium")
        with left:
            st.markdown('''<div class="section-heading"><div class="card-title">Applicant summary</div><div class="card-caption">Values used for this prediction</div></div>''', unsafe_allow_html=True)
            summary = pd.DataFrame({"Field":["Dependents","Education","Self-employed","Annual income","Loan amount","Loan term","CIBIL score","Residential assets","Commercial assets","Luxury assets","Bank assets"],"Value":[record['dependents'],record['education'],record['self_employed'],money(record['income']),money(record['loan_amount']),f"{record['loan_term']} years",record['cibil'],money(record['residential']),money(record['commercial']),money(record['luxury']),money(record['bank_assets'])]})
            st.dataframe(summary, hide_index=True, use_container_width=True)
            # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
        with right:
            st.markdown('''<div class="section-heading"><div class="card-title">Model confidence</div><div class="card-caption">Classifier probability for the predicted class, if supported by the saved model</div></div>''', unsafe_allow_html=True)
            if record["confidence"] is not None:
                st.markdown(f"<div style='font-size:38px;font-weight:800;color:#243521'>{record['confidence']:.1f}%</div>", unsafe_allow_html=True)
                st.progress(max(0,min(1,record['confidence']/100)))
                st.caption("Model confidence is not a guarantee of repayment or a bank's final lending decision.")
            else:
                st.info("This model does not expose class probabilities.")
            # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            if st.button("＋ New assessment", type="primary", use_container_width=True):
                st.session_state.page = "New Assessment"; st.rerun()
        st.caption("This result is a machine-learning prediction, not a guarantee of an actual lending decision. Applications are held in session memory only and are cleared when the session ends.")

# --------------------------- APPLICATIONS ---------------------------
elif st.session_state.page == "Applications":
    page_header("WORKSPACE / RECORDS", "Applications", "Browse assessments made during this session. The supplied training dataset is not presented as live application history.")
    records = list(reversed(st.session_state.assessments))
    if records:
        search = st.text_input("Search by application ID or status", placeholder="e.g. LN-2026…")
        filter_status = st.selectbox("Filter status", ["All statuses", "Approved", "Rejected"])
        filtered = [r for r in records if search.lower() in (r['id']+" "+r['status']).lower() and (filter_status == "All statuses" or r['status'].lower() == filter_status.lower())]
        render_card_stack(filtered, "applications_stack")
        rows = [{"Application ID":r["id"],"Created":r["created_at"],"Prediction":r["status"],"Loan amount":money(r["loan_amount"]),"CIBIL":r["cibil"],"Confidence":f"{r['confidence']:.1f}%" if r["confidence"] is not None else "Not available"} for r in filtered]
        st.markdown("### Session assessment list")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="stack-empty"><div style="font-size:28px;margin-bottom:8px">▤</div><strong>No saved assessments in this session</strong><br><span>New predictions will appear here. The original project does not include persistent application storage.</span></div>', unsafe_allow_html=True)
        if st.button("Create first assessment", type="primary"):
            st.session_state.page = "New Assessment"; st.rerun()

# --------------------------- RISK ANALYTICS ---------------------------
elif st.session_state.page == "Risk Analytics":
    page_header("DATA EXPLORATION", "Risk analytics", "Explore patterns in the supplied loan dataset. These charts describe reference data, not live lending decisions.")
    r1,r2,r3,r4 = st.columns(4)
    with r1: st.metric("Dataset records", f"{total_records:,}")
    with r2: st.metric("Average income", money(avg_income))
    with r3: st.metric("Average loan amount", money(avg_loan))
    with r4: st.metric("Average CIBIL", f"{avg_cibil:.0f}")
    c1,c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('''<div class="section-heading"><div class="card-title">Income vs loan amount</div><div class="card-caption">Each point represents a dataset record</div></div>''', unsafe_allow_html=True)
        plot_df = dataset[["income_annum","loan_amount","loan_status"]].copy()
        fig = px.scatter(plot_df, x="income_annum", y="loan_amount", color="loan_status", color_discrete_map={"Approved":"#83d83d","Rejected":"#ff7772"}, opacity=.65, labels={"income_annum":"Annual income (₹)","loan_amount":"Loan amount (₹)","loan_status":"Dataset label"})
        fig.update_layout(height=340,margin=dict(l=0,r=0,t=5,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#6f7b75",size=10),legend_title_text="")
        fig.update_xaxes(gridcolor="#edf1ed",zeroline=False);fig.update_yaxes(gridcolor="#edf1ed",zeroline=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="risk_scatter")
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
    with c2:
        st.markdown('''<div class="section-heading"><div class="card-title">CIBIL by dataset label</div><div class="card-caption">Distribution by recorded approval label</div></div>''', unsafe_allow_html=True)
        fig = px.box(dataset, x="loan_status", y="cibil_score", color="loan_status", color_discrete_map={"Approved":"#83d83d","Rejected":"#ff7772"}, labels={"loan_status":"Dataset label","cibil_score":"CIBIL score"})
        fig.update_layout(height=340,margin=dict(l=0,r=0,t=5,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#6f7b75",size=10),showlegend=False)
        fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor="#edf1ed",zeroline=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="risk_box")
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
    st.caption("Associations in this dataset do not establish causation and should not be interpreted as a complete assessment of lending fairness or model suitability.")

# --------------------------- MODEL PERFORMANCE ---------------------------
elif st.session_state.page == "Model Performance":
    page_header("MODEL GOVERNANCE", "Model performance", "Evaluation using the dataset split specified by the project's training script.")
    try:
        eval_df = dataset.copy()
        eval_df.columns = eval_df.columns.str.strip()
        for col in ["education", "self_employed", "loan_status"]:
            eval_df[col] = eval_df[col].astype(str).str.strip()
        X = eval_df.drop(columns=["loan_id", "loan_status"])
        X["education"] = education_encoder.transform(X["education"])
        X["self_employed"] = employment_encoder.transform(X["self_employed"])
        y = target_encoder.transform(eval_df["loan_status"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        # Evaluate the saved model against the same held-out split; do not retrain it.
        y_pred = model.predict(X_test[list(model.feature_names_in_)] if hasattr(model,"feature_names_in_") else X_test[FEATURES])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("Holdout accuracy", f"{accuracy*100:.2f}%")
        with m2: st.metric("Weighted precision", f"{precision*100:.2f}%")
        with m3: st.metric("Weighted recall", f"{recall*100:.2f}%")
        with m4: st.metric("Weighted F1-score", f"{f1*100:.2f}%")
        st.markdown('''<div class="section-heading"><div class="card-title">Confusion matrix</div><div class="card-caption">Saved model predictions against the held-out test labels</div></div>''', unsafe_allow_html=True)
        labels = [str(x).strip() for x in target_encoder.classes_]
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(target_encoder.classes_)))
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale=[[0,"#f5faef"],[1,"#8bdc45"]], labels=dict(x="Predicted label",y="Actual label",color="Count"))
        fig.update_layout(height=340,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#6f7b75",size=11),coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="confusion_matrix")
        # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
        st.caption("Metrics are computed on the held-out split used by the project training script. They describe performance on this dataset split and do not guarantee real-world performance.")
        if hasattr(model, "feature_importances_"):
            st.markdown('''<div class="section-heading"><div class="card-title">Random Forest feature importance</div><div class="card-caption">Importance values exposed by the saved estimator; not causal explanations</div></div>''', unsafe_allow_html=True)
            imp = pd.DataFrame({"Feature":list(model.feature_names_in_) if hasattr(model,"feature_names_in_") else FEATURES,"Importance":model.feature_importances_}).sort_values("Importance",ascending=True)
            fig = px.bar(imp,x="Importance",y="Feature",orientation="h",color_discrete_sequence=["#8bdc45"])
            fig.update_layout(height=390,margin=dict(l=0,r=0,t=5,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#6f7b75",size=10))
            fig.update_xaxes(gridcolor="#edf1ed");fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False},key="feature_importance")
            # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
    except Exception as exc:
        st.error("Could not calculate evaluation metrics using the existing saved model and dataset split.")
        st.caption("No performance values have been fabricated. Check encoder versions and the training environment.")
        st.caption(f"Technical detail: {type(exc).__name__}. Check the terminal logs for diagnostics.")

# --------------------------- SETTINGS ---------------------------
elif st.session_state.page == "Settings":
    page_header("PREFERENCES", "Settings", "Application information and current model status.")
    st.markdown('''<div class="section-heading"><div class="card-title">Model and data</div><div class="card-caption">Current local project configuration</div></div>''', unsafe_allow_html=True)
    st.write("**Model type:** Random Forest Classifier")
    st.write(f"**Model loaded:** {'Yes' if model_is_loaded() else 'No'}")
    st.write(f"**Model input features:** {getattr(model,'n_features_in_','Not available')}")
    st.write(f"**Dataset records:** {total_records:,}")
    st.write("**Assessment storage:** Session memory only")
    st.caption("Assessments are not written to a database or persistent file. Refreshing or ending the session may clear the history. No user authentication or cloud notifications are configured in the original project.")
    # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('''<div class="section-heading"><div class="card-title">Session data</div><div class="card-caption">Clear only the current session’s assessment history.</div></div>''', unsafe_allow_html=True)
    st.write(f"Assessments created this session: **{len(st.session_state.assessments)}**")
    confirm = st.checkbox("I understand this only clears this session’s assessment history")
    if st.button("Clear session assessments", disabled=not confirm):
        st.session_state.assessments = []
        st.session_state.latest_result = None
        st.session_state.stack_index = 0
        st.success("Current session assessment history cleared.")
        st.rerun()
    # Layout wrapper removed: Streamlit widgets must not be nested in raw HTML containers.

# --------------------------- FOOTER ---------------------------
st.markdown('<div style="border-top:1px solid #e4ebe6;margin-top:30px;padding-top:14px;color:#95a099;font-size:10px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap"><span>LoanGuard AI · Machine Learning Decision Support</span><span>Predictions are model outputs, not lending guarantees.</span></div>', unsafe_allow_html=True)
