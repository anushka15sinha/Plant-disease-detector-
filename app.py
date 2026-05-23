import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time
import io
import base64

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafScan",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── DISEASE CLASSES — exact 15 folders from your dataset (alphabetical) ────
# [00] Pepper__bell___Bacterial_spot
# [01] Pepper__bell___healthy
# [02] Potato___Early_blight
# [03] Potato___Late_blight
# [04] Potato___healthy
# [05] Tomato_Bacterial_spot
# [06] Tomato_Early_blight
# [07] Tomato_Late_blight
# [08] Tomato_Leaf_Mold
# [09] Tomato_Septoria_leaf_spot
# [10] Tomato_Spider_mites_Two_spotted_spider_mite
# [11] Tomato__Target_Spot
# [12] Tomato__Tomato_YellowLeaf__Curl_Virus
# [13] Tomato__Tomato_mosaic_virus
# [14] Tomato_healthy
CLASSES = [
    ("Bell Pepper", "Bacterial Spot",           "🫑", "medium"),
    ("Bell Pepper", "Healthy",                   "🫑", "none"),
    ("Potato",      "Early Blight",              "🥔", "medium"),
    ("Potato",      "Late Blight",               "🥔", "critical"),
    ("Potato",      "Healthy",                   "🥔", "none"),
    ("Tomato",      "Bacterial Spot",            "🍅", "high"),
    ("Tomato",      "Early Blight",              "🍅", "medium"),
    ("Tomato",      "Late Blight",               "🍅", "critical"),
    ("Tomato",      "Leaf Mold",                 "🍅", "medium"),
    ("Tomato",      "Septoria Leaf Spot",         "🍅", "high"),
    ("Tomato",      "Spider Mites",              "🍅", "medium"),
    ("Tomato",      "Target Spot",               "🍅", "medium"),
    ("Tomato",      "Yellow Leaf Curl Virus",     "🍅", "critical"),
    ("Tomato",      "Mosaic Virus",              "🍅", "high"),
    ("Tomato",      "Healthy",                   "🍅", "none"),
]
NUM_CLASSES = 15
CLASS_NAMES = [f"{c[0]} — {c[1]}" for c in CLASSES]

DISEASE_INFO = {
    "Bacterial Spot": {
        "cause": "Bacteria Xanthomonas spp.",
        "symptoms": "Small water-soaked spots turning brown with yellow halos on leaves and fruit.",
        "treatment": "Copper bactericides, avoid overhead watering, remove infected material.",
        "prevention": "Use disease-free transplants, copper sprays preventively, crop rotation.",
    },
    "Early Blight": {
        "cause": "Fungus Alternaria solani",
        "symptoms": "Dark concentric ring lesions (bullseye pattern) starting on older leaves.",
        "treatment": "Chlorothalonil or copper-based fungicides every 7–10 days.",
        "prevention": "Crop rotation, mulching, remove plant debris after harvest.",
    },
    "Late Blight": {
        "cause": "Oomycete Phytophthora infestans",
        "symptoms": "Dark water-soaked lesions rapidly expanding; white mold on undersides.",
        "treatment": "Apply mancozeb or chlorothalonil immediately. Destroy infected plants.",
        "prevention": "Certified disease-free seed, avoid overhead irrigation, good airflow.",
    },
    "Leaf Mold": {
        "cause": "Fungus Passalora fulva",
        "symptoms": "Pale green to yellow spots on upper leaf surface, olive-green mold below.",
        "treatment": "Fungicides (chlorothalonil, mancozeb), improve ventilation.",
        "prevention": "Reduce humidity, avoid wetting foliage, resistant varieties.",
    },
    "Septoria Leaf Spot": {
        "cause": "Fungus Septoria lycopersici",
        "symptoms": "Small circular spots with dark borders and light gray centers.",
        "treatment": "Chlorothalonil, mancozeb, or copper fungicides at first sign.",
        "prevention": "Mulching, crop rotation, remove infected plant debris.",
    },
    "Spider Mites": {
        "cause": "Tetranychus urticae (Two-Spotted Spider Mite)",
        "symptoms": "Stippled, bronzed leaves; fine webbing on undersides; leaf drop.",
        "treatment": "Miticides (abamectin, bifenazate), insecticidal soap, neem oil.",
        "prevention": "Maintain humidity, avoid dusty conditions, introduce predatory mites.",
    },
    "Target Spot": {
        "cause": "Fungus Corynespora cassiicola",
        "symptoms": "Circular lesions with concentric rings resembling a target.",
        "treatment": "Fungicides (azoxystrobin, difenoconazole) at early infection.",
        "prevention": "Crop rotation, good air circulation, avoid leaf wetness.",
    },
    "Yellow Leaf Curl Virus": {
        "cause": "Begomovirus transmitted by whiteflies",
        "symptoms": "Upward leaf curl, yellowing margins, stunted growth, reduced fruit set.",
        "treatment": "No direct cure — remove infected plants immediately, control whiteflies.",
        "prevention": "Insect-proof screens, reflective mulches, resistant varieties.",
    },
    "Mosaic Virus": {
        "cause": "Tomato Mosaic Virus (ToMV), spread by contact and seed",
        "symptoms": "Mottled light and dark green mosaic pattern, distorted leaves.",
        "treatment": "No cure — remove and destroy infected plants. Disinfect tools.",
        "prevention": "Use certified virus-free seed, wash hands, resistant varieties.",
    },
    "Healthy": {
        "cause": "No disease detected",
        "symptoms": "Plant appears healthy with no visible disease symptoms.",
        "treatment": "Continue regular monitoring and maintenance practices.",
        "prevention": "Maintain proper irrigation, fertilization, and air circulation.",
    },
}

SEVERITY_CONFIG = {
    "none":     {"label": "HEALTHY",   "color": "#22c55e", "bg": "#052e16", "icon": "✓"},
    "medium":   {"label": "MODERATE",  "color": "#f59e0b", "bg": "#1c1400", "icon": "⚠"},
    "high":     {"label": "SEVERE",    "color": "#ef4444", "bg": "#1c0000", "icon": "✕"},
    "critical": {"label": "CRITICAL",  "color": "#dc2626", "bg": "#1c0000", "icon": "☠"},
}

# ─── MODEL ──────────────────────────────────────────────────────────────────
MODEL_PATH = "plant_disease_model.pth"

@st.cache_resource
def load_model():
    import os
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        st.session_state["model_source"] = "fine-tuned"
    else:
        st.session_state["model_source"] = "imagenet-pretrained"
    model.eval()
    return model

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image: Image.Image, model):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    top5_idx = probs.argsort(descending=True)[:5].tolist()
    results = []
    for idx in top5_idx:
        plant, disease, emoji, severity = CLASSES[idx]
        results.append({
            "class_idx": idx,
            "plant": plant,
            "disease": disease,
            "emoji": emoji,
            "severity": severity,
            "confidence": float(probs[idx]),
            "label": CLASS_NAMES[idx],
        })
    return results

# ─── CSS ────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Bricolage+Grotesque:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #090f0a !important;
    color: #d4e8d0 !important;
    font-family: 'Bricolage Grotesque', sans-serif !important;
}
[data-testid="stApp"] { background: #090f0a !important; }

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stAppViewContainer"] > section > div { padding-top: 0 !important; }
[data-testid="block-container"] { padding: 0 2rem 3rem !important; max-width: 1200px !important; }

.hero { text-align: center; padding: 4rem 2rem 2rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.25em; text-transform: uppercase;
    color: #4ade80; margin-bottom: 1rem; opacity: 0.8;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(3rem, 8vw, 6.5rem);
    line-height: 0.95; color: #f0fdf4;
    letter-spacing: -0.02em; margin-bottom: 0.5rem;
}
.hero-title em { font-style: italic; color: #4ade80; }
.hero-sub {
    font-size: 1rem; color: #6b8f72; max-width: 480px;
    margin: 1.5rem auto 0; line-height: 1.6; font-weight: 300;
}
.divider { border: none; border-top: 1px solid #1a2e1d; margin: 2rem 0; }

.upload-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #4ade80; margin-bottom: 0.5rem; display: block;
}
[data-testid="stFileUploader"] {
    border: 1px solid #1e3a22 !important; border-radius: 12px !important;
    background: #0c1a0e !important; padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: #4ade80 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 2px dashed #1e3a22 !important;
    border-radius: 8px !important; min-height: 140px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #6b8f72 !important; }

.result-card {
    background: #0c1a0e; border: 1px solid #1e3a22;
    border-radius: 16px; padding: 2rem; margin-top: 1.5rem;
}
.result-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.8rem; border-radius: 999px;
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.15em; font-weight: 500; margin-bottom: 1rem;
}
.result-disease {
    font-family: 'DM Serif Display', serif; font-size: 2.2rem;
    line-height: 1.1; color: #f0fdf4; margin-bottom: 0.2rem;
}
.result-plant {
    font-family: 'DM Mono', monospace; font-size: 0.75rem;
    color: #6b8f72; letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.confidence-bar-wrap { margin: 1.5rem 0; }
.confidence-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: #6b8f72; letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.confidence-bar {
    height: 4px; background: #1e3a22; border-radius: 2px;
    overflow: hidden; margin-bottom: 0.25rem;
}
.confidence-fill {
    height: 100%; background: linear-gradient(90deg, #4ade80, #22c55e);
    border-radius: 2px;
}
.confidence-pct {
    font-family: 'DM Serif Display', serif; font-size: 2.8rem;
    color: #4ade80; line-height: 1;
}
.info-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1rem; margin-top: 1.5rem;
}
.info-block {
    background: #091209; border: 1px solid #1a2e1d;
    border-radius: 10px; padding: 1rem 1.2rem;
}
.info-block-label {
    font-family: 'DM Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #4ade80; margin-bottom: 0.4rem; opacity: 0.7;
}
.info-block-text { font-size: 0.85rem; color: #a3c4a8; line-height: 1.5; font-weight: 300; }

.top5-row {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.6rem 0; border-bottom: 1px solid #1a2e1d;
}
.top5-rank { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: #3d5c42; width: 1.2rem; flex-shrink: 0; }
.top5-name { flex: 1; font-size: 0.8rem; color: #a3c4a8; }
.top5-bar-wrap { width: 80px; height: 3px; background: #1a2e1d; border-radius: 2px; overflow: hidden; }
.top5-bar-fill { height: 100%; background: #4ade80; border-radius: 2px; }
.top5-pct { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #4ade80; width: 3rem; text-align: right; flex-shrink: 0; }

.section-header {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.25em; text-transform: uppercase;
    color: #4ade80; opacity: 0.7; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: #1a2e1d; }

.stat-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 2rem 0; }
.stat-card {
    background: #0c1a0e; border: 1px solid #1e3a22;
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.stat-num { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #4ade80; line-height: 1; margin-bottom: 0.3rem; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: #6b8f72; }

.stButton > button {
    background: #166534 !important; color: #f0fdf4 !important;
    border: 1px solid #4ade80 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.1em !important; padding: 0.6rem 1.5rem !important;
    width: 100% !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: #4ade80 !important; color: #052e16 !important; }

[data-testid="stSpinner"] { color: #4ade80 !important; }
[data-testid="column"] { padding: 0 0.5rem !important; }

.alert-healthy {
    background: #052e16; border: 1px solid #4ade80; border-radius: 10px;
    padding: 1rem 1.2rem; color: #86efac; font-size: 0.85rem; line-height: 1.5; margin-top: 1rem;
}
.alert-warning {
    background: #1c1400; border: 1px solid #f59e0b; border-radius: 10px;
    padding: 1rem 1.2rem; color: #fcd34d; font-size: 0.85rem; line-height: 1.5; margin-top: 1rem;
}
.alert-critical {
    background: #1c0000; border: 1px solid #ef4444; border-radius: 10px;
    padding: 1rem 1.2rem; color: #fca5a5; font-size: 0.85rem; line-height: 1.5; margin-top: 1rem;
}
.model-tag {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: 'DM Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.12em; color: #4ade80; opacity: 0.6;
    margin-bottom: 1rem;
}
.img-frame { border: 1px solid #1e3a22; border-radius: 12px; overflow: hidden; background: #0c1a0e; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #090f0a; }
::-webkit-scrollbar-thumb { background: #1e3a22; border-radius: 3px; }
[data-testid="stSidebar"] { background: #090f0a !important; border-right: 1px solid #1a2e1d !important; }
</style>
""", unsafe_allow_html=True)

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    inject_css()

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">🌿 Plant disease detector</div>
        <div class="hero-title">Leaf<em>Scan</em></div>
        <div class="hero-sub">Upload a plant leaf image and get instant disease diagnosis — supporting 15 conditions across Bell Pepper, Potato, and Tomato crops.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-num">15</div><div class="stat-label">Disease Classes</div></div>
        <div class="stat-card"><div class="stat-num">3</div><div class="stat-label">Plant Species</div></div>
        <div class="stat-card"><div class="stat-num">EfficientNet</div><div class="stat-label">Architecture</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.markdown('<span class="upload-label">Upload Leaf Image</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a leaf image here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            display_img = image.copy()
            display_img.thumbnail((600, 600))
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(display_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("⟶ Analyze Leaf", key="analyze")
        else:
            st.markdown("""
            <div style="border:2px dashed #1e3a22;border-radius:12px;height:280px;
                display:flex;flex-direction:column;align-items:center;justify-content:center;
                color:#3d5c42;text-align:center;padding:2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">🌿</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.1em;">AWAITING SAMPLE</div>
                <div style="font-size:0.75rem;margin-top:0.5rem;opacity:0.6;">JPG · PNG · WEBP</div>
                <div style="font-size:0.7rem;margin-top:0.8rem;opacity:0.4;">Bell Pepper · Potato · Tomato</div>
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False

    with col_right:
        if uploaded and analyze_btn:
            with st.spinner("Analyzing leaf tissue..."):
                model = load_model()
                time.sleep(0.3)
                results = predict(image, model)

            top = results[0]
            sev = top["severity"]
            scfg = SEVERITY_CONFIG[sev]
            conf = top["confidence"]
            info = DISEASE_INFO.get(top["disease"], DISEASE_INFO["Healthy"])
            src = st.session_state.get("model_source", "imagenet-pretrained")

            badge_style = f"background:{scfg['bg']};color:{scfg['color']};border:1px solid {scfg['color']}33;"
            st.markdown(f"""
            <div class="result-card">
                <div class="model-tag">⬡ {src}</div>
                <div class="result-badge" style="{badge_style}">{scfg['icon']} {scfg['label']}</div>
                <div class="result-disease">{top['emoji']} {top['disease']}</div>
                <div class="result-plant">{top['plant']} · Confidence</div>
                <div class="confidence-bar-wrap">
                    <div class="confidence-label">Model Confidence</div>
                    <div class="confidence-pct">{conf*100:.1f}%</div>
                    <div class="confidence-bar" style="margin-top:0.5rem">
                        <div class="confidence-fill" style="width:{conf*100:.1f}%"></div>
                    </div>
                </div>
                <div class="info-grid">
                    <div class="info-block">
                        <div class="info-block-label">Cause</div>
                        <div class="info-block-text">{info['cause']}</div>
                    </div>
                    <div class="info-block">
                        <div class="info-block-label">Symptoms</div>
                        <div class="info-block-text">{info['symptoms']}</div>
                    </div>
                    <div class="info-block">
                        <div class="info-block-label">Treatment</div>
                        <div class="info-block-text">{info['treatment']}</div>
                    </div>
                    <div class="info-block">
                        <div class="info-block-label">Prevention</div>
                        <div class="info-block-text">{info['prevention']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if sev == "none":
                st.markdown('<div class="alert-healthy">✓ No disease detected. This leaf appears healthy. Continue regular monitoring and good agricultural practices.</div>', unsafe_allow_html=True)
            elif sev in ("high", "critical"):
                st.markdown(f'<div class="alert-critical">⚠ {scfg["label"]} risk detected. Immediate action recommended to prevent crop loss. Consult a local agricultural extension office.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-warning">⚠ Moderate disease pressure detected. Monitor closely and consider preventive treatment within 7–14 days.</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Top Predictions</div>', unsafe_allow_html=True)

            top5_html = ""
            for i, r in enumerate(results):
                pct = r["confidence"] * 100
                top5_html += f"""
                <div class="top5-row">
                    <div class="top5-rank">#{i+1}</div>
                    <div class="top5-name">{r['emoji']} {r['label']}</div>
                    <div class="top5-bar-wrap"><div class="top5-bar-fill" style="width:{pct:.1f}%"></div></div>
                    <div class="top5-pct">{pct:.1f}%</div>
                </div>"""
            st.markdown(top5_html, unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div style="background:#0c1a0e;border:1px solid #1e3a22;border-radius:16px;
                padding:3rem 2rem;text-align:center;min-height:300px;
                display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <div style="font-size:3.5rem;margin-bottom:1.5rem;opacity:0.3;">⊕</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#3d5c42;text-transform:uppercase;">Results appear here</div>
                <div style="font-size:0.8rem;color:#3d5c42;margin-top:0.5rem;font-weight:300;">Upload an image and click analyze</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.2em;color:#3d5c42;text-transform:uppercase;">
            LeafScan · EfficientNet-B0 · 15 Classes · Bell Pepper · Potato · Tomato
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
