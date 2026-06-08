
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time

# config
CLASS_NAMES = ["Speed Limit 30", "Speed Limit 50", "Speed Limit 80", "Yield", "Stop"]
NUM_CLASSES  = len(CLASS_NAMES)
IMAGE_SIZE   = 48
MODEL_PATH   = "./checkpoints/best.pth"
CONF_DEFAULT = 0.4

# arhitectura cnn pentru recunoasterea semnelor
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=False),
            ConvBlock(128, 128, pool=True),
        )
        self.neck = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))


@st.cache_resource
def load_model():
    state = torch.load(MODEL_PATH, map_location="cpu")
    if "model_state" in state:
        state = state["model_state"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model = TrafficSignCNN()
    model.load_state_dict(state)
    model.eval()
    return model


# preprocesare
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.3337, 0.3064, 0.3171],
                         [0.2672, 0.2564, 0.2629]),
])


def predict(model, frame_bgr):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_prob, top_idx = probs.max(0)
    return CLASS_NAMES[top_idx.item()], top_prob.item(), probs.numpy()


# config pagina si css
st.set_page_config(page_title="Street Sign Detector", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0d0d0f; color: #f0ede8; }
.pred-card {
    background: #17171a; border: 1px solid #2a2a2f;
    border-radius: 16px; padding: 28px 32px; margin-bottom: 16px;
}
.pred-label {
    font-family: 'Syne', sans-serif; font-size: 2.2rem;
    font-weight: 800; letter-spacing: -0.5px; color: #f0ede8; margin: 0;
}
.pred-conf { font-size: 0.95rem; color: #888; margin-top: 4px; }
.bar-row { margin: 6px 0; }
.bar-label { font-size: 0.82rem; color: #aaa; margin-bottom: 2px; }
.bar-outer { background: #222226; border-radius: 6px; height: 10px; overflow: hidden; }
.bar-inner  { height: 10px; border-radius: 6px; }
.pill { display:inline-block; padding:4px 14px; border-radius:999px;
        font-size:0.78rem; font-weight:500; letter-spacing:0.4px; }
.pill-live { background:#1a3a1a; color:#4caf50; border:1px solid #2e5e2e; }
.pill-off  { background:#2a1a1a; color:#e55;    border:1px solid #5e2e2e; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# incarcare model
try:
    model = load_model()
except Exception as e:
    st.error(f"❌Could not load model from `{MODEL_PATH}`:\n\n`{e}`")
    st.stop()

# sidebar
with st.sidebar:
    st.markdown("## Traffic Sign Detector")
    st.markdown("---")
    st.markdown("**Classes**")
    for name in CLASS_NAMES:
        st.markdown(f"{name}")
    st.markdown("---")
    run_camera     = st.toggle("Enable Camera", value=False)
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.99, CONF_DEFAULT, 0.05)
    st.markdown("---")
    st.caption("TrafficSignCNN · model loaded")

# pagina principala
st.markdown("# Street Sign Recognition")
st.markdown("Point your camera at a **speed limit 30 / 50 / 80**, **yield**, or **stop** sign.")

col_cam, col_pred = st.columns([3, 2], gap="large")
with col_cam:
    status_ph = st.empty()
    camera_ph = st.empty()
with col_pred:
    pred_ph = st.empty()
    bars_ph = st.empty()

# helper pentru bara de procent
def bar_html(name, prob, is_top):
    pct   = int(prob * 100)
    color = "#e8c547" if is_top else "#3a3a45"
    return (
        f'<div class="bar-row">'
        f'  <div class="bar-label">{name} - {pct}%</div>'
        f'  <div class="bar-outer"><div class="bar-inner" style="width:{pct}%;background:{color};"></div></div>'
        f'</div>'
    )

# logica camerei
if run_camera:
    status_ph.markdown('<span class="pill pill-live">LIVE</span>', unsafe_allow_html=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Make sure it is connected and not used by another app.")
        st.stop()

    try:
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            camera_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True)

            label, conf, probs = predict(model, frame)

            if conf >= conf_threshold:
                pred_ph.markdown(
                    f'<div class="pred-card">'
                    f'  <p class="pred-label">{label}</p>'
                    f'  <p class="pred-conf">Confidence: {conf*100:.1f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                pred_ph.markdown(
                    '<div class="pred-card">'
                    '  <p class="pred-label">Uncertain</p>'
                    '  <p class="pred-conf">No sign detected with sufficient confidence</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            bars_ph.markdown(
                "".join(bar_html(CLASS_NAMES[i], probs[i],
                                 CLASS_NAMES[i] == label and conf >= conf_threshold)
                        for i in range(NUM_CLASSES)),
                unsafe_allow_html=True,
            )
            time.sleep(0.05)
    finally:
        cap.release()

else:
    status_ph.markdown('<span class="pill pill-off">Camera off</span>', unsafe_allow_html=True)
    camera_ph.markdown(
        '<div style="background:#17171a;border:1px dashed #333;border-radius:12px;'
        'height:320px;display:flex;align-items:center;justify-content:center;'
        'color:#555;font-size:1rem;">Enable camera in the sidebar to start</div>',
        unsafe_allow_html=True,
    )
    pred_ph.markdown(
        '<div class="pred-card">'
        '  <p class="pred-label" style="color:#444;">— —</p>'
        '  <p class="pred-conf">Waiting for camera...</p>'
        '</div>',
        unsafe_allow_html=True,
    )