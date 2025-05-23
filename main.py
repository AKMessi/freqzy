import streamlit as st
import numpy as np
from scipy.io.wavfile import write
import tempfile

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Freqzy – Binaural Beats", layout="centered")

# ─── Google Analytics ───────────────────────────────────────────────────────────
GA_CODE = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-NYRE8H5PP7"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-NYRE8H5PP7');
</script>
"""

st.markdown(GA_CODE, unsafe_allow_html=True)

st.markdown("""
    <style>
        body {
            background-color: white;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            color: #222;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Freqzy – Beats & Noise Mixer")

# ─── Presets ───────────────────────────────────────────────────────────────────
PRESETS = {
    "Study (Beta – 15 Hz)": 15,
    "Hyper Focus (Gamma – 40 Hz)": 40,
    "Relax (Alpha – 10 Hz)": 10,
    "Meditate (Theta – 6 Hz)": 6,
    "Sleep (Delta – 2 Hz)": 2,
    "Tinnitus Relief (White Noise Only)": None
}

NOISE_DEFAULTS = {
    "Study (Beta – 15 Hz)": "Pink",
    "Hyper Focus (Gamma – 40 Hz)": "Pink",
    "Relax (Alpha – 10 Hz)": "Brown",
    "Meditate (Theta – 6 Hz)": "Brown",
    "Sleep (Delta – 2 Hz)": "Brown",
    "Tinnitus Relief (White Noise Only)": "White"
}

# ─── Noise Generator ───────────────────────────────────────────────────────────
def generate_noise(noise_type: str, size: int) -> np.ndarray:
    if noise_type == "White":
        noise = np.random.normal(0, 1, size=size)
    elif noise_type == "Pink":
        N = 16
        rows = np.random.randn(N, size)
        noise = np.sum(rows, axis=0)
    elif noise_type == "Brown":
        wn = np.random.normal(0, 1, size=size)
        noise = np.cumsum(wn)
    else:
        return np.zeros(size)
    return noise / np.max(np.abs(noise))

# ─── UI Controls ───────────────────────────────────────────────────────────────
options = list(PRESETS.keys()) + ["Custom"]
choice = st.selectbox("Session type", options)

if choice == "Custom":
    beat_freq = st.slider("Custom beat frequency (Hz)", 1, 100, 10)
    st.caption("Use headphones to experience true binaural effect.")
else:
    beat_freq = PRESETS[choice]

default_noise = NOISE_DEFAULTS.get(choice, "None")
noise_type = st.selectbox(
    "Background noise",
    ["None", "White", "Pink", "Brown"],
    index=["None", "White", "Pink", "Brown"].index(default_noise)
)

duration = st.slider("Duration (seconds)", 5, 600, 300, 5)

# ─── Audio Generation ──────────────────────────────────────────────────────────
sr = 44100
carrier = 440  # Hz

if st.button("Generate & Play"):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if choice == "Tinnitus Relief (White Noise Only)":
        data = None
    else:
        fL = carrier - beat_freq / 2
        fR = carrier + beat_freq / 2
        left = 0.5 * np.sin(2 * np.pi * fL * t)
        right = 0.5 * np.sin(2 * np.pi * fR * t)
        data = np.stack([left, right], axis=1)
        st.info("🎧 Use headphones for full binaural effect.")

    noise = generate_noise(noise_type, t.shape[0])
    if data is None:
        mix = np.stack([noise, noise], axis=1)
    else:
        mix = data.copy()
        mix[:, 0] += noise * 0.2
        mix[:, 1] += noise * 0.2
        mix = mix / np.max(np.abs(mix))

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(tmp.name, sr, (mix * 32767).astype(np.int16))

    desc = choice if choice != "Custom" else f"Custom – {beat_freq} Hz"
    st.success(f"Playing: {desc} | Noise: {noise_type}")
    st.audio(tmp.name, format="audio/wav")
