# ================================================================
# SILENT CLASSROOM - Colab backend (Flask + Ngrok)
# Paste into ONE Colab cell AFTER uploading main_pipeline.py and index.html
# (left sidebar > Files > upload, or run: from google.colab import files; files.upload())
#
# 1. Put your Ngrok authtoken below (dashboard.ngrok.com)
# 2. Run the cell, copy the printed https://xxxx.ngrok-free.app URL
# 3. Open that URL: it serves the UI itself (upload index.html too). No pasting needed.
# 4. Keep this cell running
# ================================================================
NGROK_TOKEN = "PASTE_YOUR_NGROK_AUTHTOKEN_HERE"

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "flask", "flask-cors", "pyngrok", "ultralytics", "mediapipe",
                       "deepface", "tf-keras", "opencv-python-headless", "matplotlib", "kagglehub"])

import base64
import io
import os
import threading

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pyngrok import ngrok

from main_pipeline import ClassroomPipeline

app = Flask(__name__)
CORS(app)

print("Loading models (first run downloads weights)...")
pipeline = ClassroomPipeline()
lock = threading.Lock()          # one frame at a time through the shared models
eda_cache = {}


def decode_image(data_url):
    raw = base64.b64decode(data_url.split(",")[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def encode_jpg(bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def home():
    """Serve the UI from the same public URL (upload index.html next to this cell)."""
    if not os.path.exists("index.html"):
        return "Upload index.html to /content to serve the UI here.", 404
    return send_file("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    """Body: {"image": data URL, "live": bool}. live=True uses frame skipping + caching for speed."""
    body = request.get_json(force=True, silent=True) or {}
    if not body.get("image"):
        return jsonify(error="Send JSON with an 'image' field."), 400
    frame = decode_image(body["image"])
    if frame is None:
        return jsonify(error="Could not decode image."), 400
    h, w = frame.shape[:2]
    if max(h, w) > 1280:                      # keep latency low for big uploads
        s = 1280 / max(h, w)
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    with lock:
        annotated, report = pipeline.process_frame(frame, force=not body.get("live"))
    return jsonify(image=encode_jpg(annotated), report=report)


def build_eda():
    """Charts taken from the project notebook (FER2013 + model comparison)."""
    import kagglehub
    path = kagglehub.dataset_download("msambare/fer2013")
    counts, sizes = {}, []
    for split in ("train", "test"):
        for emo in sorted(os.listdir(os.path.join(path, split))):
            d = os.path.join(path, split, emo)
            if os.path.isdir(d):
                files_ = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                counts[(emo, split)] = len(files_)

    emotions = sorted({e for e, _ in counts})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(emotions))
    ax.bar(x - 0.2, [counts.get((e, "train"), 0) for e in emotions], 0.4, label="train")
    ax.bar(x + 0.2, [counts.get((e, "test"), 0) for e in emotions], 0.4, label="test")
    ax.set_xticks(x, emotions)
    ax.set_title("FER2013 emotion class counts (train vs test)")
    ax.set_ylabel("images")
    ax.legend()
    chart1 = fig_to_b64(fig)

    # brightness vs contrast sample, as in the notebook
    rng = np.random.default_rng(42)
    all_files = [(e, os.path.join(path, s, e, f)) for (e, s) in counts
                 for f in rng.choice(os.listdir(os.path.join(path, s, e)), min(25, counts[(e, s)]), replace=False)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in emotions:
        pts = []
        for emo, p in all_files:
            if emo == e:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    pts.append((img.mean(), img.std()))
        if pts:
            ax.scatter(*zip(*pts), s=14, alpha=0.7, label=e)
    ax.set_xlabel("mean brightness")
    ax.set_ylabel("contrast (std)")
    ax.set_title("Brightness vs contrast by emotion (sample)")
    ax.legend(fontsize=7)
    chart2 = fig_to_b64(fig)

    # accuracies printed by the notebook's baseline comparison
    models = {"Logistic Regression": 32.50, "Random Forest": 32.50,
              "Support Vector Machine": 34.25, "Decision Tree": 22.75}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models.keys(), models.values(), color="#3b82f6")
    ax.bar_label(bars, fmt="%.2f%%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Baseline classifier comparison (FER2013 subset)")
    plt.setp(ax.get_xticklabels(), rotation=15)
    chart3 = fig_to_b64(fig)

    return [{"title": "Emotion class counts", "image": chart1},
            {"title": "Brightness vs contrast", "image": chart2},
            {"title": "Baseline model accuracy", "image": chart3}]


@app.get("/get_eda")
def get_eda():
    if "charts" not in eda_cache:
        eda_cache["charts"] = build_eda()
    return jsonify(charts=eda_cache["charts"])


if __name__ == "__main__" or "google.colab" in sys.modules:
    try:
        ngrok.kill()                      # close tunnels left from an earlier run
    except Exception:
        pass
    try:
        ngrok.set_auth_token(NGROK_TOKEN)
    except Exception as e:                # "already set" is harmless on re-runs
        print("ngrok token note:", e)
    public_url = ngrok.connect(5000).public_url
    print("\n" + "=" * 60)
    print("PUBLIC URL (paste into the UI):", public_url)
    print("=" * 60 + "\n")
    app.run(port=5000)
