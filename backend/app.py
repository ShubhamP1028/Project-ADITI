"""
Endpoints
---------
POST /api/predict
    Body   : multipart/form-data with field 'image' (JPG / PNG / WebP, max 20 MB)
    Returns: JSON { disease, confidence, isHealthy, description, treatment }
GET /api/health
    Returns: {"status": "ok", "model_loaded": true}
GET  /  (and all other paths)
    Returns the React SPA — index.html is served for all non-API routes.
"""

import io
import os
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from labels import CLASS_NAMES, DISEASE_INFO
from model import PlantDiseaseSNN, RS_PARAMS

# ─────────────────────────── Configuration ──────────────────────────────────
IMG_SIZE = 128           # must match training resize (trained on 128×128)

# Resolve paths relative to this file so the app works no matter which
# directory you launch it from (e.g. `cd backend && python app.py` or
# `python backend/app.py` from the project root).
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(_PROJECT_ROOT / "regular_spiking_model.pt")
)
MAX_CONTENT_LENGTH = 20 * 1024 * 1024   # 20 MB
FRONTEND_DIST = os.environ.get(
    "FRONTEND_DIST",
    str(_PROJECT_ROOT / "frontend" / "dist")
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────── Logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aditi-backend")

# ─────────────────────────── Flask app ──────────────────────────────────────
# Point Flask's static folder at the React build output
app = Flask(
    __name__,
    static_folder=FRONTEND_DIST,
    static_url_path="",   # serve static files from root /
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ─────────────────────────── Image transform ─────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# ─────────────────────────── Model loading ───────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

model = None
model_loaded = False

def load_model():
    global model, model_loaded
    try:
        logger.info("Loading model from: %s", MODEL_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            # Current .pt is a state_dict — reconstruct architecture and load weights
            m = PlantDiseaseSNN(
                neuron_params=RS_PARAMS,
                num_classes=len(CLASS_NAMES),
                feature_dim=512,
                snn_hidden=256,
                T=8,
                name="Regular_Spiking",
            )
            m.load_state_dict(checkpoint)
            model = m.to(device)
        else:
            # Full model object (after replacement)
            model = checkpoint.to(device)

        model.eval()
        model_loaded = True
        logger.info("Model loaded successfully — %d output classes", len(CLASS_NAMES))
    except FileNotFoundError:
        logger.error("Model file not found at '%s'. Set MODEL_PATH env var.", MODEL_PATH)
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)


# ─────────────────────────── Inference helper ────────────────────────────────
def run_inference(image_bytes: bytes) -> dict:
    """
    Accept raw image bytes, run the SNN model, and return a result dict.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        logits = model(tensor)

    # Support both plain logits and tuple outputs (some SNN wrappers return (spikes, logits))
    if isinstance(logits, (tuple, list)):
        logits = logits[-1]

    # Average over time-step dimension if present [T, B, C] → [B, C]
    if logits.dim() == 3:
        logits = logits.mean(dim=0)

    probs = F.softmax(logits, dim=-1)
    confidence_tensor, pred_idx = torch.max(probs, dim=-1)
    confidence = round(float(confidence_tensor.item()) * 100, 1)
    class_key = CLASS_NAMES[int(pred_idx.item())]

    info = DISEASE_INFO.get(class_key, {
        "display": class_key.replace("___", " — ").replace("_", " "),
        "isHealthy": "healthy" in class_key.lower(),
        "description": "No additional information available for this class.",
        "treatment": ["Consult a local agricultural extension officer for advice."],
    })

    return {
        "disease":     info["display"],
        "confidence":  confidence,
        "isHealthy":   info["isHealthy"],
        "description": info["description"],
        "treatment":   info["treatment"],
    }


# ─────────────────────────── Routes ──────────────────────────────────────────

# ── Frontend SPA catch-all ───────────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    """
    Serve React static files. For any path that is not an actual file
    (e.g. /workflow, /about, /diseases) return index.html so React Router
    can handle client-side navigation.
    """
    dist = Path(FRONTEND_DIST)
    target = dist / path
    if path and target.exists():
        return send_from_directory(FRONTEND_DIST, path)
    index = dist / "index.html"
    if index.exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return (
        "<h2>Frontend not built.</h2>"
        "<p>Run <code>cd frontend &amp;&amp; npm run build</code> first, "
        "then restart app.py.</p>",
        404,
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model_loaded, "device": str(device)}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send a multipart/form-data request with field 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename received."}), 400

    allowed_mime = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
    if file.content_type and file.content_type not in allowed_mime:
        return jsonify({"error": f"Unsupported file type: {file.content_type}"}), 415

    try:
        image_bytes = file.read()
        result = run_inference(image_bytes)
        logger.info(
            "Prediction: %s (%.1f%%) | healthy=%s",
            result["disease"], result["confidence"], result["isHealthy"]
        )
        return jsonify(result), 200

    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        return jsonify({"error": "Inference failed. Please try a different image."}), 500


# ─────────────────────────── Entry point ─────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    # Loaded by gunicorn
    load_model()
