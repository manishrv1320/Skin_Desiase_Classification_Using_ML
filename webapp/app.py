import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

import model_card
from classifier import InvalidImage, SkinClassifier
from conditions import CONDITIONS, describe

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = os.environ.get("MODEL_PATH", MODEL_DIR / "scin_skin_condition_classifier.keras")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", MODEL_DIR / "scin_class_names.json")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TOP_K = 5
LOW_CONFIDENCE = 0.40

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

classifier = SkinClassifier(MODEL_PATH, CLASS_NAMES_PATH)
if classifier.ready:
    print(f"Loaded model with {len(classifier.class_names)} classes from {MODEL_PATH}")
else:
    print(f"WARNING: {classifier.error}")


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_ready=classifier.ready,
        model_error=classifier.error,
        max_upload_bytes=MAX_UPLOAD_BYTES,
        low_confidence=LOW_CONFIDENCE,
        top_k=TOP_K,
    )


@app.get("/about")
def about():
    names = classifier.class_names or sorted(CONDITIONS)
    return render_template(
        "about.html",
        conditions=[(name, describe(name)) for name in names],
        card=model_card.load(),
    )


@app.post("/api/predict")
def predict():
    if not classifier.ready:
        return jsonify(error="The model is not loaded on the server."), 503

    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="No image was attached."), 400

    try:
        results = classifier.predict(upload.read(), top_k=TOP_K)
    except InvalidImage as exc:
        return jsonify(error=str(exc)), 400

    predictions = [
        {"label": label, "probability": round(prob, 4), **describe(label)}
        for label, prob in results
    ]
    return jsonify(
        predictions=predictions,
        low_confidence=results[0][1] < LOW_CONFIDENCE,
    )


@app.get("/api/health")
def health():
    return jsonify(
        model_loaded=classifier.ready,
        classes=len(classifier.class_names),
        error=classifier.error,
    )


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify(error=f"The file is larger than {limit_mb} MB."), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
