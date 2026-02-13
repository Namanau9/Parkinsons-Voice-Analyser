# app.py  -- ALWAYS uses heavy extractor (feature_extractor.py)
import os
import time
import json
import tempfile
import traceback
import hashlib
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

import joblib
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"
CACHE_FOLDER = os.path.join(MODEL_FOLDER, "cache")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

ALLOWED_EXT = {"wav", "mp3", "flac", "ogg", "webm", "m4a"}
TRAIN_CSV_PATH = "parkinsons.data"   # your CSV for mdvp medians

# Project-local ffmpeg if present (./ffmpeg/bin/ffmpeg[.exe])
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_FFMPEG_BIN = os.path.join(PROJECT_ROOT, "ffmpeg", "bin")
if os.path.isdir(PROJECT_FFMPEG_BIN):
    os.environ["PATH"] = PROJECT_FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
FFMPEG_EXEC = (os.path.join(PROJECT_FFMPEG_BIN, "ffmpeg.exe") if os.name == "nt"
               else os.path.join(PROJECT_FFMPEG_BIN, "ffmpeg"))
if not os.path.exists(FFMPEG_EXEC):
    FFMPEG_EXEC = "ffmpeg"  # fallback to system path

# ------------------------------------------------------------------
# Heavy extractor import (required)
# ------------------------------------------------------------------
try:
    from feature_extractor import extract_features as heavy_extract
except Exception as e:
    heavy_extract = None
    print("Warning: heavy extractor (feature_extractor.py) not importable:", e)
    traceback.print_exc()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def transcode_to_wav(input_path):
    """Transcode input audio to 22050Hz mono WAV with ffmpeg; return path."""
    fd, tmpwav = tempfile.mkstemp(suffix=".wav", prefix="conv_", dir=UPLOAD_FOLDER)
    os.close(fd)
    cmd = [FFMPEG_EXEC, "-y", "-i", input_path, "-ar", "22050", "-ac", "1", tmpwav]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmpwav
    except FileNotFoundError as e:
        raise FileNotFoundError("ffmpeg executable not found. Place ffmpeg in project/ffmpeg/bin or install system ffmpeg.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg failed to transcode uploaded audio: " + str(e)) from e

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_for_json(obj):
    """Convert numpy / pandas scalars and NaN to JSON-friendly values."""
    import math
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    try:
        return float(obj)
    except Exception:
        return str(obj)

# ------------------------------------------------------------------
# Models loading (optional)
# ------------------------------------------------------------------
audio_model = None
audio_feature_names = None
mdvp_model = None
mdvp_feature_names = None

AUDIO_MODEL_PATH = os.path.join(MODEL_FOLDER, "parkinson_audio_model.joblib")
AUDIO_FEAT_PATH = os.path.join(MODEL_FOLDER, "parkinson_audio_feature_names.pkl")
MDVP_MODEL_PATH = os.path.join(MODEL_FOLDER, "parkinson_mdvp_model.joblib")
MDVP_FEAT_PATH = os.path.join(MODEL_FOLDER, "parkinson_mdvp_feature_names.pkl")

def load_models():
    global audio_model, audio_feature_names, mdvp_model, mdvp_feature_names
    try:
        if os.path.exists(AUDIO_MODEL_PATH):
            audio_model = joblib.load(AUDIO_MODEL_PATH)
            if os.path.exists(AUDIO_FEAT_PATH):
                audio_feature_names = joblib.load(AUDIO_FEAT_PATH)
            print("Loaded audio model:", AUDIO_MODEL_PATH)
        if os.path.exists(MDVP_MODEL_PATH):
            mdvp_model = joblib.load(MDVP_MODEL_PATH)
            if os.path.exists(MDVP_FEAT_PATH):
                mdvp_feature_names = joblib.load(MDVP_FEAT_PATH)
            print("Loaded mdvp model:", MDVP_MODEL_PATH)
    except Exception as e:
        print("Model load error:", e)
        traceback.print_exc()

load_models()

_mdvp_medians_cache = None
def get_mdvp_medians():
    global _mdvp_medians_cache
    if _mdvp_medians_cache is not None:
        return _mdvp_medians_cache
    try:
        if os.path.exists(TRAIN_CSV_PATH):
            df = pd.read_csv(TRAIN_CSV_PATH)
            for c in ("name","status"):
                if c in df.columns:
                    df = df.drop(columns=[c])
            _mdvp_medians_cache = df.median()
            return _mdvp_medians_cache
    except Exception as e:
        print("Could not load medians from CSV:", e)
    _mdvp_medians_cache = pd.Series(dtype=float)
    return _mdvp_medians_cache

# ------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:p>")
def static_files(p):
    return send_from_directory("static", p)

@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    # remove all cache files
    try:
        cnt = 0
        for f in os.listdir(CACHE_FOLDER):
            os.remove(os.path.join(CACHE_FOLDER, f))
            cnt += 1
        return jsonify({"cleared": cnt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        # ensure heavy extractor exists
        if heavy_extract is None:
            return jsonify({"error": "no_heavy_extractor", "message": "feature_extractor.py not importable on server"}), 500

        if "audio" not in request.files:
            return jsonify({"error":"no_file","message":"Missing 'audio' file field"}), 400

        f = request.files["audio"]
        if f.filename == "":
            return jsonify({"error":"no_filename","message":"Empty filename"}), 400

        if not allowed_file(f.filename):
            return jsonify({"error":"unsupported_format","allowed": list(ALLOWED_EXT)}), 400

        filename = secure_filename(f.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(save_path)

        # If not WAV, transcode to WAV for heavy extractor (heavy expects wav)
        ext = filename.rsplit(".", 1)[1].lower()
        wav_path = save_path
        if ext != "wav":
            try:
                wav_path = transcode_to_wav(save_path)
            except FileNotFoundError as e:
                return jsonify({"error":"ffmpeg_missing","message": str(e)}), 500
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error":"ffmpeg_error","message": str(e)}), 500

        # We purposely DO NOT use cached JSON here (cache disabled to avoid stale mdvp outputs).
        # If you want caching, compute file hash and store/lookup in model/cache/<hash>.json

        # Run heavy extractor (this is the key change: ALWAYS call heavy extractor)
        t0 = time.time()
        try:
            feats = heavy_extract(wav_path)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error":"heavy_extraction_failed","message": str(e)}), 500
        t1 = time.time()
        extraction_time = t1 - t0

        # Convert features to JSON-friendly form
        clean_feats = clean_for_json(feats)

        # 1) If an audio model is available, prefer it (uses MFCC/spectral names)
        if audio_model is not None and audio_feature_names is not None:
            try:
                row = {fn: clean_feats.get(fn, None) for fn in audio_feature_names}
                df_row = pd.DataFrame([row]).astype(float)
                df_row = df_row.fillna(df_row.median()).fillna(0.0)
                pred = int(audio_model.predict(df_row)[0])
                proba = float(audio_model.predict_proba(df_row)[0][1]) if hasattr(audio_model, "predict_proba") else None
                resp = {"source":"audio_model","prediction":pred,"probability":proba,"features": clean_for_json(row), "meta":{"extraction_time_s": extraction_time}}
                resp["meta"]["total_time_s"] = time.time() - start_time
                return jsonify(resp)
            except Exception:
                traceback.print_exc()
                # fall back to mdvp next

        # 2) If mdvp model present use median imputation for missing MDVP features
        if mdvp_model is not None and mdvp_feature_names is not None:
            try:
                medians = get_mdvp_medians()
                row = {}
                for fn in mdvp_feature_names:
                    v = feats.get(fn) if fn in feats else None
                    if v is None:
                        row[fn] = float(medians.get(fn, 0.0)) if fn in medians.index else 0.0
                    else:
                        row[fn] = v
                df_row = pd.DataFrame([row]).astype(float)
                pred = int(mdvp_model.predict(df_row)[0])
                proba = float(mdvp_model.predict_proba(df_row)[0][1]) if hasattr(mdvp_model,"predict_proba") else None
                resp = {"source":"mdvp_model","prediction":pred,"probability":proba,"features": clean_for_json(row), "meta":{"extraction_time_s": extraction_time}}
                resp["meta"]["total_time_s"] = time.time() - start_time
                return jsonify(resp)
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error":"mdvp_prediction_failed","message": str(e)}), 500

        # No model found
        return jsonify({"error":"no_model","message":"No prediction model found (place a joblib in model/)","features": clean_feats}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"internal_server_error","message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
