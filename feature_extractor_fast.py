# feature_extractor_fast.py
# Fast, low-latency feature extractor for short vowel recordings.
# Returns a flat dict of numeric features (no pandas required).

import librosa
import numpy as np

def extract_features_fast(path, sr=22050, n_mfcc=13):
    """
    Load audio, trim silence, compute MFCC means and basic spectral features.
    Returns dict[str, float].
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    # trim leading/trailing silence to remove long quiet parts
    y, _ = librosa.effects.trim(y, top_db=25)
    # ensure minimum length
    if len(y) < 1024:
        y = np.pad(y, (0, max(0, 1024 - len(y))), 'constant')

    # compute features
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rmse = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    feats = {
        "zcr": zcr,
        "rmse": rmse,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
    }
    for i, v in enumerate(mfcc_mean, start=1):
        feats[f"mfcc_{i}"] = float(v)

    return feats
