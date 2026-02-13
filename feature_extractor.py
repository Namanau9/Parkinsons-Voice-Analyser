# feature_extractor.py
# Robust heavy feature extractor using parselmouth (Praat) + librosa.
# Normalizes + trims audio, uses a robust pitch fallback procedure,
# and returns MDVP-like features in a flat dict with None for missing values.

import os
import tempfile
import math
import numpy as np

# audio libs
import soundfile as sf
import librosa
import parselmouth

# small helper utilities

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, float)):
            if math.isnan(float(x)):
                return None
            return float(x)
        return float(x)
    except Exception:
        return None

def normalize_and_trim_to_temp(in_path, sr=22050, top_db=25):
    """
    Load audio, trim silence, normalize peak, save to a temp WAV, return path.
    """
    y, sr = librosa.load(in_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y) < 1024:
        y = np.pad(y, (0, max(0, 1024 - len(y))), 'constant')
    maxa = np.max(np.abs(y)) + 1e-12
    if maxa > 0:
        y = y / maxa
    fd, outpath = tempfile.mkstemp(suffix=".wav", prefix="norm_")
    os.close(fd)
    sf.write(outpath, y, sr)
    return outpath

def robust_pitch(snd, attempts=None):
    """
    Try multiple pitch algorithms and ranges to build a decent pitch object.
    Returns: parselmouth.Pitch (may be sparse)
    Attempts is list of (method, fmin, fmax) where method in {"ac","cc","auto"}.
    """
    if attempts is None:
        # order: good defaults, then wider ranged for low voices, then tighter for high
        attempts = [
            ("ac", 75, 600),
            ("ac", 40, 300),
            ("cc", 75, 600),
            ("cc", 40, 300),
            ("auto", 60, 400),
        ]
    best = None
    best_voiced_ratio = -1.0
    for method, fmin, fmax in attempts:
        try:
            if method == "ac":
                pitch = snd.to_pitch_ac(0.01, fmin, fmax)
            elif method == "cc":
                pitch = snd.to_pitch_cc(0.01, fmin, fmax)
            else:
                pitch = snd.to_pitch(0.01, fmin, fmax)
            vals = [pitch.get_value_at_time(t) for t in pitch.xs()]
            valid = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            voiced_ratio = len(valid) / len(vals) if len(vals) else 0.0
            # prefer 0.2+ voiced ratio tracks; otherwise keep best seen
            if voiced_ratio > best_voiced_ratio:
                best_voiced_ratio = voiced_ratio
                best = pitch
            # early accept if voiced_ratio is good
            if voiced_ratio >= 0.35:
                return pitch
        except Exception:
            continue
    return best

def pitch_stats_from_pitch(pitch):
    """
    Given a parselmouth Pitch object, return mean/min/max & voiced ratio
    """
    if pitch is None:
        return {"mean_f0": None, "min_f0": None, "max_f0": None, "voiced_ratio": 0.0}
    times = pitch.xs()
    vals = [pitch.get_value_at_time(t) for t in times]
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) == 0:
        return {"mean_f0": None, "min_f0": None, "max_f0": None, "voiced_ratio": 0.0}
    voiced_ratio = len(valid) / len(vals)
    mean_f0 = float(sum(valid)/len(valid)) if len(valid) else None
    min_f0 = float(min(valid)) if len(valid) else None
    max_f0 = float(max(valid)) if len(valid) else None
    return {"mean_f0": _safe_float(mean_f0), "min_f0": _safe_float(min_f0), "max_f0": _safe_float(max_f0), "voiced_ratio": _safe_float(voiced_ratio)}

# Small wrappers for Praat measures - each returns None on failure

def get_hnr(snd):
    try:
        return _safe_float(parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1).get_mean())
    except Exception:
        try:
            return _safe_float(parselmouth.praat.call(snd, "To Harmonicity (AC)", 0.01, 75, 0.1).get_mean())
        except Exception:
            return None

def get_jitter_shimmer_from_point_process(snd, pitch):
    """
    Use To PointProcess (periodic, pitch) and get jitter/shimmer measures where possible.
    Returns tuple (jitter_local, jitter_local_abs, jitter_rap, shimmer_local, apq3)
    """
    try:
        # need a pitch to create point process
        point_process = parselmouth.praat.call(pitch, "To PointProcess (periodic, cc)", 0.01, 75, 600)
    except Exception:
        try:
            point_process = parselmouth.praat.call(pitch, "To PointProcess (periodic, ac)", 0.01, 75, 600)
        except Exception:
            return (None, None, None, None, None)
    out = (None, None, None, None, None)
    try:
        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_local_abs = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter_local = jitter_local_abs = jitter_rap = None
    try:
        shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3)
        apq3 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        shimmer_local = apq3 = None
    return (_safe_float(jitter_local), _safe_float(jitter_local_abs), _safe_float(jitter_rap), _safe_float(shimmer_local), _safe_float(apq3))

# RPDE/DFA/PPE implementations can be complex and sometimes slow.
# We'll attempt to compute them if the user has the functions present; otherwise return None.
# For RPDE/DFA/PPE we provide lightweight wrappers that call possible helpers if defined
# in a local helper module (user could optionally implement rpde/ppe/dfa there).
def compute_rpde_dfa_ppe(y, sr):
    """
    Try to compute RPDE, DFA, PPE using available implementation.
    If not available, return (None, None, None)
    """
    try:
        # try imported implementations if present (user may have rpde implementations)
        # e.g., a local module 'nolds' or custom rpde/dfa functions
        # We'll attempt basic DFA via nolds if present (not mandatory).
        import importlib
        rpde = None
        dfa = None
        ppe = None
        # attempt nolds for DFA
        try:
            import nolds
            dfa = _safe_float(nolds.dfa(y))
        except Exception:
            dfa = None
        # rpde and ppe usually custom - skip if not present
        return (_safe_float(rpde), _safe_float(dfa), _safe_float(ppe))
    except Exception:
        return (None, None, None)

# Main extractor - returns a flat dict of features
def extract_features(path):
    """
    Heavy extractor. Returns dict of MDVP-like features. Values are floats or None.
    Steps:
     - normalize & trim
     - construct parselmouth.Sound on normalized audio
     - robust_pitch selection
     - compute jitter/shimmer/HNR etc with safe fallbacks
    """
    tmp = None
    try:
        tmp = normalize_and_trim_to_temp(path)
        snd = parselmouth.Sound(tmp)
    except Exception:
        # fallback: try to load original file directly via parselmouth
        try:
            snd = parselmouth.Sound(path)
        except Exception:
            # cannot proceed
            return {}

    # build pitch robustly
    pitch = robust_pitch(snd)
    pstats = pitch_stats_from_pitch(pitch)

    # jitter/shimmer using point process - safe wrappers
    jl, jla, jrap, shimmer_local, apq3 = (None, None, None, None, None)
    try:
        jl, jla, jrap, shimmer_local, apq3 = get_jitter_shimmer_from_point_process(snd, pitch)
    except Exception:
        jl = jla = jrap = shimmer_local = apq3 = (None, None, None, None, None)

    # HNR
    hnr = None
    try:
        hnr = get_hnr(snd)
    except Exception:
        hnr = None

    # RPDE/DFA/PPE attempt (fast) but safe
    rpde, dfa, ppe = compute_rpde_dfa_ppe(snd.values.T.flatten(), snd.sampling_frequency)

    # Some MDVP values like MDVP:Flo, Fhi, Fo can be estimated from pitch stats
    mdvp_fo = pstats.get("mean_f0")
    mdvp_flo = pstats.get("min_f0")
    mdvp_fhi = pstats.get("max_f0")

    # Additional shimmer measures, NHR etc from standard Praat calls (safe)
    try:
        # NHR (Noise-to-Harmonics ratio) via To Harmonicity might be used (practical wrapper)
        nhr = None
        try:
            harm = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1)
            nhr = parselmouth.praat.call(harm, "Get mean", 0, 0)
            # convert typical Harmonicity -> HNR (Praat returns harmonicity values)
            # If direct NHR measure not available, keep HNR above.
        except Exception:
            nhr = None
    except Exception:
        nhr = None

    # Build final dict - names match typical MDVP dataset naming where possible
    feats = {
        "MDVP:Fo(Hz)": _safe_float(mdvp_fo),
        "MDVP:Fhi(Hz)": _safe_float(mdvp_fhi),
        "MDVP:Flo(Hz)": _safe_float(mdvp_flo),
        "mean_f0": _safe_float(pstats.get("mean_f0")),
        "min_f0": _safe_float(pstats.get("min_f0")),
        "max_f0": _safe_float(pstats.get("max_f0")),
        "voiced_ratio": _safe_float(pstats.get("voiced_ratio")),

        "Jitter:DDP": _safe_float(jl),
        "MDVP:Jitter(%)": _safe_float(jrap),
        "MDVP:Jitter(Abs)": _safe_float(jla),
        "MDVP:PPQ": _safe_float(jrap),   # approximate mapping
        "MDVP:RAP": _safe_float(jrap),

        "MDVP:Shimmer": _safe_float(shimmer_local),
        "MDVP:Shimmer(dB)": _safe_float(apq3),
        "Shimmer:APQ3": _safe_float(apq3),
        "Shimmer:APQ5": None,
        "Shimmer:DDA": None,

        "HNR": _safe_float(hnr),
        "NHR": _safe_float(nhr),

        "RPDE": _safe_float(rpde),
        "DFA": _safe_float(dfa),
        "PPE": _safe_float(ppe),

        # spread / spectral spreads - compute via librosa if possible
        "spread1": None,
        "spread2": None,
    }

    # attempt to compute a couple of spectral summaries too (helpful if MDVP missing)
    try:
        import librosa
        y, sr = librosa.load(tmp if tmp is not None else path, sr=22050, mono=True)
        # spectral centroid and bandwidth
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        feats["spec_centroid"] = _safe_float(centroid)
        feats["spec_bandwidth"] = _safe_float(bandwidth)
        # basic energy
        feats["rmse"] = _safe_float(float(np.mean(librosa.feature.rms(y=y))))
    except Exception:
        pass

    # cleanup tmp
    try:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass

    # final conversion: ensure all numeric types are Python floats or None
    for k, v in list(feats.items()):
        feats[k] = _safe_float(v)

    return feats
