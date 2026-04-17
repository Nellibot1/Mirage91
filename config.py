# config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ─── STREAM SETTINGS ────────────────────────────────────────────
STREAM_NAME = "X.on-103502-0170"
STREAM_TYPE = "EEG"
# STREAM_NAME should match your amplifier LSL stream when connecting by name.
# STREAM_TYPE is used as a fallback if STREAM_NAME is left empty.

SFREQ = 250
# Nominal sampling rate. The live stream value is treated as source-of-truth
# when available, but this is still used for tests and some defaults.

# ─── CHANNELS ───────────────────────────────────────────────────
# Classifier v4.2 was trained on these channels in this exact order.
TARGET_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4", "BIP", "accX", "accY", "accZ"]
FEATURE_EEG_CHANNELS = ["F3", "F4", "BIP"]
ACC_CHANNELS = ["accX", "accY", "accZ"]
N_CHANNELS = len(TARGET_CHANNELS)

# ─── EPOCH / WINDOW SETTINGS ────────────────────────────────────
# These should stay aligned with the v4.2 training metadata.
EPOCH_LENGTH = 0.5
SLIDE_STEP = 0.1

# ─── PREPROCESSING SETTINGS ─────────────────────────────────────
L_FREQ = 0.5
H_FREQ = 100.0
USE_NOTCH = True
NOTCH_FREQ = 50.0

# ─── MODEL FILE PATHS ───────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "eeg_artifact_game_control_v4_2.joblib"
METADATA_PATH = MODELS_DIR / "eeg_artifact_game_control_v4_2_metadata.json"

# ─── CLASS LABELS ───────────────────────────────────────────────
CLASS_LABELS = {
    0: "REST",
    1: "BLINK",
    2: "CLENCH",
}

# ─── COMMAND SETTINGS ───────────────────────────────────────────
COMMAND_HOST = "127.0.0.1"
COMMAND_PORT = 5005
BLINK_COMMAND = "DUCK"
CLENCH_COMMAND = "ACTION"

# ─── SMOOTHING / EVENT DETECTION ────────────────────────────────
SMOOTHING_WINDOW = 5
MIN_BURST_WINDOWS = 2
DOUBLE_BLINK_MIN_GAP_SEC = 0.1
DOUBLE_BLINK_MAX_GAP_SEC = 0.6
COMMAND_COOLDOWN_SEC = 0.75
