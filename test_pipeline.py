# test_pipeline.py

import collections
import traceback
from pathlib import Path

import numpy as np

import config
from classifier import Classifier
from feature_extractor import FeatureExtractor, bandpower
from preprocessor import preprocess, select_channels

PASS = "\033[92m  OK\033[0m"
FAIL = "\033[91m  FAIL\033[0m"


def run_test(name, fn):
    print(f"  {name} ...", end=" ", flush=True)
    try:
        fn()
        print(PASS)
        return True
    except Exception:
        print(FAIL)
        traceback.print_exc()
        return False


N_CH = config.N_CHANNELS
N_SMP = int(config.EPOCH_LENGTH * config.SFREQ)
FAKE = np.random.randn(N_CH, N_SMP) * 50e-6

print()
print("-- 1. Config -----------------------------------------------")


def test_config_channels():
    assert config.N_CHANNELS == 11, f"Expected 11 channels, got {config.N_CHANNELS}"
    assert config.TARGET_CHANNELS == ["F3", "F4", "C3", "Cz", "C4", "P3", "P4", "BIP", "accX", "accY", "accZ"]


def test_config_paths():
    assert Path(config.MODEL_PATH).suffix == ".joblib"
    assert Path(config.METADATA_PATH).suffix == ".json"


run_test("Channel count and names", test_config_channels)
run_test("Model and metadata paths", test_config_paths)

print()
print("-- 2. Preprocessor -----------------------------------------")


def test_preprocess_output_shape():
    out = preprocess(FAKE, sfreq=config.SFREQ)
    assert out.shape == (N_CH, N_SMP)


def test_channel_selection_correct_order():
    stream_names = ["misc0", "accY", "F4", "C4", "BIP", "Cz", "P4", "accX", "C3", "F3", "P3", "accZ", "misc1"]
    stream_data = np.random.randn(len(stream_names), N_SMP)
    stream_data[stream_names.index("BIP")] = 999.0
    selected = select_channels(stream_data, stream_names)
    assert selected.shape[0] == N_CH
    bip_idx = config.TARGET_CHANNELS.index("BIP")
    assert np.all(selected[bip_idx] == 999.0)


run_test("Output shape", test_preprocess_output_shape)
run_test("Channel selection order", test_channel_selection_correct_order)

print()
print("-- 3. Feature Extractor ------------------------------------")


def test_feature_shape():
    ext = FeatureExtractor()
    ext.load()
    feats = ext.extract(FAKE)
    assert feats.shape == (1, 43), f"Expected (1, 43), got {feats.shape}"


def test_feature_no_nan_or_inf():
    ext = FeatureExtractor()
    ext.load()
    feats = ext.extract(FAKE)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))


def test_bandpower_values():
    signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, config.SFREQ))
    bp_alpha = bandpower(signal, config.SFREQ, 8.0, 13.0)
    bp_emg = bandpower(signal, config.SFREQ, 40.0, 80.0)
    assert bp_alpha > bp_emg


run_test("Feature shape", test_feature_shape)
run_test("No NaN/Inf in features", test_feature_no_nan_or_inf)
run_test("Bandpower sanity", test_bandpower_values)

print()
print("-- 4. Classifier -------------------------------------------")


def test_classifier_loads_and_predicts():
    clf = Classifier()
    clf.load()
    ext = FeatureExtractor()
    ext.load()
    feats = ext.extract(FAKE)
    predicted_class, label = clf.predict(feats)
    assert predicted_class in [0, 1, 2]
    assert label in config.CLASS_LABELS.values()


run_test("Classifier load and predict", test_classifier_loads_and_predicts)

print()
print("-- 5. Smoothing --------------------------------------------")


def majority_vote(history):
    if not history:
        return 0
    counts = collections.Counter(history)
    best_count = max(counts.values())
    winners = {label for label, count in counts.items() if count == best_count}
    for label in reversed(history):
        if label in winners:
            return label
    return history[-1]


def test_smoothing_majority():
    history = collections.deque([0, 1, 1, 1, 0], maxlen=config.SMOOTHING_WINDOW)
    assert majority_vote(history) == 1


run_test("Majority vote", test_smoothing_majority)
