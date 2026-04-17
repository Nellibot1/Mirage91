# feature_extractor.py

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

import config


def bandpower(signal_1d, sfreq, fmin, fmax):
    signal_1d = np.asarray(signal_1d)
    nperseg = min(len(signal_1d), max(32, int(round(sfreq))))
    if nperseg < 8:
        return 0.0

    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0

    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(psd[mask], freqs[mask]))


def safe_ratio(a, b, eps=1e-10):
    return float(a / (b + eps))


def compact_feature_names(ch_names, feature_eeg_channels, acc_channels):
    names = []
    for ch_name in ch_names:
        if ch_name in feature_eeg_channels:
            names.extend(
                [
                    f"{ch_name}_rms",
                    f"{ch_name}_ptp",
                    f"{ch_name}_diff_rms",
                    f"{ch_name}_blink_bp",
                    f"{ch_name}_alpha_bp",
                    f"{ch_name}_beta_bp",
                    f"{ch_name}_emg_low_bp",
                    f"{ch_name}_emg_high_bp",
                    f"{ch_name}_blink_rel",
                    f"{ch_name}_emg_rel",
                ]
            )
        elif ch_name in acc_channels:
            names.extend(
                [
                    f"{ch_name}_std",
                    f"{ch_name}_rms",
                    f"{ch_name}_ptp",
                    f"{ch_name}_diff_rms",
                ]
            )
    if "F3" in ch_names and "F4" in ch_names:
        names.append("frontal_alpha_asymmetry")
    return names


class FeatureExtractor:

    def __init__(self):
        self.ch_names = list(config.TARGET_CHANNELS)
        self.sfreq = config.SFREQ
        self.feature_names = None
        self.expected_feature_names = None

    def load(self):
        metadata_path = Path(config.METADATA_PATH)
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Copy the v4.2 metadata JSON into the models folder first."
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.expected_feature_names = list(metadata["feature_names"])
        self.feature_names = compact_feature_names(
            self.ch_names,
            feature_eeg_channels=list(metadata["feature_eeg_channels"]),
            acc_channels=list(metadata["acc_channels"]),
        )

        if self.feature_names != self.expected_feature_names:
            raise ValueError(
                "Online feature order does not match v4.2 metadata. "
                f"Expected {self.expected_feature_names}, got {self.feature_names}"
            )

        print(
            "Feature extractor ready - using classifier v4.2 compact features "
            f"({len(self.feature_names)} features)."
        )

    def extract(self, filtered_data):
        features = self._extract_features_v4_2(filtered_data)
        return features.reshape(1, -1)

    def _extract_features_v4_2(self, window):
        feats = []
        sfreq = self.sfreq

        for ch_idx, ch_name in enumerate(self.ch_names):
            if ch_name not in config.FEATURE_EEG_CHANNELS and ch_name not in config.ACC_CHANNELS:
                continue

            x = window[ch_idx]
            rms_val = float(np.sqrt(np.mean(x ** 2)))
            ptp_val = float(np.ptp(x))
            dx = np.diff(x)
            diff_rms = float(np.sqrt(np.mean(dx ** 2))) if len(dx) else 0.0

            if ch_name in config.ACC_CHANNELS:
                feats.extend(
                    [
                        float(np.std(x)),
                        rms_val,
                        ptp_val,
                        diff_rms,
                    ]
                )
                continue

            bp_blink = bandpower(x, sfreq, 0.5, 4.0)
            bp_alpha = bandpower(x, sfreq, 8.0, 13.0)
            bp_beta = bandpower(x, sfreq, 13.0, 30.0)
            bp_emg_low = bandpower(x, sfreq, 20.0, 40.0)
            bp_emg_high = bandpower(x, sfreq, 40.0, min(80.0, sfreq / 2.0 - 1e-6))
            bp_total = bandpower(x, sfreq, 0.5, min(100.0, sfreq / 2.0 - 1e-6))

            feats.extend(
                [
                    rms_val,
                    ptp_val,
                    diff_rms,
                    bp_blink,
                    bp_alpha,
                    bp_beta,
                    bp_emg_low,
                    bp_emg_high,
                    safe_ratio(bp_blink, bp_total),
                    safe_ratio(bp_emg_low + bp_emg_high, bp_total),
                ]
            )

        f3_idx = self.ch_names.index("F3")
        f4_idx = self.ch_names.index("F4")
        alpha_asymmetry = bandpower(window[f4_idx], sfreq, 8.0, 13.0) - bandpower(
            window[f3_idx], sfreq, 8.0, 13.0
        )
        feats.append(float(alpha_asymmetry))

        return np.asarray(feats, dtype=float)
