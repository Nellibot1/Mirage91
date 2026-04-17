#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABEL_TO_NAME = {
    0: "neutral",
    1: "blink",
    2: "clench",
}

RUN_MARKER_TO_LABEL = {
    "rest": 0,
    "blinking": 1,
    "teeth_clenching": 2,
}

TARGET_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4", "BIP", "accX", "accY", "accZ"]
FEATURE_EEG_CHANNELS = ["F3", "F4", "BIP"]
ACC_CHANNELS = ["accX", "accY", "accZ"]

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 100.0
USE_NOTCH = True
NOTCH_FREQ = 50.0

WINDOW_SEC = 0.5
STEP_SEC = 0.1
MIN_SEGMENT_SEC = 0.5
TRANSITION_PAD_SEC = 0.1

TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass
class RunData:
    subject_name: str
    group_id: str
    data: np.ndarray
    ch_names: list[str]
    sfreq: float
    markers: pd.DataFrame | None
    file_path: str


def parse_subject_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    match = re.search(r"_p(\d+)", stem)
    if match:
        return f"p{match.group(1)}"
    return stem


def safe_marker_name(value: object) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return ""
        value = value[0]
    return str(value).strip()


def load_xdf_run(xdf_file: str, target_channels: list[str]) -> RunData:
    try:
        import pyxdf
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyxdf. Install it with `pip install pyxdf` before training."
        ) from exc

    streams, _header = pyxdf.load_xdf(str(xdf_file))

    eeg_stream = None
    marker_stream = None
    for stream in streams:
        name = stream["info"].get("name", [""])[0]
        stream_type = stream["info"].get("type", [""])[0]

        if stream_type.upper() == "EEG" and eeg_stream is None:
            eeg_stream = stream

        if stream_type.lower() == "markers" or name.lower() == "paradigm":
            marker_stream = stream

    if eeg_stream is None:
        raise ValueError(f"No EEG stream found in {xdf_file}")

    eeg_data = np.asarray(eeg_stream["time_series"], dtype=float).T
    eeg_timestamps = np.asarray(eeg_stream["time_stamps"], dtype=float)
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])

    ch_info = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ch_names = [channel["label"][0] for channel in ch_info]

    picks = [idx for idx, ch_name in enumerate(ch_names) if ch_name in target_channels]
    if not picks:
        raise ValueError(f"No target channels found in {xdf_file}")

    eeg_data = eeg_data[picks, :]
    ch_names = [ch_names[idx] for idx in picks]

    ordered_names = [ch for ch in target_channels if ch in ch_names]
    ordered_indices = [ch_names.index(ch) for ch in ordered_names]
    eeg_data = eeg_data[ordered_indices, :]
    ch_names = [ch_names[idx] for idx in ordered_indices]

    markers_df = None
    if marker_stream is not None:
        marker_timestamps = np.asarray(marker_stream["time_stamps"], dtype=float)
        marker_values = [safe_marker_name(v) for v in marker_stream["time_series"]]
        eeg_start_ts = eeg_timestamps[0]
        markers_df = pd.DataFrame(
            {
                "timestamp": marker_timestamps,
                "marker": marker_values,
                "time_from_eeg_start_sec": marker_timestamps - eeg_start_ts,
            }
        )

    return RunData(
        subject_name=Path(xdf_file).stem,
        group_id=parse_subject_id(xdf_file),
        data=eeg_data,
        ch_names=ch_names,
        sfreq=sfreq,
        markers=markers_df,
        file_path=xdf_file,
    )


def load_all_runs(data_folder: Path, pattern: str, target_channels: list[str]) -> list[RunData]:
    files = sorted(glob(str(data_folder / pattern)))
    if not files:
        raise FileNotFoundError(f"No XDF files found in {data_folder} with pattern {pattern!r}")

    runs = [load_xdf_run(file_path, target_channels=target_channels) for file_path in files]
    print(f"Loaded {len(runs)} XDF run(s).")
    for run in runs:
        marker_names = [] if run.markers is None else sorted(run.markers["marker"].unique())
        print(f"  - {run.subject_name} | group={run.group_id} | markers={marker_names}")
    return runs


def preprocess_data(data: np.ndarray, sfreq: float, use_notch: bool, notch_freq: float) -> np.ndarray:
    nyquist = sfreq / 2.0
    high_cut = min(BANDPASS_HIGH, nyquist - 1.0)
    if high_cut <= BANDPASS_LOW:
        raise ValueError(f"Invalid bandpass for sampling rate {sfreq}")

    band_b, band_a = butter(4, [BANDPASS_LOW / nyquist, high_cut / nyquist], btype="band")
    filtered = filtfilt(band_b, band_a, data, axis=1)

    if use_notch and notch_freq < nyquist:
        notch_b, notch_a = iirnotch(notch_freq / nyquist, Q=30.0)
        filtered = filtfilt(notch_b, notch_a, filtered, axis=1)

    return filtered


def extract_run_segments(
    data: np.ndarray, sfreq: float, markers_df: pd.DataFrame | None
) -> list[dict[str, object]]:
    if markers_df is None or markers_df.empty:
        return []

    markers_df = markers_df.sort_values("timestamp").reset_index(drop=True)
    rel_times = markers_df["time_from_eeg_start_sec"].to_numpy()
    marker_names = markers_df["marker"].astype(str).str.strip().to_numpy()
    total_duration_sec = data.shape[1] / sfreq
    segments: list[dict[str, object]] = []

    for idx, marker_name in enumerate(marker_names):
        if marker_name not in RUN_MARKER_TO_LABEL:
            continue

        pause_idx = None
        for inner_idx in range(idx + 1, len(marker_names)):
            if marker_names[inner_idx] == "pause":
                pause_idx = inner_idx
                break

        if pause_idx is None:
            continue

        start_sec = max(float(rel_times[idx]) + TRANSITION_PAD_SEC, 0.0)
        end_sec = min(float(rel_times[pause_idx]) - TRANSITION_PAD_SEC, total_duration_sec)
        if end_sec <= start_sec:
            continue

        duration_sec = end_sec - start_sec
        if duration_sec < MIN_SEGMENT_SEC:
            continue

        start_sample = max(0, int(round(start_sec * sfreq)))
        end_sample = min(data.shape[1], int(round(end_sec * sfreq)))
        if end_sample <= start_sample:
            continue

        segments.append(
            {
                "marker": marker_name,
                "label": RUN_MARKER_TO_LABEL[marker_name],
                "sfreq": sfreq,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
                "data": data[:, start_sample:end_sample],
            }
        )

    return segments


def create_windows(segment: np.ndarray, sfreq: float, window_sec: float, step_sec: float) -> np.ndarray:
    n_channels, n_samples = segment.shape
    window_samples = int(round(window_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))

    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_sec and step_sec must produce positive sample counts")
    if n_samples < window_samples:
        return np.empty((0, n_channels, window_samples))

    starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    return np.stack([segment[:, start:start + window_samples] for start in starts], axis=0)


def bandpower(signal_1d: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    nperseg = min(len(signal_1d), max(32, int(round(sfreq))))
    if nperseg < 8:
        return 0.0

    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(psd[mask], freqs[mask]))


def safe_ratio(a: float, b: float, eps: float = 1e-10) -> float:
    return float(a / (b + eps))


def compact_feature_names(ch_names: list[str]) -> list[str]:
    names: list[str] = []
    for ch_name in ch_names:
        if ch_name in FEATURE_EEG_CHANNELS:
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
        elif ch_name in ACC_CHANNELS:
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


def extract_features(window: np.ndarray, sfreq: float, ch_names: list[str]) -> np.ndarray:
    feats: list[float] = []

    for ch_idx, ch_name in enumerate(ch_names):
        if ch_name not in FEATURE_EEG_CHANNELS and ch_name not in ACC_CHANNELS:
            continue

        x = window[ch_idx]
        rms_val = float(np.sqrt(np.mean(x ** 2)))
        ptp_val = float(np.ptp(x))
        dx = np.diff(x)
        diff_rms = float(np.sqrt(np.mean(dx ** 2))) if len(dx) else 0.0

        if ch_name in ACC_CHANNELS:
            feats.extend([float(np.std(x)), rms_val, ptp_val, diff_rms])
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

    if "F3" in ch_names and "F4" in ch_names:
        f3_idx = ch_names.index("F3")
        f4_idx = ch_names.index("F4")
        alpha_asymmetry = bandpower(window[f4_idx], sfreq, 8.0, 13.0) - bandpower(
            window[f3_idx], sfreq, 8.0, 13.0
        )
        feats.append(float(alpha_asymmetry))

    return np.asarray(feats, dtype=float)


def build_dataset_from_runs(
    runs: Iterable[RunData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    group_list: list[str] = []
    meta_rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None

    for run in runs:
        filtered_data = preprocess_data(
            run.data, run.sfreq, use_notch=USE_NOTCH, notch_freq=NOTCH_FREQ
        )
        segments = extract_run_segments(filtered_data, run.sfreq, run.markers)
        ch_names = run.ch_names

        if feature_names is None:
            feature_names = compact_feature_names(ch_names)

        print(f"Building windows for {run.subject_name}: {len(segments)} segment(s)")

        for segment_index, segment in enumerate(segments):
            windows = create_windows(
                segment["data"],
                float(segment["sfreq"]),
                window_sec=WINDOW_SEC,
                step_sec=STEP_SEC,
            )
            if len(windows) == 0:
                continue

            for window_index, window in enumerate(windows):
                feats = extract_features(window, float(segment["sfreq"]), ch_names)
                X_list.append(feats)
                y_list.append(int(segment["label"]))
                group_list.append(run.group_id)
                meta_rows.append(
                    {
                        "subject_name": run.subject_name,
                        "group_id": run.group_id,
                        "segment_index": segment_index,
                        "window_index": window_index,
                        "marker": segment["marker"],
                        "label": int(segment["label"]),
                        "label_name": LABEL_TO_NAME[int(segment["label"])],
                    }
                )

    if not X_list:
        raise RuntimeError("No windows were extracted from the provided recordings.")

    return (
        np.asarray(X_list, dtype=float),
        np.asarray(y_list, dtype=int),
        np.asarray(group_list),
        pd.DataFrame(meta_rows),
        feature_names or [],
    )


def balance_by_participant_and_class(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame({"idx": np.arange(len(y)), "group": groups, "label": y})
    counts = df.groupby(["group", "label"]).size()
    min_count = int(counts.min())

    keep_indices: list[int] = []
    for (group_id, label), _count in counts.items():
        idxs = df.loc[(df["group"] == group_id) & (df["label"] == label), "idx"].to_numpy()
        chosen = rng.choice(idxs, size=min_count, replace=False)
        keep_indices.extend(int(idx) for idx in chosen)

    keep_indices = np.asarray(sorted(keep_indices))
    return (
        X[keep_indices],
        y[keep_indices],
        groups[keep_indices],
        metadata.iloc[keep_indices].reset_index(drop=True),
    )


def describe_dataset(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    print(f"\n{name}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  groups: {sorted(np.unique(groups).tolist())}")
    class_counts = {
        LABEL_TO_NAME[int(label)]: int(count)
        for label, count in zip(*np.unique(y, return_counts=True))
    }
    print(f"  class counts: {class_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classifier Version 4.2 "
            "L1-based automatic feature selection evaluated with LOSO."
        )
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path("/Users/janniella/Desktop/Mirage/prototype"),
        help="Folder of XDF recordings.",
    )
    parser.add_argument(
        "--pattern",
        default="*.xdf",
        help="Glob pattern for XDF files inside --data-folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Trained model, metadata, and LOSO report should be saved.",
    )
    parser.add_argument(
        "--skip-balance",
        action="store_true",
        help="Train on the raw window dataset instead of participant/class-balanced data.",
    )
    parser.add_argument(
        "--selector-c",
        type=float,
        default=0.05,
        help="Inverse regularization strength for the L1 logistic selector.",
    )
    return parser.parse_args()


def make_model_candidates(selector_c: float) -> dict[str, Pipeline]:
    return {
        "compact_linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000)),
            ]
        ),
        "compact_linear_svm_l1_select": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "selector",
                    SelectFromModel(
                        LogisticRegression(
                            penalty="l1",
                            solver="saga",
                            C=selector_c,
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=RANDOM_STATE,
                        ),
                        threshold=1e-8,
                    ),
                ),
                ("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000)),
            ]
        ),
    }


def evaluate_loso_with_tracking(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
) -> dict[str, object]:
    logo = LeaveOneGroupOut()
    fold_results: list[dict[str, object]] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    train_accuracies: list[float] = []
    selector_counts: list[int] = []
    selector_features: list[dict[str, object]] = []
    for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        fitted = clone(model)
        fitted.fit(X[train_idx], y[train_idx])
        y_train_pred = fitted.predict(X[train_idx])
        y_test_pred = fitted.predict(X[test_idx])
        train_accuracy = float(accuracy_score(y[train_idx], y_train_pred))
        test_accuracy = float(accuracy_score(y[test_idx], y_test_pred))
        train_accuracies.append(train_accuracy)

        fold_results.append(
            {
                "fold": fold_index,
                "test_groups": sorted(set(groups[test_idx].tolist())),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }
        )
        all_true.extend(y[test_idx].tolist())
        all_pred.extend(y_test_pred.tolist())

        if "selector" in fitted.named_steps:
            selector = fitted.named_steps["selector"]
            support = selector.get_support()
            selected = [feature_names[idx] for idx, keep in enumerate(support) if keep]
            selector_counts.append(len(selected))
            selector_features.append(
                {
                    "fold": fold_index,
                    "test_groups": sorted(set(groups[test_idx].tolist())),
                    "selected_feature_count": len(selected),
                    "selected_features": selected,
                }
            )

    result = {
        "train_accuracy_mean": float(np.mean(train_accuracies)),
        "accuracy": float(accuracy_score(np.asarray(all_true), np.asarray(all_pred))),
        "confusion_matrix": confusion_matrix(
            np.asarray(all_true),
            np.asarray(all_pred),
            labels=sorted(LABEL_TO_NAME),
        ).tolist(),
        "folds": fold_results,
    }

    if selector_features:
        result["selector"] = {
            "selected_feature_count_per_fold": selector_counts,
            "selected_features_per_fold": selector_features,
        }
    else:
        result["selector"] = None

    return result


def evaluate_random_split_with_train(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, object]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(X, y))

    fitted = clone(model)
    fitted.fit(X[train_idx], y[train_idx])
    y_train_pred = fitted.predict(X[train_idx])
    y_test_pred = fitted.predict(X[test_idx])

    return {
        "train_accuracy": float(accuracy_score(y[train_idx], y_train_pred)),
        "accuracy": float(accuracy_score(y[test_idx], y_test_pred)),
        "train_confusion_matrix": confusion_matrix(
            y[train_idx], y_train_pred, labels=sorted(LABEL_TO_NAME)
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            y[test_idx], y_test_pred, labels=sorted(LABEL_TO_NAME)
        ).tolist(),
    }


def print_loso_summary(title: str, result: dict[str, object]) -> None:
    print(f"\n{title}")
    print(f"  mean train accuracy: {float(result['train_accuracy_mean']):.4f}")
    print(f"  LOSO test accuracy: {float(result['accuracy']):.4f}")
    print("  confusion matrix:")
    for row in result["confusion_matrix"]:
        print(f"    {row}")
    selector_info = result.get("selector")
    if selector_info:
        print(
            "  selected feature counts per fold: "
            f"{selector_info['selected_feature_count_per_fold']}"
        )


def print_random_summary(title: str, result: dict[str, object]) -> None:
    print(f"\n{title}")
    print(f"  train accuracy: {float(result['train_accuracy']):.4f}")
    print(f"  test accuracy: {float(result['accuracy']):.4f}")
    print("  test confusion matrix:")
    for row in result["confusion_matrix"]:
        print(f"    {row}")


def pick_best_model(
    candidates: dict[str, Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
) -> tuple[str, Pipeline, dict[str, object], dict[str, dict[str, object]]]:
    all_results: dict[str, dict[str, object]] = {}
    best_name = ""
    best_model: Pipeline | None = None
    best_result: dict[str, object] | None = None

    for name, model in candidates.items():
        result = evaluate_loso_with_tracking(model, X, y, groups, feature_names)
        all_results[name] = result
        print(f"{name} LOSO accuracy: {result['accuracy']:.4f}")
        if best_result is None or float(result["accuracy"]) > float(best_result["accuracy"]):
            best_name = name
            best_model = model
            best_result = result

    if best_model is None or best_result is None:
        raise RuntimeError("Could not select a best model.")

    return best_name, best_model, best_result, all_results


def save_outputs(
    output_dir: Path,
    model_name: str,
    fitted_model: Pipeline,
    metadata: dict[str, object],
    report: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "eeg_artifact_game_control_v4_2.joblib"
    metadata_path = output_dir / "eeg_artifact_game_control_v4_2_metadata.json"
    report_path = output_dir / "eeg_artifact_game_control_v4_2_report.json"

    joblib.dump(fitted_model, model_path)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    save_plots(output_dir, report)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved report: {report_path}")
    print(f"Selected model: {model_name}")


def save_plots(output_dir: Path, report: dict[str, object]) -> None:
    loso_result = report["selected_model_loso"]
    cm = np.asarray(loso_result["confusion_matrix"], dtype=int)
    labels = [LABEL_TO_NAME[idx] for idx in sorted(LABEL_TO_NAME)]

    heatmap_path = output_dir / "eeg_artifact_game_control_v4_2_loso_confusion_matrix.png"
    random_heatmap_path = output_dir / "eeg_artifact_game_control_v4_2_random_confusion_matrix.png"
    summary_path = output_dir / "eeg_artifact_game_control_v4_2_model_comparison.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("ClassifierVersion4.2 LOSO Confusion Matrix")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    random_result = report.get("selected_model_random_split")
    if random_result is not None:
        random_cm = np.asarray(random_result["confusion_matrix"], dtype=int)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=random_cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Greens", colorbar=True, values_format="d")
        ax.set_title("ClassifierVersion4.2 Random Split Confusion Matrix")
        fig.tight_layout()
        fig.savefig(random_heatmap_path, dpi=180)
        plt.close(fig)

    model_results = report["model_selection_loso"]
    model_names = list(model_results.keys())
    accuracies = [float(model_results[name]["accuracy"]) for name in model_names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(model_names, accuracies, color=["#4C78A8", "#72B7B2"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("LOSO Accuracy")
    ax.set_title("ClassifierVersion4.2 Model Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(summary_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    runs = load_all_runs(args.data_folder, args.pattern, target_channels=TARGET_CHANNELS)
    X, y, groups, metadata_df, feature_names = build_dataset_from_runs(runs)
    describe_dataset("Raw dataset", X, y, groups)

    train_X = X
    train_y = y
    train_groups = groups
    train_meta = metadata_df

    if not args.skip_balance:
        train_X, train_y, train_groups, train_meta = balance_by_participant_and_class(
            X,
            y,
            groups,
            metadata_df,
            random_state=RANDOM_STATE,
        )
        describe_dataset("Balanced dataset", train_X, train_y, train_groups)

    candidates = make_model_candidates(selector_c=args.selector_c)
    best_name, best_model, best_result, all_results = pick_best_model(
        candidates,
        train_X,
        train_y,
        train_groups,
        feature_names,
    )
    random_result = evaluate_random_split_with_train(best_model, train_X, train_y)

    print_loso_summary("Best-model LOSO", best_result)
    print_random_summary("Best-model random split", random_result)

    final_model = clone(best_model)
    final_model.fit(train_X, train_y)

    final_selected_features: list[str] | None = None
    if "selector" in final_model.named_steps:
        support = final_model.named_steps["selector"].get_support()
        final_selected_features = [
            feature_names[idx] for idx, keep in enumerate(support) if keep
        ]

    metadata = {
        "version": "ClassifierVersion4.2",
        "description": (
            "Compact physiological feature base with train/test reporting, "
            "LOSO-primary evaluation, and "
            "optional L1 logistic feature selection inside the CV pipeline."
        ),
        "selected_model": best_name,
        "evaluation_protocol": "LeaveOneGroupOut primary, random split secondary",
        "target_channels": TARGET_CHANNELS,
        "feature_eeg_channels": FEATURE_EEG_CHANNELS,
        "acc_channels": ACC_CHANNELS,
        "feature_names": feature_names,
        "final_selected_features": final_selected_features,
        "label_mapping": RUN_MARKER_TO_LABEL,
        "label_names": LABEL_TO_NAME,
        "training_source": "XDF run recordings",
        "segment_rule": "task marker to next pause, trimmed by transition pad",
        "transition_pad_sec": TRANSITION_PAD_SEC,
        "bandpass_low": BANDPASS_LOW,
        "bandpass_high": BANDPASS_HIGH,
        "use_notch": USE_NOTCH,
        "notch_freq": NOTCH_FREQ,
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "selector_c": args.selector_c,
        "n_samples": int(train_X.shape[0]),
        "n_features": int(train_X.shape[1]),
        "n_groups": int(len(np.unique(train_groups))),
        "group_ids": sorted(str(group_id) for group_id in np.unique(train_groups)),
        "files_used": sorted(Path(run.file_path).name for run in runs),
        "balanced_training": not args.skip_balance,
    }

    report = {
        "model_selection_loso": all_results,
        "selected_model_loso": best_result,
        "selected_model_random_split": random_result,
        "dataset": {
            "raw_samples": int(X.shape[0]),
            "train_samples": int(train_X.shape[0]),
            "train_groups": sorted(str(group_id) for group_id in np.unique(train_groups)),
            "train_class_counts": {
                LABEL_TO_NAME[int(label)]: int(count)
                for label, count in zip(*np.unique(train_y, return_counts=True))
            },
        },
    }

    save_outputs(args.output_dir, best_name, final_model, metadata, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
