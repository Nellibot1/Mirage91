#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

MODEL_PATH = Path("artifacts/eeg_artifact_game_control_v4_2.joblib")
METADATA_PATH = Path("artifacts/eeg_artifact_game_control_v4_2_metadata.json")

LABEL_TO_NAME = {
    0: "neutral",
    1: "blink",
    2: "clench",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time EEG classification from an LSL EEG stream for classifier v4.2."
    )
    parser.add_argument(
        "--stream-name",
        help="Exact LSL stream name to connect to.",
    )
    parser.add_argument(
        "--stream-type",
        default="EEG",
        help="LSL stream type to search for when --stream-name is not given.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to the trained scikit-learn model.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=METADATA_PATH,
        help="Path to the training metadata JSON.",
    )
    parser.add_argument(
        "--channel-names",
        help="Comma-separated fallback channel list if the LSL stream does not expose labels.",
    )
    parser.add_argument(
        "--command-host",
        default="127.0.0.1",
        help="UDP host for game commands.",
    )
    parser.add_argument(
        "--command-port",
        type=int,
        default=8765,
        help="UDP port for game commands.",
    )
    parser.add_argument(
        "--blink-command",
        default="DUCK",
        help="Game command emitted for a detected double blink.",
    )
    parser.add_argument(
        "--clench-command",
        default="ACTION",
        help="Game command emitted for a detected clench burst.",
    )
    parser.add_argument(
        "--majority-span",
        type=int,
        default=5,
        help="Rolling majority-vote span over recent window predictions.",
    )
    parser.add_argument(
        "--min-burst-windows",
        type=int,
        default=2,
        help="Minimum smoothed windows required before a blink/clench burst counts.",
    )
    parser.add_argument(
        "--double-blink-min-gap-sec",
        type=float,
        default=0.1,
        help="Minimum gap between two blink bursts to count as a double blink.",
    )
    parser.add_argument(
        "--double-blink-max-gap-sec",
        type=float,
        default=0.6,
        help="Maximum gap between two blink bursts to count as a double blink.",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=0.75,
        help="Minimum time between identical emitted commands.",
    )
    parser.add_argument(
        "--print-predictions",
        action="store_true",
        help="Print raw and smoothed predictions for debugging.",
    )
    parser.add_argument(
        "--print-command-debug",
        action="store_true",
        help="Print detector state transitions for debugging.",
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_channel_names_from_xml(xml_text: str) -> list[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    labels: list[str] = []
    for channel in root.findall(".//channels/channel"):
        label = channel.findtext("label")
        if label:
            labels.append(label.strip())
    return labels


def resolve_lsl_stream(stream_name: str | None, stream_type: str):
    from pylsl import StreamInlet, resolve_byprop

    if stream_name:
        results = resolve_byprop("name", stream_name, timeout=5.0)
    else:
        results = resolve_byprop("type", stream_type, timeout=5.0)

    if not results:
        target = f"name={stream_name!r}" if stream_name else f"type={stream_type!r}"
        raise RuntimeError(f"No LSL stream found for {target}.")

    stream_info = results[0]
    inlet = StreamInlet(stream_info, max_buflen=5, max_chunklen=1)
    return inlet, stream_info


def build_filters(sfreq: float, bandpass_low: float, bandpass_high: float, use_notch: bool, notch_freq: float):
    nyquist = sfreq / 2.0
    high_cut = min(bandpass_high, nyquist - 1.0)
    if high_cut <= bandpass_low:
        raise ValueError(
            f"Invalid filter settings for sampling rate {sfreq:.3f} Hz: "
            f"low={bandpass_low}, high={bandpass_high}"
        )

    band_b, band_a = butter(4, [bandpass_low / nyquist, high_cut / nyquist], btype="band")
    notch = None
    if use_notch and notch_freq < nyquist:
        notch = iirnotch(notch_freq / nyquist, Q=30.0)
    return (band_b, band_a), notch


def apply_filters(data: np.ndarray, filters: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None]) -> np.ndarray:
    bandpass, notch = filters
    band_b, band_a = bandpass
    try:
        filtered = filtfilt(band_b, band_a, data, axis=1)
        if notch is not None:
            notch_b, notch_a = notch
            filtered = filtfilt(notch_b, notch_a, filtered, axis=1)
        return filtered
    except ValueError:
        return data


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


def compact_feature_names(ch_names: list[str], feature_eeg_channels: list[str], acc_channels: list[str]) -> list[str]:
    names: list[str] = []
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


def extract_features(
    window: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    feature_eeg_channels: list[str],
    acc_channels: list[str],
) -> np.ndarray:
    feats: list[float] = []

    for ch_idx, ch_name in enumerate(ch_names):
        if ch_name not in feature_eeg_channels and ch_name not in acc_channels:
            continue

        x = window[ch_idx]
        rms_val = float(np.sqrt(np.mean(x ** 2)))
        ptp_val = float(np.ptp(x))
        dx = np.diff(x)
        diff_rms = float(np.sqrt(np.mean(dx ** 2))) if len(dx) else 0.0

        if ch_name in acc_channels:
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


def majority_vote(labels: deque[int]) -> int:
    counts = Counter(labels)
    best_count = max(counts.values())
    winners = {label for label, count in counts.items() if count == best_count}
    for label in reversed(labels):
        if label in winners:
            return int(label)
    return int(labels[-1])


class CommandDetector:
    def __init__(
        self,
        min_burst_windows: int,
        double_blink_min_gap_sec: float,
        double_blink_max_gap_sec: float,
        cooldown_sec: float,
        blink_command: str,
        clench_command: str,
        print_debug: bool = False,
    ) -> None:
        self.min_burst_windows = min_burst_windows
        self.double_blink_min_gap_sec = double_blink_min_gap_sec
        self.double_blink_max_gap_sec = double_blink_max_gap_sec
        self.cooldown_sec = cooldown_sec
        self.blink_command = blink_command
        self.clench_command = clench_command
        self.print_debug = print_debug

        self.current_label: int | None = None
        self.current_run_length = 0
        self.last_blink_burst_time: float | None = None
        self.last_command_time: dict[str, float] = {}
        self.clench_emitted_in_run = False

    def update(self, smoothed_label: int, event_time: float) -> list[tuple[str, str]]:
        emitted: list[tuple[str, str]] = []

        if self.current_label is None:
            self.current_label = smoothed_label
            self.current_run_length = 1
        elif smoothed_label == self.current_label:
            self.current_run_length += 1
        else:
            emitted.extend(self._finalize_run(event_time))
            self.current_label = smoothed_label
            self.current_run_length = 1
            self.clench_emitted_in_run = False

        if self.current_label == 2 and not self.clench_emitted_in_run:
            if self.current_run_length >= self.min_burst_windows:
                if self._cooldown_ready(self.clench_command, event_time):
                    emitted.append((self.clench_command, "clench"))
                    self.last_command_time[self.clench_command] = event_time
                self.clench_emitted_in_run = True

        return emitted

    def _finalize_run(self, event_time: float) -> list[tuple[str, str]]:
        if self.current_label != 1 or self.current_run_length < self.min_burst_windows:
            return []

        if self.print_debug:
            print(
                f"[detector] blink burst ended at {event_time:.3f}s "
                f"(length={self.current_run_length})"
            )

        emitted: list[tuple[str, str]] = []
        if self.last_blink_burst_time is not None:
            gap = event_time - self.last_blink_burst_time
            if self.double_blink_min_gap_sec <= gap <= self.double_blink_max_gap_sec:
                if self._cooldown_ready(self.blink_command, event_time):
                    emitted.append((self.blink_command, "double_blink"))
                    self.last_command_time[self.blink_command] = event_time
                self.last_blink_burst_time = None
                return emitted

        self.last_blink_burst_time = event_time
        return emitted

    def _cooldown_ready(self, command: str, now_sec: float) -> bool:
        previous = self.last_command_time.get(command)
        return previous is None or (now_sec - previous) >= self.cooldown_sec


def send_udp_command(sock: socket.socket, host: str, port: int, command: str, source: str, time_sec: float) -> None:
    payload = {
        "command": command,
        "time_sec": round(float(time_sec), 3),
        "source": source,
        "timestamp": time.time(),
    }
    sock.sendto(json.dumps(payload).encode("utf-8"), (host, port))


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.metadata_path)
    model = joblib.load(args.model_path)

    inlet, stream_info = resolve_lsl_stream(args.stream_name, args.stream_type)
    sfreq = float(stream_info.nominal_srate())
    if sfreq <= 0:
        raise RuntimeError("LSL stream reports a non-positive nominal sample rate.")

    stream_ch_names = parse_channel_names_from_xml(stream_info.as_xml())
    if not stream_ch_names and args.channel_names:
        stream_ch_names = [item.strip() for item in args.channel_names.split(",") if item.strip()]

    target_channels = list(metadata["target_channels"])
    if not stream_ch_names:
        raise RuntimeError(
            "Could not read channel labels from the LSL stream. Pass --channel-names as a fallback."
        )

    missing = [name for name in target_channels if name not in stream_ch_names]
    if missing:
        raise RuntimeError(
            "The LSL stream is missing required channels for classifier v4.2: "
            + ", ".join(missing)
        )

    channel_indices = [stream_ch_names.index(name) for name in target_channels]
    selected_ch_names = [stream_ch_names[idx] for idx in channel_indices]
    feature_names = compact_feature_names(
        selected_ch_names,
        feature_eeg_channels=list(metadata["feature_eeg_channels"]),
        acc_channels=list(metadata["acc_channels"]),
    )

    expected_feature_names = list(metadata["feature_names"])
    if feature_names != expected_feature_names:
        warnings.warn(
            "Online feature order does not match the saved training metadata exactly. "
            "Predictions may be invalid.",
            stacklevel=2,
        )

    filters = build_filters(
        sfreq,
        bandpass_low=float(metadata["bandpass_low"]),
        bandpass_high=float(metadata["bandpass_high"]),
        use_notch=bool(metadata["use_notch"]),
        notch_freq=float(metadata["notch_freq"]),
    )

    window_sec = float(metadata["window_sec"])
    step_sec = float(metadata["step_sec"])
    min_samples = max(2, int(round(window_sec * sfreq)))
    prediction_history: deque[int] = deque(maxlen=max(1, args.majority_span))
    samples: deque[np.ndarray] = deque()
    timestamps: deque[float] = deque()
    detector = CommandDetector(
        min_burst_windows=max(1, args.min_burst_windows),
        double_blink_min_gap_sec=args.double_blink_min_gap_sec,
        double_blink_max_gap_sec=args.double_blink_max_gap_sec,
        cooldown_sec=args.cooldown_sec,
        blink_command=args.blink_command,
        clench_command=args.clench_command,
        print_debug=args.print_command_debug,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    next_eval_time: float | None = None
    lsl_start_timestamp: float | None = None
    max_buffer_age_sec = max(window_sec * 2.5, 2.0)

    print(f"Connected to LSL stream: {stream_info.name()} ({stream_info.type()}) @ {sfreq:.2f} Hz")
    print("Using channels:", ",".join(selected_ch_names))
    print(
        f"Window={window_sec:.3f}s Step={step_sec:.3f}s "
        f"UDP={args.command_host}:{args.command_port}"
    )

    try:
        while True:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue

            selected = np.asarray([sample[idx] for idx in channel_indices], dtype=float)
            samples.append(selected)
            timestamps.append(float(timestamp))
            if lsl_start_timestamp is None:
                lsl_start_timestamp = float(timestamp)

            newest_time = timestamps[-1]
            while timestamps and (newest_time - timestamps[0]) > max_buffer_age_sec:
                timestamps.popleft()
                samples.popleft()

            if next_eval_time is None:
                next_eval_time = newest_time

            if newest_time < next_eval_time:
                continue

            window_start_time = newest_time - window_sec
            valid_indices = [idx for idx, ts in enumerate(timestamps) if ts >= window_start_time]
            if len(valid_indices) < min_samples:
                next_eval_time += step_sec
                continue

            window_matrix = np.stack([samples[idx] for idx in valid_indices], axis=1)
            filtered_window = apply_filters(window_matrix, filters)
            feature_vector = extract_features(
                filtered_window,
                sfreq,
                selected_ch_names,
                feature_eeg_channels=list(metadata["feature_eeg_channels"]),
                acc_channels=list(metadata["acc_channels"]),
            )

            raw_pred = int(model.predict(feature_vector.reshape(1, -1))[0])
            prediction_history.append(raw_pred)
            smoothed_pred = majority_vote(prediction_history)

            if args.print_predictions:
                rel_time = newest_time - lsl_start_timestamp
                print(
                    f"[pred] t={rel_time:7.3f}s raw={LABEL_TO_NAME.get(raw_pred, raw_pred)} "
                    f"smooth={LABEL_TO_NAME.get(smoothed_pred, smoothed_pred)}"
                )

            rel_time = newest_time - lsl_start_timestamp
            emitted = detector.update(smoothed_pred, rel_time)
            for command, source in emitted:
                send_udp_command(
                    sock,
                    args.command_host,
                    args.command_port,
                    command=command,
                    source=source,
                    time_sec=rel_time,
                )
                print(f"[command] {command} from {source} at {rel_time:.3f}s")

            next_eval_time += step_sec
    except KeyboardInterrupt:
        print("\nStopped real-time inference.")
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
