# preprocessor.py

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

import config


def select_channels(raw_data, stream_ch_names):
    selected_rows = []
    for ch in config.TARGET_CHANNELS:
        if ch not in stream_ch_names:
            raise ValueError(
                f"Channel '{ch}' required by the model is not in the stream.\n"
                f"Stream channels: {stream_ch_names}\n"
                f"Required channels: {config.TARGET_CHANNELS}"
            )
        idx = stream_ch_names.index(ch)
        selected_rows.append(raw_data[idx])
    return np.asarray(selected_rows, dtype=float)


def _build_filters(sfreq):
    nyquist = sfreq / 2.0
    high_cut = min(config.H_FREQ, nyquist - 1.0)
    if high_cut <= config.L_FREQ:
        raise ValueError(
            f"Invalid bandpass for sampling rate {sfreq}: "
            f"low={config.L_FREQ}, high={config.H_FREQ}"
        )

    band_b, band_a = butter(4, [config.L_FREQ / nyquist, high_cut / nyquist], btype="band")
    notch = None
    if config.USE_NOTCH and config.NOTCH_FREQ < nyquist:
        notch = iirnotch(config.NOTCH_FREQ / nyquist, Q=30.0)
    return (band_b, band_a), notch


def preprocess(raw_data, stream_ch_names=None, sfreq=None):
    if stream_ch_names is not None:
        raw_data = select_channels(raw_data, stream_ch_names)
    else:
        raw_data = np.asarray(raw_data, dtype=float)

    sfreq = float(config.SFREQ if sfreq is None else sfreq)
    filters = _build_filters(sfreq)
    bandpass, notch = filters
    band_b, band_a = bandpass

    filtered = filtfilt(band_b, band_a, raw_data, axis=1)
    if notch is not None:
        notch_b, notch_a = notch
        filtered = filtfilt(notch_b, notch_a, filtered, axis=1)
    return filtered
