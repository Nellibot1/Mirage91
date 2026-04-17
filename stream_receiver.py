# stream_receiver.py

import numpy as np
from mne_lsl.stream import StreamLSL

import config


class StreamReceiver:

    def __init__(self):
        self.stream = None
        self.is_connected = False
        self._channel_names = None
        self._sfreq = config.SFREQ

    def connect(self):
        stream_name = config.STREAM_NAME or None
        print(f"Connecting to LSL stream: name={stream_name!r}, type={config.STREAM_TYPE!r}...")
        self.stream = StreamLSL(
            bufsize=config.EPOCH_LENGTH * 4,
            name=stream_name,
        )
        self.stream.connect()
        self.is_connected = True

        self._channel_names = list(self.stream.ch_names)
        self._sfreq = self._resolve_sfreq()

        print(f"Connected! Stream channels ({len(self._channel_names)}): {self._channel_names}")
        print(f"Stream sampling rate: {self._sfreq}")

        missing = [ch for ch in config.TARGET_CHANNELS if ch not in self._channel_names]
        if missing:
            print(f"WARNING: These required channels are missing from the stream: {missing}")
        else:
            print(f"All required channels found: {config.TARGET_CHANNELS}")

    def _resolve_sfreq(self):
        info = getattr(self.stream, "info", None)
        if info is not None:
            sfreq_attr = getattr(info, "sfreq", None)
            if callable(sfreq_attr):
                return float(sfreq_attr())
            if sfreq_attr is not None:
                return float(sfreq_attr)

        stream_sfreq = getattr(self.stream, "sfreq", None)
        if stream_sfreq is not None:
            return float(stream_sfreq)
        return float(config.SFREQ)

    def get_channel_names(self):
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._channel_names

    def get_sfreq(self):
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._sfreq

    def get_chunk(self):
        if not self.is_connected:
            raise RuntimeError("Stream is not connected. Call connect() first.")
        data, timestamps = self.stream.get_data(winsize=config.EPOCH_LENGTH)
        return data, timestamps

    def disconnect(self):
        if self.is_connected:
            self.stream.disconnect()
            self.is_connected = False
            print("Stream disconnected.")


class FakeStreamReceiver:
    def __init__(self):
        self._channel_names = list(config.TARGET_CHANNELS)
        self._sfreq = config.SFREQ

    def connect(self):
        print("Fake stream connected.")
        print(f"Simulating {config.N_CHANNELS} channels: {config.TARGET_CHANNELS}")

    def get_channel_names(self):
        return self._channel_names

    def get_sfreq(self):
        return self._sfreq

    def get_chunk(self):
        n_samples = int(config.EPOCH_LENGTH * self._sfreq)
        fake_data = np.random.randn(config.N_CHANNELS, n_samples) * 50e-6
        fake_timestamps = np.linspace(0, config.EPOCH_LENGTH, n_samples)
        return fake_data, fake_timestamps

    def disconnect(self):
        print("Fake stream disconnected.")
