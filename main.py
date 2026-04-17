# main.py

import collections
import time

import config
from classifier import Classifier
from controller import Controller
from feature_extractor import FeatureExtractor
from preprocessor import preprocess
from stream_receiver import FakeStreamReceiver, StreamReceiver


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


class CommandDetector:
    def __init__(self):
        self.current_label = None
        self.current_run_length = 0
        self.last_blink_burst_time = None
        self.last_command_time = {}
        self.clench_emitted_in_run = False

    def update(self, smoothed_label, event_time):
        emitted = []

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
            if self.current_run_length >= config.MIN_BURST_WINDOWS:
                if self._cooldown_ready(config.CLENCH_COMMAND, event_time):
                    emitted.append((config.CLENCH_COMMAND, "clench"))
                    self.last_command_time[config.CLENCH_COMMAND] = event_time
                self.clench_emitted_in_run = True

        return emitted

    def _finalize_run(self, event_time):
        if self.current_label != 1 or self.current_run_length < config.MIN_BURST_WINDOWS:
            return []

        emitted = []
        if self.last_blink_burst_time is not None:
            gap = event_time - self.last_blink_burst_time
            if config.DOUBLE_BLINK_MIN_GAP_SEC <= gap <= config.DOUBLE_BLINK_MAX_GAP_SEC:
                if self._cooldown_ready(config.BLINK_COMMAND, event_time):
                    emitted.append((config.BLINK_COMMAND, "double_blink"))
                    self.last_command_time[config.BLINK_COMMAND] = event_time
                self.last_blink_burst_time = None
                return emitted

        self.last_blink_burst_time = event_time
        return emitted

    def _cooldown_ready(self, command, now_sec):
        previous = self.last_command_time.get(command)
        return previous is None or (now_sec - previous) >= config.COMMAND_COOLDOWN_SEC


def main():
    use_fake_stream = False
    controller_mode = "socket"

    receiver = FakeStreamReceiver() if use_fake_stream else StreamReceiver()
    extractor = FeatureExtractor()
    classifier = Classifier()
    controller = Controller(mode=controller_mode)

    receiver.connect()
    channel_names = receiver.get_channel_names()
    sfreq = receiver.get_sfreq()

    extractor.sfreq = sfreq
    extractor.load()
    classifier.load()

    prediction_history = collections.deque(maxlen=config.SMOOTHING_WINDOW)
    detector = CommandDetector()
    stream_start_time = None

    print()
    print("=== Online BCI Running ===")
    print(f"Channels : {config.TARGET_CHANNELS}")
    print(f"Window   : {config.EPOCH_LENGTH}s  |  Step: {config.SLIDE_STEP}s")
    print(f"Smoothing: majority vote over last {config.SMOOTHING_WINDOW} predictions")
    print(f"Classes  : {config.CLASS_LABELS}")
    print(f"Commands : blink->{config.BLINK_COMMAND}, clench->{config.CLENCH_COMMAND}")
    print("Press Ctrl+C to stop")
    print()

    try:
        while True:
            raw_data, timestamps = receiver.get_chunk()
            if timestamps is None or len(timestamps) == 0:
                time.sleep(config.SLIDE_STEP)
                continue

            if stream_start_time is None:
                stream_start_time = float(timestamps[0])

            filtered_data = preprocess(raw_data, stream_ch_names=channel_names, sfreq=sfreq)
            features = extractor.extract(filtered_data)

            raw_class, _raw_label = classifier.predict(features)
            prediction_history.append(raw_class)
            smoothed_class = majority_vote(prediction_history)
            smoothed_label = config.CLASS_LABELS.get(smoothed_class, "UNKNOWN")

            controller.send_prediction(smoothed_class, smoothed_label)

            event_time = float(timestamps[-1]) - stream_start_time
            emitted = detector.update(smoothed_class, event_time)
            for command, source in emitted:
                controller.send_event(command, source, smoothed_class, smoothed_label)

            time.sleep(config.SLIDE_STEP)

    except KeyboardInterrupt:
        print()
        print("Stopping BCI...")
    finally:
        receiver.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
