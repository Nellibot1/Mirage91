# Real-Time EEG Control

This folder now includes a small online inference path that reuses the trained SVM model in `artifacts/`.

## Files

- `realtime_eeg_inference.py`: Connects to an EEG LSL stream, extracts the same feature set as training, runs the trained model in real time, smooths predictions, and emits UDP game commands.
- `game_command_listener.py`: Minimal UDP listener you can use to test commands before wiring them into a game engine.
- `eeg_test_game.py`: Small local jump/duck game for end-to-end testing without Unity.
- `UnityEegCommandReceiver.cs`: Unity MonoBehaviour that listens for UDP EEG commands and raises UnityEvents for duck and action.

## Expected Signal Path

1. EEG device publishes an LSL stream with the training channels: `F3,F4,C3,Cz,C4,P3,P4`
2. `realtime_eeg_inference.py` keeps a rolling 0.5 second window and classifies every 0.1 seconds.
3. A double blink emits the blink command.
4. A clench burst emits the clench command.
5. Commands are sent as UDP JSON packets to your game.

## Install

The runtime needs at least:

```bash
pip install numpy scipy joblib scikit-learn pylsl pygame
```

If you want the runtime environment to match training exactly, use the same scikit-learn version that produced the model artifact. The current saved model warns about a version mismatch when loaded in this workspace.

## Test Locally

Start the UDP listener:

```bash
python3 game_command_listener.py
```

Or launch the local test game:

```bash
python3 eeg_test_game.py
```

Run the online classifier:

```bash
python3 realtime_eeg_inference.py --stream-type EEG --print-predictions
```

If your LSL stream does not include channel labels, pass them explicitly:

```bash
python3 realtime_eeg_inference.py \
  --stream-name "MyEEGStream" \
  --channel-names "F3,F4,C3,Cz,C4,P3,P4"
```

## Game Integration

By default, the script emits UDP JSON like:

```json
{"command":"DUCK","time_sec":12.3,"source":"double_blink","timestamp":1712480000.0}
```

and

```json
{"command":"ACTION","time_sec":15.1,"source":"clench","timestamp":1712480002.0}
```

You can change the emitted commands:

```bash
python3 realtime_eeg_inference.py \
  --blink-command DUCK \
  --clench-command FIRE
```

## Unity Setup

1. Copy `UnityEegCommandReceiver.cs` into your Unity project's `Assets/Scripts/` folder.
2. Add the `UnityEegCommandReceiver` component to a GameObject in your scene.
3. In the Inspector, keep the port at `8765` unless you changed the Python sender.
4. Wire `On Duck` to your player's duck method.
5. Wire `On Action` to your player's action, attack, or interact method.
6. Start the Unity scene, then start `realtime_eeg_inference.py`.

If you want the game to treat clench as duck instead, either:

- change the Python sender to `--clench-command DUCK`, or
- change the Unity receiver's `Action Command` field to `DUCK`.

The receiver also exposes `On Any Command` if you want one script to handle everything manually.

## Full Real-Time Test Stack

For the full pipeline on your laptop, you typically need:

1. Your EEG device software running and publishing an LSL EEG stream.
2. Python 3 with:
   - `numpy`
   - `scipy`
   - `joblib`
   - `scikit-learn`
   - `pylsl`
   - `pygame` for the local test game
3. The trained model files in `artifacts/`.
4. Either:
   - `python3 eeg_test_game.py` for a lightweight local game, or
   - Unity with `UnityEegCommandReceiver.cs`

The test order is:

1. Start the EEG/LSL application for the headset.
2. Start `python3 eeg_test_game.py` or your Unity scene.
3. Start `python3 realtime_eeg_inference.py ...`
4. Confirm the game reacts to `DUCK` and `ACTION`.

## Notes


- It applies the same bandpass and notch settings from the saved metadata.
- The detector uses majority-vote smoothing plus burst logic so that it reacts to deliberate actions instead of single noisy windows.
