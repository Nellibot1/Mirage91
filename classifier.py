# classifier.py

import json
from pathlib import Path

import joblib
import numpy as np

import config


class Classifier:

    def __init__(self):
        self.model = None
        self.metadata = None
        self.is_loaded = False

    def load(self):
        model_path = Path(config.MODEL_PATH)
        metadata_path = Path(config.METADATA_PATH)

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Copy the v4.2 metadata JSON into the models folder first."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Copy the v4.2 .joblib artifact into the models folder first."
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self.model = joblib.load(model_path)
        self.is_loaded = True

        print(f"Model loaded from {model_path}")
        print(f"Metadata loaded from {metadata_path}")
        print(f"  Selected model: {self.metadata.get('selected_model')}")
        print(f"  Expected feature count: {len(self.metadata.get('feature_names', []))}")

        if hasattr(self.model, "steps"):
            step_names = [name for name, _ in self.model.steps]
            print(f"  Pipeline steps: {step_names}")

    def expected_feature_count(self):
        if not self.metadata:
            return None
        return len(self.metadata.get("feature_names", []))

    def predict(self, features):
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return 0, config.CLASS_LABELS[0]

        if features.ndim != 2 or features.shape[0] != 1:
            raise ValueError(
                f"Expected features shaped (1, n_features), got {features.shape}"
            )

        expected = self.expected_feature_count()
        if expected is not None and features.shape[1] != expected:
            raise ValueError(
                f"Feature length mismatch: model expects {expected}, got {features.shape[1]}"
            )

        if not self.is_loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")

        prediction = self.model.predict(features)
        predicted_class = int(prediction[0])
        label = config.CLASS_LABELS.get(predicted_class, "UNKNOWN")
        return predicted_class, label

    def predict_proba(self, features):
        if self.is_loaded and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)[0]
        return None
