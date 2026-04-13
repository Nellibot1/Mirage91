#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from old.ClassifierVersion5 import (
    ACC_CHANNELS,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FEATURE_EEG_CHANNELS,
    LABEL_TO_NAME,
    NOTCH_FREQ,
    RANDOM_STATE,
    RUN_MARKER_TO_LABEL,
    STEP_SEC,
    TARGET_CHANNELS,
    TEST_SIZE,
    TRANSITION_PAD_SEC,
    USE_NOTCH,
    WINDOW_SEC,
    balance_by_participant_and_class,
    build_dataset_from_runs,
    describe_dataset,
    load_all_runs,
)


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
