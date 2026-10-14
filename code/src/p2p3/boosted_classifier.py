"""Low-latency calibrated classifier for the mobile demo.

The RM-Net remains the source of continuous intensity and explanations.  This
module adds a compact ordinal decision path for the phone: four robust temporal
statistics per modality, SVD compression and a multinomial logistic head.  The
two class biases are selected on the validation split only, matching the
project's existing threshold-freeze protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

MODALITIES = ("text", "audio", "vision")
CLASS_NAMES = np.array(["Negative", "Neutral", "Positive"])


def sequence_features(data: dict) -> np.ndarray:
    """Return fixed-size statistics while respecting per-modality masks."""
    out = []
    lengths = np.asarray(data["lengths"], dtype=np.int64)
    for j, name in enumerate(MODALITIES):
        x = np.asarray(data[name], dtype=np.float32)
        obs = np.asarray(data["obs"][:, j], dtype=bool)[..., None]
        den = obs.sum(1).clip(1)
        mean = (x * obs).sum(1) / den
        std = np.sqrt(((x - mean[:, None, :]) ** 2 * obs).sum(1) / den)
        first = x[:, 0]
        last_idx = np.clip(lengths - 1, 0, x.shape[1] - 1)[:, None, None]
        last = np.take_along_axis(x, last_idx, axis=1)[:, 0]
        out.extend((mean, std, first, last))
    return np.concatenate(out, axis=1).astype(np.float32, copy=False)


def _labels(y: np.ndarray) -> np.ndarray:
    return np.sign(np.asarray(y, dtype=np.float32)).astype(np.int64) + 1


def tune_bias(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Tune class offsets by a coarse grid followed by a local fine grid."""
    best = (-1.0, (0.0, 0.0))
    for step, radius in ((0.025, 2.0), (0.002, 0.05)):
        center = best[1]
        for neg in np.arange(center[0] - radius, center[0] + radius + step / 2, step):
            for pos in np.arange(center[1] - radius, center[1] + radius + step / 2, step):
                adjusted = scores.copy()
                adjusted[:, 0] += neg
                adjusted[:, 2] += pos
                f1 = _macro_f1(labels, adjusted.argmax(1))
                if f1 > best[0] + 1e-12:
                    best = (f1, (float(neg), float(pos)))
    return best[1]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values = []
    for c in range(3):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        values.append(float(2 * tp / denom) if denom else 0.0)
    return float(np.mean(values))


def fit(train: dict, valid: dict, output: str | Path) -> dict:
    """Fit and persist the phone model from prepared train/valid caches."""
    x_train = sequence_features(train)
    x_valid = sequence_features(valid)
    y_train = _labels(train["y"])
    y_valid = _labels(valid["y"])
    svd = TruncatedSVD(n_components=384, random_state=7, n_iter=5)
    z_train = svd.fit_transform(x_train)
    z_valid = svd.transform(x_valid)
    clf = LogisticRegression(C=0.003, solver="saga", max_iter=250, random_state=7)
    clf.fit(z_train, y_train)
    raw = clf.decision_function(z_valid)
    bias = tune_bias(raw, y_valid)
    adjusted = raw.copy()
    adjusted[:, 0] += bias[0]
    adjusted[:, 2] += bias[1]
    result = {
        "validation_macro_f1": _macro_f1(y_valid, adjusted.argmax(1)),
        "validation_accuracy": float(np.mean(adjusted.argmax(1) == y_valid)),
        "baseline_macro_f1": _macro_f1(y_valid, raw.argmax(1)),
        "bias": list(bias),
        "feature_count": int(x_train.shape[1]),
        "components": 384,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"svd": svd, "classifier": clf, "bias": np.asarray(bias)}, target)
    target.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def predict(model: dict, data: dict) -> dict:
    scores = model["classifier"].decision_function(model["svd"].transform(sequence_features(data)))
    adjusted = scores.copy()
    adjusted[:, 0] += model["bias"][0]
    adjusted[:, 2] += model["bias"][1]
    shifted = adjusted - adjusted.max(1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(1, keepdims=True)
    pred = probs.argmax(1)
    return {"class": CLASS_NAMES[pred], "class_id": pred, "probabilities": probs, "scores": adjusted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="outputs/problem2/cache/train.npz")
    parser.add_argument("--valid-cache", default="outputs/problem2/cache/valid.npz")
    parser.add_argument("--output", default="app_demo/model.joblib")
    args = parser.parse_args()
    train = dict(np.load(args.train_cache, allow_pickle=False))
    valid = dict(np.load(args.valid_cache, allow_pickle=False))
    print(json.dumps(fit(train, valid, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
