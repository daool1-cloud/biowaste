"""Paper-aligned 224-model ensemble scoring, strict 1:1 join by model identity."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np


PRIMARY_SCORE = 0.70
PRIMARY_SD = 0.40
STRONG_SCORE = 0.80
STRONG_SD = 0.15


def calculate_weight(cv_auc: float, test_mcc: float, test_ba: float) -> float:
    values = (float(cv_auc), float(test_mcc), float(test_ba))
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite CV AUC / test MCC / test balanced accuracy')
    return math.prod(max(value, 0.01) for value in values)


def classify(score: float, sd: float) -> str:
    if not math.isfinite(score) or not math.isfinite(sd):
        raise ValueError('Non-finite suitability score or SD')
    # A weighted mean of identical probabilities (e.g., 0.80) can round to
    # 0.7999999999999999. Do not misclassify that exact intended boundary.
    at_least = lambda value, threshold: (value >= threshold or
                                         math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12))
    if at_least(score, STRONG_SCORE) and sd < STRONG_SD:
        return 'Strongly Recommended'
    if at_least(score, PRIMARY_SCORE) and (sd <= PRIMARY_SD or
                                           math.isclose(sd, PRIMARY_SD, rel_tol=0, abs_tol=1e-12)):
        return 'Useful'
    return 'Not Useful'


def load_weights(path, expected):
    """Return (key->weight, sha256); fail closed on missing/extra/duplicate keys."""
    path = Path(path)
    expected = set(expected)
    result = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = {'PLM', 'Case', 'Model', 'CV_AUC', 'Test_MCC',
                  'Test_Balanced_Accuracy', 'model_weight'}
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise ValueError('Missing required model weight columns: '+repr(fields))
        for row in reader:
            key = (row['PLM'], row['Case'], row['Model'])
            if key in result:
                raise ValueError('Duplicate model weight key: '+repr(key))
            reference_weight = calculate_weight(
                row['CV_AUC'], row['Test_MCC'], row['Test_Balanced_Accuracy'])
            supplied = float(row['model_weight'])
            if not math.isfinite(supplied) or supplied <= 0 or not math.isclose(
                supplied, reference_weight, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError('Weight mismatch: '+repr(key))
            result[key] = supplied
    if set(result) != expected:
        missing, extra = expected-set(result), set(result)-expected
        raise ValueError('224-model weight identities mismatch: missing=%r extra=%r' %
                         (sorted(missing)[:5], sorted(extra)[:5]))
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


def score_predictions(rows, model_weights):
    """A weighted positive probability; unweighted sample SD across 224 models."""
    rows = list(rows)
    actual_keys = [(r['PLM'],r['Case'],r['Model']) for r in rows]
    if len(actual_keys) != 224 or len(set(actual_keys)) != 224 or set(actual_keys) != set(model_weights):
        raise ValueError('Prediction rows and 224 model weights do not match 1:1')
    p = np.array([float(r['probability_positive']) for r in rows], dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Prediction probabilities must be finite within [0,1]')
    w = np.array([model_weights[key] for key in actual_keys], dtype=np.float64)
    weighted_score = float(np.dot(p, w) / w.sum())
    sd = float(p.std(ddof=1))  # R::sd uses sample SD, N-1.
    contributions = []
    for r, wi, pi in zip(rows, w, p):
        contributions.append({
            'assembly': r['assembly'], 'PLM': r['PLM'], 'Case': r['Case'],
            'Model': r['Model'], 'probability_positive': float(pi),
            'model_weight': float(wi), 'normalized_weight': float(wi/w.sum()),
            'weighted_contribution': float(wi*pi/w.sum())
        })
    return weighted_score, sd, contributions
