#!/usr/bin/env python3
"""Build a strict, reproducible 224-model weight table from research metrics."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from ensemble_scoring import calculate_weight
from pipeline import BUNDLES, EXPECTED_CASES, EXPECTED_MODELS


def build(source, output):
    expected = {(BUNDLES[key][2], case, model)
                for key in BUNDLES for case in EXPECTED_CASES for model in EXPECTED_MODELS}
    selected = {}
    with open(source, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = (row['PLM_Key'], row['Case'], row['Model'])
            if key not in expected:
                continue  # Excludes the additional GLM_PCA algorithm (16 rows).
            if key in selected:
                raise ValueError('Duplicate PLM/Case/Model performance row: '+repr(key))
            if row['Status'] != 'OK':
                raise ValueError('Performance row has non-OK status: '+repr(key))
            weight = calculate_weight(row['CV_ROC_best'], row['Test_MCC'],
                                      row['Test_Balanced_Accuracy'])
            selected[key] = {
                'PLM': key[0], 'Case': key[1], 'Model': key[2],
                'CV_AUC': row['CV_ROC_best'], 'Test_MCC': row['Test_MCC'],
                'Test_Balanced_Accuracy': row['Test_Balanced_Accuracy'],
                'model_weight': format(weight, '.17g')
            }
    if set(selected) != expected:
        raise ValueError('Missing models in source file: '+repr(sorted(expected-set(selected))[:10]))
    keys = sorted(expected)
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[keys[0]]))
        writer.writeheader()
        writer.writerows(selected[k] for k in keys)
    print(f'Wrote exactly {len(keys)} model weights: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='?', default='all_plm_case_model_scores.csv')
    parser.add_argument('output', nargs='?', default='model_weights_224.csv')
    args = parser.parse_args()
    build(args.source, args.output)
