"""FASTA -> four protein PLMs -> four pooling cases -> 224 R/caret predictions.

Never trains models or fabricates probabilities. If any step fails, no final
Useful/Not Useful decision is emitted.
"""
from __future__ import annotations
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable

import numpy as np

from fasta_utils import parse_annotated_fasta
from embedders import make_embedder, PLM_CONFIG, clean_gpu
from ensemble_scoring import load_weights, score_predictions, classify, PRIMARY_SCORE, PRIMARY_SD, STRONG_SCORE, STRONG_SD
from gpu_control import GPU_SLOTS, R_SLOTS

ROOT = Path(os.getenv('BIOWASTE_MODELS_ROOT', str(Path(__file__).resolve().parent)))
EXPECTED_CASES = ('case1_all_proteins','case2_target_weight16','case3_target_weight32','case4_target_only')
EXPECTED_MODELS = (
    'Logistic_regression_GLMNET', 'C5_0_tree','Random_Forest_Ranger', 'SVM_linear','SVM_radial',
    'Gradient_Boosting_GBM','Random_Forest','LDA_PCA','PLS','Neural_network_nnet','MARS_earth',
    'Naive_Bayes','kNN','Decision_tree_RPART',
)
BUNDLES = {
    'esmc': ('01.chatbot_esmc','ESMC_600M_4cases_14algorithms.pth','ESMC_600M','esmc-caret-rds-in-torch-v1'),
    'esm2': ('02.chatbot_esm2','ESM2_t33_650M_4cases_14algorithms.pth','ESM2_t33_650M','esm2-caret-rds-in-torch-v1'),
    'prott5': ('03.chatbot_prott5','ProtT5_XL_U50_4cases_14algorithms.pth','ProtT5_XL_U50','prott5-caret-rds-in-torch-v1'),
    'protbert': ('04.chatbot_protbert','ProtBERT_UniRef100_4cases_14algorithms.pth','ProtBERT_UniRef100','protbert-caret-rds-in-torch-v1'),
}


def check_model_files():
    paths={key:ROOT / dirname / filename for key,(dirname,filename,_,_) in BUNDLES.items()}
    missing=[str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing PLM model bundles:\n' + '\n'.join(missing))
    return paths


def extract_bundle(key, dest):
    """Extract one trusted model bundle; verify SHA256 for every RDS before use."""
    import torch
    dirname,filename,plm_id,schema=BUNDLES[key]
    src=ROOT/dirname/filename
    # Only trusted local model artifacts should be loaded with torch.load.
    bundle=torch.load(str(src),map_location='cpu',weights_only=True)
    if (bundle.get('schema_version')!=schema or bundle.get('plm')!=plm_id or
        bundle.get('embedding_dim')!=PLM_CONFIG[key]['dim'] or bundle.get('n_models')!=56 or
        bundle.get('n_cases')!=4 or bundle.get('n_algorithms_per_case')!=14 or
        tuple(bundle.get('cases',()))!=EXPECTED_CASES or tuple(bundle.get('algorithms',()))!=EXPECTED_MODELS):
        raise ValueError(f'Model bundle metadata mismatch for {key}')
    if set(bundle['entries']) != set(EXPECTED_CASES):
        raise ValueError(f'Unexpected case entries in {key} model bundle')
    out=Path(dest)
    count=0
    for case in EXPECTED_CASES:
        folder=out/case
        folder.mkdir(parents=True, exist_ok=True)
        if set(bundle['entries'][case])!=set(EXPECTED_MODELS):
            raise ValueError(f'Wrong algorithms in {key}/{case}')
        for algorithm in EXPECTED_MODELS:
            entry=bundle['entries'][case][algorithm]
            raw=entry['bytes'].cpu().numpy().tobytes()
            if (len(raw)!=int(entry['size_bytes']) or
                hashlib.sha256(raw).hexdigest()!=entry['sha256'] or
                entry['filename']!=algorithm+'.rds'):
                raise ValueError(f'Corrupt trained model: {key}/{case}/{algorithm}')
            (folder/(algorithm+'.rds')).write_bytes(raw)
            count+=1
    del bundle
    gc.collect()
    if count!=56:
        raise RuntimeError('Wrong number of extracted R models')


def pool_cases(vectors,records):
    """Per-protein embeddings mean-weighted with ×1, ×16, ×32, target-only.

    R input CSV reproduces the historical '.9g' serialization; confirm the
    actual legacy pooling implementation against a known training proteome.
    """
    vectors=np.asarray(vectors,dtype=np.float64)
    target=np.array([r.is_target for r in records],dtype=bool)
    if vectors.ndim!=2 or vectors.shape[0]!=len(records) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding matrix')
    if not target.any():
        raise ValueError('No target enzymes; cannot calculate Case 4')
    output={}
    for case,multiplier in zip(EXPECTED_CASES,(1,16,32,None)):
        if multiplier is None:
            # Case 4: only proteins annotated as CAZyme/lipase/protease.
            selected=vectors[target]
            pooled=selected.mean(axis=0)
        else:
            w=np.where(target,float(multiplier),1.0)
            pooled=np.sum(vectors*w[:,None],axis=0)/w.sum()
        # Training CSVs wrote embeddings with nine significant digits.
        output[case]=np.array([float(format(float(v),'.9g')) for v in pooled],dtype=np.float64)
    return output


def save_embedding_csvs(vectors,records,analysis_id,embedding_dir):
    embedded=pool_cases(vectors,records)
    embedding_dir=Path(embedding_dir)
    embedding_dir.mkdir(parents=True,exist_ok=True)
    for case,vec in embedded.items():
        dest=embedding_dir/(case+'.csv')
        with dest.open('w',newline='') as out:
            writer=csv.writer(out)
            writer.writerow(['assembly']+[f'emb_{i:04d}' for i in range(1,len(vec)+1)])
            writer.writerow([analysis_id]+[format(float(v),'.9g') for v in vec])
    return embedded


def _load_predictions(path,key,analysis_id):
    with open(path,newline='') as f:
        predictions=list(csv.DictReader(f))
    _,_,plm_key,_=BUNDLES[key]
    if len(predictions)!=56:
        raise ValueError(f'{key}: expected 56 predictions, received {len(predictions)}')
    actual={(row['Case'],row['Model']) for row in predictions}
    expected={(case,model) for case in EXPECTED_CASES for model in EXPECTED_MODELS}
    if actual!=expected or any(row['assembly']!=analysis_id or row['PLM']!=plm_key for row in predictions):
        raise ValueError(f'{key}: prediction identities do not match the requested model and strain')
    for row in predictions:
        row['probability_positive']=float(row['probability_positive'])
        row['probability_threshold']=float(row['probability_threshold'])
        p=row['probability_positive']
        if not np.isfinite(p) or p<0 or p>1:
            raise ValueError(f'{key}: invalid predicted probability')
        if row['predicted'] != ('positive' if p>=row['probability_threshold'] else 'negative'):
            raise ValueError(f'{key}: inconsistent predicted class')
    return predictions


def _write_prediction_csv(path,rows):
    fields=('assembly','PLM','Case','Model','probability_positive','predicted','probability_threshold')
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, summary, weight_file=None):
    """Paper-aligned ensemble: performance-weighted score + unweighted sample SD."""
    weights, weights_sha = check_model_weights(weight_file)
    score, sd, contributions = score_predictions(rows, weights)
    p = np.array([r['probability_positive'] for r in rows], dtype=np.float64)
    vote = np.array([r['predicted'] == 'positive' for r in rows], dtype=bool)
    by_plm = []
    for key in BUNDLES:
        subset = [r for r in rows if r['PLM'] == BUNDLES[key][2]]
        scores = np.array([r['probability_positive'] for r in subset])
        plm_w = np.array([weights[(r['PLM'], r['Case'], r['Model'])] for r in subset])
        by_plm.append({
            'plm': PLM_CONFIG[key]['name'], 'models': len(subset),
            'mean_positive_probability': float(scores.mean()),
            'positive_votes': sum(r['predicted'] == 'positive' for r in subset),
            'weighted_score_within_plm': float(np.dot(scores, plm_w)/plm_w.sum()),
            'prediction_sd_within_plm': float(scores.std(ddof=1))
        })
    label = classify(score, sd)
    return {
        'analysis_id': summary['analysis_id'],
        'predicted_label': label,
        'predicted_label_ko': {'Strongly Recommended': '강력추천',
                               'Useful': 'Useful', 'Not Useful': 'Not Useful'}[label],
        'weighted_suitability_score': score,
        'prediction_sd': sd,
        'sd_definition': 'Unweighted sample SD across 224 positive-class probabilities (ddof=1)',
        'weight_formula': 'max(CV_ROC_best,0.01) * max(Test_MCC,0.01) * max(Test_Balanced_Accuracy,0.01)',
        'weight_source_sha256': weights_sha,
        'screening_thresholds': {'weighted_score_min': PRIMARY_SCORE, 'sd_max_inclusive': PRIMARY_SD},
        'strong_thresholds': {'weighted_score_min': STRONG_SCORE, 'sd_max_exclusive': STRONG_SD},
        'positive_votes': int(vote.sum()), 'total_models': 224,
        'positive_vote_ratio': float(vote.mean()),
        'mean_positive_probability': float(p.mean()),
        'plm_results': by_plm, 'protein_summary': summary,
        '_contributions': contributions,
        'decision_note': ('Research-derived prioritization, not experimentally measured '
                          'food-waste degradation; Not Useful means thresholds were not met.')
    }


def check_model_weights(weight_file=None):
    """Fail fast before lengthy GPU inference if any of 224 weights is unusable."""
    expected = {(BUNDLES[key][2], case, model)
                for key in BUNDLES for case in EXPECTED_CASES for model in EXPECTED_MODELS}
    return load_weights(weight_file or os.getenv('BIOWASTE_WEIGHTS_FILE',
            str(Path(__file__).with_name('model_weights_224.csv'))), expected)


class BiowasteInferenceEngine:
    def __init__(self, device=None,rscript=None):
        self.device=device or os.getenv('BIOWASTE_DEVICE','cuda')
        self.rscript=rscript or os.getenv('BIOWASTE_RSCRIPT','Rscript')
        self.r_code=Path(__file__).with_name('r_predict_all.R')

    def run(self,fasta_path,output_dir,progress:Callable[[str],None]|None=None):
        progress=progress or (lambda _:None)
        check_model_files()
        check_model_weights()  # Fail early if ensemble metadata are missing or inconsistent.
        records,summary=parse_annotated_fasta(fasta_path)
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        result_rows=[]
        progress(f"Parsed {summary['total_proteins']:,} proteins; {summary['target_proteins']:,} target enzymes.")
        # Sequential processing limits VRAM. A model is unloaded before the next is loaded.
        for key in BUNDLES:
            name=PLM_CONFIG[key]['name']
            progress(f'Waiting for GPU allocation for {name}...')
            # GPU phases may overlap up to BIOWASTE_GPU_MAX_JOBS.
            # Each request retains its own model object and output directory.
            with GPU_SLOTS:
                progress(f'Loading {name} protein language model...')
                embedder = None
                try:
                    embedder = make_embedder(key, self.device)
                    progress(f'Analyzing with {name} — generating protein embeddings...')
                    def on_progress(done,total):
                        progress(f'Analyzing with {name} — embedding proteins ({done:,}/{total:,})...')
                    vectors = embedder.embed_many(records, on_progress)
                    if vectors.shape != (len(records), PLM_CONFIG[key]['dim']):
                        raise ValueError(f'{name} embedding dimension mismatch')
                finally:
                    try:
                        if embedder is not None:
                            embedder.close()
                            del embedder
                    finally:
                        clean_gpu()
            progress(f'Analyzing with {name} — pooling four cases...')
            emb_dir=output_dir/'embeddings'/key
            save_embedding_csvs(vectors,records,summary['analysis_id'],emb_dir)
            del vectors
            progress(f'Analyzing with {name} — evaluating 14 classifiers across four cases...')
            out_csv=output_dir/f'predictions_{key}.csv'
            progress(f'Waiting for CPU/R allocation for {name}...')
            with R_SLOTS:
                with tempfile.TemporaryDirectory(prefix=f'biowaste_{key}_models_') as tdir:
                    extract_bundle(key,tdir)
                    cmd=[self.rscript,'--vanilla',str(self.r_code),
                         '--model-root',tdir,'--embedding-dir',str(emb_dir),
                         '--out',str(out_csv),'--plm-key',BUNDLES[key][2],
                         '--embedding-dim',str(PLM_CONFIG[key]['dim'])]
                    try:
                        proc=subprocess.run(cmd,check=True,text=True,capture_output=True,
                                            timeout=int(os.getenv('BIOWASTE_R_TIMEOUT','3600')))
                    except subprocess.CalledProcessError as exc:
                        raise RuntimeError(f'R/caret prediction failed for {name}: {(exc.stderr or exc.stdout)[-2200:]}') from exc
            model_rows=_load_predictions(out_csv,key,summary['analysis_id'])
            result_rows.extend(model_rows)
            progress(f'{name} complete: {len(model_rows)} predictions.')
        progress('Combining results from all 224 classifiers...')
        report=aggregate(result_rows,summary)
        report['predictions_csv_path']=str(output_dir/'all_224_predictions.csv')
        report['report_json_path']=str(output_dir/'report.json')
        report['weighted_contributions_csv_path']=str(output_dir/'weighted_contributions.csv')
        contributions = report.pop('_contributions')
        with open(report['weighted_contributions_csv_path'],'w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(contributions[0]))
            writer.writeheader()
            writer.writerows(contributions)
        _write_prediction_csv(report['predictions_csv_path'],result_rows)
        with open(report['report_json_path'],'w') as f:
            json.dump(report,f,indent=2)
        return report
