#!/usr/bin/env python3
"""Compare new FASTA->PLM->pooling against a known training strain's CSV rows.

Run before scientific use. Failure means embeddings are not training-compatible.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from fasta_utils import parse_annotated_fasta
from embedders import PLM_CONFIG, make_embedder, clean_gpu
from pipeline import pool_cases, EXPECTED_CASES

TRAINING_SUBDIRS={
 'esmc':'01.ESM_C/ESMC_600M',
 'esm2':'02.ESM_2/ESM2_t33_650M_UR50D',
 'prott5':'03.ProtT5/ProtT5_XL_U50',
 'protbert':'04.ProtBERT/ProtBERT_UniRef100',
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta',required=True,help='Previously annotated, known TRAINING proteome')
    p.add_argument('--plm',choices=list(PLM_CONFIG),required=True)
    p.add_argument('--training-base',default='/var2/daool1/..biodegradation/08.PLM')
    p.add_argument('--assembly',help='Assembly name to find in training CSV (defaults to FASTA header)')
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-abs',type=float,default=0.001,
                   help='Maximum allowed absolute embedding error; choose based on original numeric precision')
    args=p.parse_args()
    records,info=parse_annotated_fasta(args.fasta)
    assembly=args.assembly or info['analysis_id']
    embedder=make_embedder(args.plm,args.device)
    try:
        vectors=embedder.embed_many(records,lambda n,total:print(f'{args.plm}: {n}/{total} proteins',flush=True))
    finally:
        embedder.close()
        clean_gpu()
    cases=pool_cases(vectors,records)
    success=True
    root=Path(args.training_base)/TRAINING_SUBDIRS[args.plm]/'strain_embeddings/training'
    for case in EXPECTED_CASES:
        path=root/(case+'.csv')
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline='') as f:
            matching=[r for r in csv.DictReader(f) if r.get('assembly')==assembly]
        if len(matching)!=1:
            raise RuntimeError(f'Expected exactly one reference row for {assembly} in {path}; found {len(matching)}')
        expected=np.array([float(matching[0][f'emb_{i:04d}']) for i in range(1,len(cases[case])+1)])
        difference=np.abs(expected-cases[case]); max_abs=float(difference.max())
        rmse=float(np.sqrt(np.mean(difference**2)))
        passed=bool(np.isfinite(difference).all() and max_abs<=args.max_abs)
        print(f'{args.plm} {case}: max_abs_error={max_abs:.8g}, RMSE={rmse:.8g}, PASS={passed}')
        success=success and passed
    if not success:
        raise SystemExit('PARITY FAILED: training-time embedding / annotation / pooling settings may differ.')
    print('All four cases passed. Note that this tests ONE reference proteome; validate multiple strains and long sequences.')

if __name__=='__main__': main()
