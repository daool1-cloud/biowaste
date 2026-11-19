"""Four pretrained protein-language-model adapters.

IMPORTANT: these are reference inference implementations. Validate their exact
training-time tokenization, windowing, pooling and checkpoint revisions before
interpreting experimental predictions. No simulated embeddings are used.
"""
from __future__ import annotations
import gc
import os
import re
import time
# BIOWASTE_LOCAL_MODULE_PATH
import sys
from pathlib import Path

_APP_DIR = str(Path(__file__).resolve().parent)

if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Load the local ProtT5 module before worker threads start.
import legacy_prott5

from gpu_control import GPUPacer, GPU_FORWARD_SLOT
from typing import Callable

import numpy as np

PLM_CONFIG = {
    'esmc': {'name':'ESM-C', 'dim':1152, 'context':2048, 'overlap':512, 'model':'esmc_600m'},
    'esm2': {'name':'ESM-2', 'dim':1280, 'context':1024, 'overlap':256, 'model':'facebook/esm2_t33_650M_UR50D'},
    'prott5': {'name':'ProtT5', 'dim':1024, 'context':1024, 'overlap':256, 'model':'Rostlab/prot_t5_xl_half_uniref50-enc'},
    'protbert': {'name':'ProtBERT', 'dim':1024, 'context':1024, 'overlap':256, 'model':'Rostlab/prot_bert'},
}

def clean_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _window_starts(length, window_size, overlap):
    if length <= window_size:
        return [0]
    step = window_size - overlap
    if step <= 0:
        raise ValueError('Window overlap must be smaller than window length')
    starts = list(range(0, length - window_size + 1, step))
    final = length - window_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _window_weights(n, overlap):
    weights = np.ones(n, dtype=np.float32)
    length = min(overlap, n // 2)
    if length:
        ramp = np.arange(1, length + 1, dtype=np.float32) / (length + 1)
        weights[:length] = ramp
        weights[-length:] = ramp[::-1]
    return weights


def _window_items(records, residue_limit, overlap):
    """Yield (index, fragment, normalized residue weights) in input order."""
    for i, rec in enumerate(records):
        seq = rec.sequence
        starts = _window_starts(len(seq), residue_limit, overlap)
        coverage = np.zeros(len(seq), dtype=np.float32)
        prepared = []
        for start in starts:
            frag = seq[start:start + residue_limit]
            raw = _window_weights(len(frag), overlap)
            coverage[start:start + len(frag)] += raw
            prepared.append((start, frag, raw))
        if (coverage <= 0).any():
            raise RuntimeError('Window coverage is incomplete')
        for start, frag, raw in prepared:
            normalized = raw / coverage[start:start+len(frag)]
            yield i, frag, normalized / len(seq)


class HFEmbedder:
    def __init__(self, key, device):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.cfg = PLM_CONFIG[key]
        self.key = key
        self.device = torch.device(device)
        self.pacer = GPUPacer(device)
        self.model_id = os.getenv('BIOWASTE_'+key.upper()+'_MODEL_ID', self.cfg['model'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, do_lower_case=False)
        self.special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        self.context = int(os.getenv('BIOWASTE_'+key.upper()+'_CONTEXT',str(self.cfg['context'])))
        self.overlap = int(os.getenv('BIOWASTE_'+key.upper()+'_OVERLAP',str(self.cfg['overlap'])))
        if not 0 <= self.overlap < self.context-self.special_tokens:
            raise ValueError('Invalid window overlap or context setting')
        self.residue_limit = self.context - self.special_tokens
        self.model = AutoModel.from_pretrained(
            self.model_id, low_cpu_mem_usage=False
        )
        meta_tensors = [
            name for name, param in self.model.named_parameters()
            if param.is_meta
        ]
        meta_tensors += [
            "buffer:" + name for name, buf in self.model.named_buffers()
            if buf.is_meta
        ]
        if meta_tensors:
            raise RuntimeError(
                f"{self.key}: model contains meta tensors: {meta_tensors[:10]}"
            )
        self.model = self.model.to(self.device).eval()
        if int(self.model.config.hidden_size) != self.cfg['dim']:
            raise ValueError('Loaded PLM has the wrong embedding dimension')
        self.batch_size = int(os.getenv('BIOWASTE_EMBED_BATCH', '8'))
        self.max_batch_tokens = int(os.getenv('BIOWASTE_MAX_BATCH_TOKENS','4096'))

    def _encode(self, fragments):
        with GPU_FORWARD_SLOT:
            return self._encode_locked(fragments)

    def _encode_locked(self, fragments):
        import torch
        batch_start = time.monotonic()
        if self.key == 'protbert':
            inputs = [' '.join(re.sub(r'[UZOB]', 'X', f)) for f in fragments]
        else:
            inputs = list(fragments)
        encoded = self.tokenizer(
            inputs, padding=True, return_tensors='pt',
            return_attention_mask=True, return_special_tokens_mask=True,
            truncation=False, add_special_tokens=True,
        )
        special = encoded.pop('special_tokens_mask').bool()
        if encoded['input_ids'].shape[1] > self.context:
            raise RuntimeError('Tokenized protein exceeds configured context')
        attention = encoded['attention_mask'].bool()
        model_input = {k:v.to(self.device) for k,v in encoded.items()}
        with torch.inference_mode():
            out = self.model(**model_input).last_hidden_state
        if out.shape[-1] != self.cfg['dim']:
            raise ValueError('Model hidden dimension mismatch')
        result = []
        for j, frag in enumerate(fragments):
            mask = attention[j] & ~special[j]
            if int(mask.sum()) != len(frag):
                raise RuntimeError(f'Token/residue mismatch: {int(mask.sum())} tokens for {len(frag)} residues')
            result.append(out[j, mask.to(out.device)].float().cpu().numpy())
        self.pacer.after_batch(batch_start)
        return result

    def embed_many(self, records, progress: Callable[[int,int],None]):
        import torch
        sums = np.zeros((len(records), self.cfg['dim']), dtype=np.float64)
        processed = 0
        pending = []
        tokens = 0
        def flush():
            nonlocal processed, pending, tokens
            if not pending:
                return
            try:
                vectors = self._encode([r[1] for r in pending])
            except torch.cuda.OutOfMemoryError:
                if len(pending) == 1:
                    raise RuntimeError('GPU out of memory on a single window; reduce context length')
                clean_gpu()
                mid = len(pending) // 2
                sub = [pending[:mid], pending[mid:]]
                vectors = []
                for group in sub:
                    # Retry each smaller group; caller may lower BIOWASTE_EMBED_BATCH.
                    vectors.extend(self._encode([r[1] for r in group]))
            for (i,frag,weights), residue in zip(pending, vectors):
                sums[i] += np.sum(residue.astype(np.float64)*weights[:,None], axis=0)
                processed += 1
                if processed % 100 == 0:
                    progress(min(len(records), i+1), len(records))
            pending, tokens = [], 0
        # Preserve input order; batch by padded token count.
        for entry in _window_items(records, self.residue_limit, self.overlap):
            expected = len(entry[1])+self.special_tokens
            proposed = max(tokens, expected)
            if pending and (len(pending)>=self.batch_size or proposed*(len(pending)+1)>self.max_batch_tokens):
                flush()
            pending.append(entry)
            tokens = max(tokens,expected)
        flush()
        progress(len(records),len(records))
        result = sums.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Embedding contains NaN or Inf')
        return result

    def close(self):
        self.model.cpu()
        del self.model, self.tokenizer
        clean_gpu()


class ESMCEmbedder:
    def __init__(self, device):
        import torch
        from esm.models.esmc import ESMC
        self.cfg = PLM_CONFIG['esmc']
        self.device = torch.device(device)
        self.pacer = GPUPacer(device)
        self.context = int(os.getenv('BIOWASTE_ESMC_CONTEXT',str(self.cfg['context'])))
        self.overlap = int(os.getenv('BIOWASTE_ESMC_OVERLAP',str(self.cfg['overlap'])))
        model_id = os.getenv('BIOWASTE_ESMC_MODEL_ID', 'esmc_600m')
        # IMPORTANT: Let the ESM-C SDK place checkpoint weights directly on
        # the requested device. Calling .to() afterwards can fail if the SDK
        # leaves meta-only tensors (e.g. unmaterialized buffers) behind.
        # NEVER use .to_empty() on a pretrained model as a blanket workaround:
        # it creates uninitialized weights and invalid embeddings.
        self.model = ESMC.from_pretrained(model_id, device=self.device)

        meta_parameters = [
            name for name, value in self.model.named_parameters()
            if value.is_meta
        ]
        meta_buffers = [
            name for name, value in self.model.named_buffers()
            if value.is_meta
        ]
        if meta_parameters or meta_buffers:
            raise RuntimeError(
                'ESM-C pretrained checkpoint still contains meta tensors after '
                'loading: parameters=' + repr(meta_parameters[:12]) +
                ', buffers=' + repr(meta_buffers[:12]) + '. '
                'Check the installed esm package version and pretrained '
                'checkpoint loading; do not replace weights with to_empty().'
            )
        print('[ESM-C] Pretrained weights loaded on',
              next(self.model.parameters()).device, 'without meta tensors', flush=True)
        self.model.eval()
        # ESM-C adds BOS/EOS (verify with the version used in your training).
        self.residue_limit = self.context - 2

    def _encode(self, sequence):
        with GPU_FORWARD_SLOT:
            return self._encode_locked(sequence)

    def _encode_locked(self, sequence):
        import torch
        batch_start = time.monotonic()
        from esm.sdk.api import ESMProtein, LogitsConfig
        protein = ESMProtein(sequence=sequence)
        with torch.inference_mode():
            representation = self.model.encode(protein)
            out = self.model.logits(representation, LogitsConfig(sequence=True, return_embeddings=True))
        residue = out.embeddings
        if residue.ndim == 3:
            residue = residue[0]
        if residue.shape[0] == len(sequence)+2:
            residue = residue[1:-1]
        elif residue.shape[0] != len(sequence):
            raise ValueError(f'ESM-C embedding token count {residue.shape[0]} unexpected for {len(sequence)} residues')
        if residue.shape[-1] != 1152:
            raise ValueError('ESM-C embedding dimension must be 1152')
        result = residue.float().cpu().numpy()
        self.pacer.after_batch(batch_start)
        return result

    def embed_many(self, records, progress):
        sums = np.zeros((len(records),1152),dtype=np.float64)
        n_seen = 0
        previous_index = -1
        for i, frag, weights in _window_items(records,self.residue_limit,self.overlap):
            residue = self._encode(frag)
            sums[i] += np.sum(residue.astype(np.float64)*weights[:,None],axis=0)
            if i != previous_index:
                n_seen += 1
                previous_index = i
                if n_seen % 50 == 0:
                    progress(n_seen,len(records))
        progress(len(records),len(records))
        result = sums.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('ESM-C embeddings contain NaN or Inf')
        return result

    def close(self):
        self.model.cpu()
        del self.model
        clean_gpu()


class ProtT5Embedder:
    """Reuse the user's previously tested ProtT5 per-residue windowing code."""
    def __init__(self, device):
        import torch
        from transformers import T5EncoderModel, T5Tokenizer
        from legacy_prott5 import BiowasteInferenceEngine
        class EncoderOnly(BiowasteInferenceEngine):
            def __init__(self, requested_device):
                self.device = torch.device(requested_device)
                self.pacer = GPUPacer(requested_device)
                self.embedding_dim = 1024
                self.hf_model_id = os.getenv('BIOWASTE_PROTT5_MODEL_ID','Rostlab/prot_t5_xl_half_uniref50-enc')
                self.context_tokens = int(os.getenv('BIOWASTE_PROTT5_CONTEXT','1024'))
                self.overlap = int(os.getenv('BIOWASTE_PROTT5_OVERLAP','256'))
                self.max_batch_tokens = int(os.getenv('BIOWASTE_MAX_BATCH_TOKENS','4096'))
                self.max_batch_sequences = int(os.getenv('BIOWASTE_EMBED_BATCH','8'))
                try:
                    self.tokenizer = T5Tokenizer.from_pretrained(self.hf_model_id, legacy=True, do_lower_case=False)
                except TypeError:
                    self.tokenizer = T5Tokenizer.from_pretrained(self.hf_model_id, do_lower_case=False)
                self.special_token_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
                self.max_residues = self.context_tokens - self.special_token_count
                if not 0 <= self.overlap < self.max_residues:
                    raise ValueError('Invalid ProtT5 overlap or context')
                dtype = torch.float16 if self.device.type=='cuda' else torch.float32
                self.prott5 = T5EncoderModel.from_pretrained(self.hf_model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
                self.prott5 = self.prott5.to(self.device).eval()
                if int(self.prott5.config.d_model) != self.embedding_dim:
                    raise ValueError('ProtT5 embedding dimension mismatch')

            def _encode_residue_batch(self, sequences):
                with GPU_FORWARD_SLOT:
                    batch_start = time.monotonic()
                    outputs = super()._encode_residue_batch(sequences)
                    self.pacer.after_batch(batch_start)
                    return outputs

        self.encoder=EncoderOnly(device)

    def embed_many(self, records, progress):
        from legacy_prott5 import ProteinRecord
        result = np.empty((len(records), 1024),dtype=np.float32)
        chunk_size = max(1, int(os.getenv('BIOWASTE_PROTT5_PROTEIN_GROUP','64')))
        for start in range(0,len(records),chunk_size):
            subset = records[start:start+chunk_size]
            legacy_records = [ProteinRecord(r.protein_id,r.sequence,len(r.sequence),r.replaced,0) for r in subset]
            embedded = self.encoder.embed_records(legacy_records,batch_size=self.encoder.max_batch_sequences)
            for offset,item in enumerate(embedded):
                result[start+offset] = item.vector.float().numpy()
            progress(min(len(records),start+len(subset)),len(records))
        if not np.isfinite(result).all():
            raise ValueError('ProtT5 embeddings contain NaN or Inf')
        return result

    def close(self):
        self.encoder.prott5.cpu()
        del self.encoder.prott5, self.encoder.tokenizer, self.encoder
        clean_gpu()


def make_embedder(key, device='cuda'):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no CUDA GPU is available. Set BIOWASTE_DEVICE=cpu to use CPU.')
    if key=='esmc': return ESMCEmbedder(device)
    if key=='prott5': return ProtT5Embedder(device)
    if key in ('esm2','protbert'): return HFEmbedder(key,device)
    raise KeyError(key)
