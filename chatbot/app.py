"""Chainlit UI: concurrent user analyses, bounded GPU/R capacity, paper scoring."""
import asyncio
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import uuid

import chainlit as cl

from gpu_control import async_gpu_slot
from pipeline import BiowasteInferenceEngine, check_model_files, check_model_weights
from llm import GemmaChatEngine

for ext in ('.faa', '.fa', '.fasta'):
    mimetypes.add_type('text/plain', ext)

log = logging.getLogger('biowaste')
ROOT = Path(os.getenv('BIOWASTE_OUTPUT_ROOT', '/tmp/biowaste_web_results'))
ROOT.mkdir(parents=True, exist_ok=True)
ANALYSIS_SLOTS = asyncio.Semaphore(int(os.getenv('BIOWASTE_MAX_ACTIVE_ANALYSES', '4')))
CHAT_LOCK = asyncio.Lock()  # GemmaChatEngine is a shared, mutable model instance.
engine = BiowasteInferenceEngine()
llm = GemmaChatEngine()


@cl.on_chat_start
async def start():
    cl.user_session.set('history', [])
    cl.user_session.set('analysis_running', False)
    await cl.Message(content=(
        '# Biowaste Microbial Suitability Predictor\n\n'
        'Upload **one strain-level annotated protein FASTA** (`.faa`, `.fa`, `.fasta`). '
        'FASTA headers should contain `|CAZyme|`, `|lipase|`, `|protease|`, or `|other|`. '
        'Example: `>GCF_028891345.1_WP_004087316.1|lipase|GCF_028891345.1|`\n\n'
        'Four PLMs (ESM-C, ESM-2, ProtT5 and ProtBERT), four cases, 14 classifiers '
        'per case: **224 predictions**. Multiple browser sessions may submit analyses '
        'independently; GPU/R steps are capacity-limited.\n\n'
        '**Scoring:** Weighted score ≥0.80 AND SD <0.15 → **Strongly Recommended**; '
        'otherwise weighted score ≥0.70 AND SD ≤0.40 → **Useful**; otherwise **Not Useful**.\n\n'
        '**Important:** Predictions are computational prioritization, not measured degradation.'
    )).send()


async def process_fasta(message):
    uploaded = [f for f in (message.elements or [])
                if (getattr(f, 'name', '') or '').lower().endswith(('.faa', '.fa', '.fasta'))]
    if len(uploaded) != 1 or len(message.elements) != 1:
        await cl.Message(content='Please upload exactly one protein FASTA (.faa, .fa, .fasta).').send()
        return
    item = uploaded[0]
    run_dir = ROOT / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    file_path = run_dir / ('input' + Path(item.name).suffix.lower())
    try:
        if getattr(item, 'path', None) and Path(item.path).is_file():
            shutil.copyfile(item.path, file_path)
        elif getattr(item, 'content', None) is not None:
            file_path.write_bytes(item.content)
        else:
            await cl.Message(content='Unable to read the uploaded file.').send()
            return
        limit = int(os.getenv('BIOWASTE_MAX_FILE_MB', '100')) * 1024 * 1024
        if file_path.stat().st_size > limit:
            file_path.unlink(missing_ok=True)
            await cl.Message(content='Uploaded file exceeds server file-size limit.').send()
            return
        check_model_files()
        check_model_weights()
    except Exception as exc:
        log.exception('File import or configuration error')
        await cl.Message(content='Input/configuration error: '+str(exc)[:1000]).send()
        return

    status = cl.Message(content='⏳ Request accepted. Waiting for an available analysis worker...')
    await status.send()
    async with ANALYSIS_SLOTS:
        status.content = '⏳ Reading uploaded protein sequences...'
        await status.update()
        loop = asyncio.get_running_loop()
        updates = asyncio.Queue()

        def on_progress(text):
            try:
                loop.call_soon_threadsafe(updates.put_nowait, text)
            except RuntimeError:
                pass  # The browser/event loop may already have disconnected.

        task = asyncio.create_task(asyncio.to_thread(
            engine.run, str(file_path), str(run_dir), on_progress))
        while not task.done() or not updates.empty():
            try:
                update = await asyncio.wait_for(updates.get(), timeout=0.4)
            except asyncio.TimeoutError:
                continue
            status.content = '⏳ ' + update
            await status.update()
        try:
            result = await task
        except Exception as exc:
            log.exception('FASTA analysis failed')
            status.content = 'Analysis failed. No suitability label was generated.'
            await status.update()
            await cl.Message(content='**Error:** ' + str(exc)[:1500]).send()
            return
        status.content = '✅ Completed four PLMs and 224 classifier predictions.'
        await status.update()

    summary = result['protein_summary']
    label = result['predicted_label']
    display_label = 'Strongly Recommended' if label == 'Strongly Recommended' else label
    parts = [
        f'## {display_label}',
        f'**Strain:** `{result["analysis_id"]}`',
        f'**Proteins analyzed:** {summary["total_proteins"]:,} (target enzymes: {summary["target_proteins"]:,})',
        f'**CAZymes:** {summary["counts"].get("cazyme",0):,} · **Lipases:** {summary["counts"].get("lipase",0):,} · **Proteases:** {summary["counts"].get("protease",0):,}',
        f'**Performance-weighted suitability score:** **{result["weighted_suitability_score"]:.4f}**',
        f'**Inter-model SD (224 probabilities):** **{result["prediction_sd"]:.4f}**',
        '**Decision:** Strongly Recommended if score ≥ 0.80 and SD < 0.15; otherwise Useful if score ≥ 0.70 and SD ≤ 0.40; otherwise Not Useful.',
        f'**Supporting information:** Positive predictions {result["positive_votes"]}/224 ({result["positive_vote_ratio"]:.1%}); mean positive-class probability {result["mean_positive_probability"]:.1%}.',
        '\n### PLM breakdown',
        '| PLM | Positive votes | Weighted score within PLM | SD within PLM |',
        '|---|---:|---:|---:|',
    ]
    for r in result['plm_results']:
        parts.append(f'| {r["plm"]} | {r["positive_votes"]}/56 | {r["weighted_score_within_plm"]:.4f} | {r["prediction_sd_within_plm"]:.4f} |')
    parts += [
        '\n**Method:** Weight = max(CV AUC, 0.01) × max(Test MCC, 0.01) × max(Test balanced accuracy, 0.01). '
        'Score is the normalized weighted mean of 224 positive-class probabilities. '
        'SD is the unweighted sample SD (N − 1) of those 224 probabilities.',
        '**Interpretation:** Not Useful means the screening criteria were not met; '
        'it does not establish experimentally absent food-waste degradation.',
        '**Reproducibility:** Confirm PLM embeddings/pooling against original training references before scientific use.'
    ]
    if summary['untagged_proteins']:
        parts.append(f'**Header note:** {summary["untagged_proteins"]} proteins lacked a `|type|` tag and were treated as `other`.')
    downloads = [
        cl.File(name='all_224_predictions.csv', path=result['predictions_csv_path'], mime='text/csv'),
        cl.File(name='weighted_contributions.csv', path=result['weighted_contributions_csv_path'], mime='text/csv'),
        cl.File(name='report.json', path=result['report_json_path'], mime='application/json'),
    ]
    await cl.Message(content='\n\n'.join(parts), elements=downloads).send()


@cl.on_message
async def main(message: cl.Message):
    if cl.user_session.get('analysis_running', False):
        await cl.Message(content='This browser session already has an active task. Please use another session for a separate analysis.').send()
        return
    cl.user_session.set('analysis_running', True)
    try:
        if message.elements:
            await process_fasta(message)
            return
        if os.getenv('BIOWASTE_ENABLE_LLM', '1') != '1':
            await cl.Message(content='Upload an annotated protein FASTA file to run the 224-model analysis.').send()
            return
        async with CHAT_LOCK:
            # Shares bounded GPU admission with PLM inference; does not block R work.
            async with async_gpu_slot():
                history = cl.user_session.get('history', [])
                messages = llm.format_history(history, message.content)
                response = cl.Message(content='')
                await response.send()
                answer = ''
                try:
                    async for token in llm.generate_stream(messages):
                        answer += token
                        await response.stream_token(token)
                    await response.update()
                    history.extend([{'role':'user','content':message.content},
                                    {'role':'assistant','content':answer}])
                    cl.user_session.set('history', history[-8:])
                except Exception as exc:
                    log.exception('Gemma chat failed')
                    await cl.Message(content='Text chat error: '+str(exc)[:600]).send()
                finally:
                    if os.getenv('BIOWASTE_LLM_KEEP_LOADED','0') != '1':
                        await asyncio.to_thread(llm.release)
    finally:
        cl.user_session.set('analysis_running', False)
