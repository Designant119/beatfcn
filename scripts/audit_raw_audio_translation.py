"""Actual waveform translation and complete frozen-MERT/mel re-extraction.

Run only when the serial MPS training queues have ended. Model and waveform
identity checks precede inference. This is distinct from output circular
shifts and feature-array rolls. No chart labels enter either audio cache.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from scripts.run_local_portfolio_scale import sha,write


ROOT=Path(__file__).resolve().parents[1];IDENT='local-raw-audio-translation-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_raw_audio_translation_v1.json'
CACHE=ROOT/'reports/local_raw_audio_translation_features_v1.json';ART=ROOT/f'artifacts/{IDENT}'


def waveform(source):
    path=ROOT/source['raw_path']
    if sha(path)!=source['audio_sha256']:raise ValueError('Raw audio identity changed')
    data=np.frombuffer(subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vn','-f','f32le','-ac','1','-ar','24000','pipe:1'],timeout=30),dtype='<f4').copy()
    if len(data)!=source['decoded_samples'] or not np.isfinite(data).all():raise ValueError('Decoded sample clock changed')
    return data


def encode():
    import torch
    from beatfcn.audio.local_mert import CpuMelTransfer,LocalFrozenMert2
    from beatfcn.audio.mert_attention import install_query_chunks
    plan=json.loads(PLAN.read_text())
    if CACHE.exists():raise FileExistsError('No encoder retry')
    if sha(ROOT/plan['checkpoint_dir']/'model.safetensors')!=plan['weight_sha256']:raise ValueError('Pretrained checkpoint changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen re-extraction source drift')
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.5)
    sys.path.insert(0,str(ROOT/'inputs/d96_mert2_source'))
    official=importlib.import_module('mert2_pinned.modeling_mert2');cfgmod=importlib.import_module('mert2_pinned.configuration_mert2')
    config=cfgmod.MERT2Config(**json.loads((ROOT/plan['config']).read_text()));config._attn_implementation='sdpa'
    model,info=official.MERT2Model.from_pretrained(str(ROOT/plan['checkpoint_dir']),config=config,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',output_loading_info=True,local_files_only=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Checkpoint load mismatch')
    model.eval().requires_grad_(False).to('mps');model.feature_extractor=CpuMelTransfer(model.feature_extractor,'mps');install_query_chunks(model,512)
    encoder=LocalFrozenMert2(model);started=time.monotonic()
    report=dict(status='encoding',complete=False,planned_songs=len(plan['audios']),committed_songs=0,records=[],zero_update=True,cloud_compute_calls=0)
    write(CACHE,report)
    for source in plan['audios']:
        if time.monotonic()-started>plan['encode_wall_s']:raise TimeoutError('Raw-shift encoder wall')
        audio=source['audio_sha256'];wave=waveform(source)
        original=encoder.encode(wave);old=ROOT/source['full_cache_path']
        expected=np.stack([np.load(old/f'L{l}.npy',mmap_mode='r',allow_pickle=False) for l in (12,23)])
        if not np.array_equal(original['layers'],expected):raise ValueError('Zero-shift frozen-MERT parity failed; do not interpret translation')
        shifted=np.load(ART/'waveforms'/f'{audio}.npy',allow_pickle=False)
        result=encoder.encode(shifted)
        if not np.array_equal(result['times_s'],np.load(old/'times_s.npy')):raise ValueError('Translation changed nominal extraction clock')
        folder=ART/'features'/audio;folder.mkdir(parents=True,exist_ok=False)
        for j,l in enumerate((12,23)):np.save(folder/f'L{l}.npy',np.ascontiguousarray(result['layers'][j]))
        np.save(folder/'times_s.npy',result['times_s']);np.save(folder/'intervals_s.npy',result['intervals_s'])
        write(folder/'metadata.json',dict(result['metadata'],original_audio_sha256=audio,benchmark_transform='Fixed-length right shift with zero padding/crop',
            shift_us=plan['shift_us'],transformed_wave_sha256=hashlib.sha256(shifted.tobytes()).hexdigest(),checkpoint_sha256=plan['weight_sha256']))
        report['records'].append(dict(audio_sha256=audio,path=str(folder.relative_to(ROOT)),samples=len(shifted),frames=len(result['times_s']),
            shifted_mel=str((ART/'mel'/f'{audio}.npy').relative_to(ROOT)),zero_shift_mert_exact=True,
            files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir()}))
        report.update(committed_songs=len(report['records']),wall_s=time.monotonic()-started);write(CACHE,report)
        del wave,shifted,original,expected,result;torch.mps.empty_cache()
    report.update(status='completed',complete=True,wall_s=time.monotonic()-started);write(CACHE,report)


def main():
    import torch
    from beatfcn.audio.local_mel_cache import complete_song_mel
    from beatfcn.model.event_canvas import RhythmEvent
    from beatfcn.model.full_event_model import FullEventConfig
    from beatfcn.model.local_mamba_canvas import MambaCanvas
    from beatfcn.model.sample import match_f1
    from beatfcn.train.canonical_probability import explicit_row
    from beatfcn.train.mert_canvas_features import MertCanvasFeatures
    from beatfcn.train.relative_canvas import relative_batch,generate_relative
    from scripts.run_frontend_canvas_comparison import setup
    from scripts.run_local_portfolio_scale import row_factory
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve raw translation attempt')
    torch.set_num_threads(2);started=time.monotonic()
    parent_path=ROOT/'bench/rhythm-v2/local-portfolio-mamba-quality-v1.json';parent=json.loads(parent_path.read_text())
    panel,full=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,full)
    acquisition=json.loads((ROOT/'bench/rhythm-v2/d96-local-mert-cache-v1.json').read_text());sources={r['audio_sha256']:r for r in acquisition['audios']}
    indices=parent['indices']['dev'][:8:2];audios=[]
    for i in indices:
        audio=panel.records[i]['audio_sha256'];audios.append({**sources[audio],'full_cache_path':full.mert[audio]['path']})
    weight=ROOT/'artifacts/local-portfolio-v1/mamba-quality/mamba/quality-1000.pt'
    plan=dict(id=IDENT,indices=indices,audios=audios,shift_us=137000,shift_samples=3288,interior_margin_us=400000,
        checkpoint_sha256=sha(weight),parent_plan_sha256=sha(parent_path),checkpoint_dir=acquisition['checkpoint_dir'],config=acquisition['config'],
        weight_sha256=acquisition['weight_sha256'],environment=acquisition['environment'],wall_s=1800,encode_wall_s=900,
        optimizer_updates=0,cloud_compute_calls=0,confirmation_scored=False,
        protocol='Original float32 mono24000Hz waveform right-shifted137ms, same length, leading zeros and final crop. Re-extract fullMERT L12/L23 and complete10msmel. Zero-shift mel andMERT must reproduce original caches exactly. Clocks, masks, request, controls and references stay fixed. Score only a preregistered400ms interior; archive every emitted event.',
        interpretation='Four fixed reused dev components. Onset-only translation diagnostic; cropped long-hold tails are not a duration-superiority test. This is not output shifting, feature rolling, training or emotion evidence.',
        source_sha256={n:sha(ROOT/n) for n in ('scripts/audit_raw_audio_translation.py','beatfcn/audio/local_mert.py','beatfcn/audio/mert2.py','beatfcn/audio/mert_attention.py',
            'beatfcn/audio/local_mel_cache.py','beatfcn/audio/preprocess.py','inputs/d96_mert2_source/mert2_pinned/modeling_mert2.py','inputs/d96_mert2_source/mert2_pinned/configuration_mert2.py')})
    write(PLAN,plan)
    for name in plan['source_sha256']:
        dest=ART/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(PLAN,ART/'source/frozen-plan.json')
    report=dict(id=IDENT,status='preparing_waveforms',rows=[],zero_shift_mel=[],optimizer_updates=0,cloud_compute_calls=0,confirmation_scored=False);write(REPORT,report)
    try:
        for source in audios:
            audio=source['audio_sha256'];wave=waveform(source);zero=complete_song_mel(wave)['mel']
            original=np.load(ROOT/full.pupu.records[audio]['complete_mel'],mmap_mode='r',allow_pickle=False)
            if not np.array_equal(zero,original):raise ValueError('Zero-shift commonmel parity failed')
            shifted=np.zeros_like(wave);shifted[plan['shift_samples']:]=wave[:-plan['shift_samples']]
            (ART/'waveforms').mkdir(parents=True,exist_ok=True);(ART/'mel').mkdir(parents=True,exist_ok=True)
            np.save(ART/'waveforms'/f'{audio}.npy',shifted);np.save(ART/'mel'/f'{audio}.npy',complete_song_mel(shifted)['mel'])
            report['zero_shift_mel'].append(dict(audio_sha256=audio,exact=True));write(REPORT,report)
            del wave,zero,original,shifted
        report['status']='reextracting_MERT';write(REPORT,report)
        env={**os.environ,'PYTORCH_ENABLE_MPS_FALLBACK':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        with (ART/'encoder.log').open('x') as out:
            child=subprocess.Popen([str(ROOT/plan['environment']/'bin/python'),'-u','-m','scripts.audit_raw_audio_translation','--encode'],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=plan['encode_wall_s']+120)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        cache=json.loads(CACHE.read_text())
        if code or not cache['complete']:raise ValueError('Re-extraction failed; no translation quality verdict')
        transformed=MertCanvasFeatures(ROOT,full.pupu,str(CACHE.relative_to(ROOT)));cache_by={r['audio_sha256']:r for r in cache['records']}
        model=MambaCanvas(FullEventConfig(**parent['model']),density_reference=parent['density_reference']).eval()
        saved=torch.load(weight,map_location='cpu',weights_only=False)
        if saved['updates']!=1000 or saved['plan_sha256']!=sha(parent_path):raise ValueError('Reference checkpoint changed')
        model.load_state_dict(saved['model']);del saved
        report['status']='evaluating';write(REPORT,report)
        for index in indices:
            if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Translation audit wall')
            e=explicit_row(row(index,'draft'));truth=e['targets'];empty={**e,'targets':[]};audio=panel.records[index]['audio_sha256'];request=e['request']
            window=transformed.window(audio,request.start_us,request.end_us)
            mel=np.load(ROOT/cache_by[audio]['shifted_mel'],mmap_mode='r',allow_pickle=False);mlo=max(0,request.start_us//40000-100)*4
            window={**window,'mel':np.ascontiguousarray(mel[mlo:mlo+len(window['mel'])],dtype=np.float32)}
            shifted={**empty,'inputs':{**empty['inputs'],**window}};predictions={}
            with torch.no_grad():
                for name,example in [('correct',empty),('translated',shifted)]:
                    inp,_=relative_batch([example],'cpu');prediction=generate_relative(model,inp,request,deadline=started+plan['wall_s'])
                    if not prediction['complete']:raise ValueError('Translation inference incomplete')
                    predictions[name]=prediction
            left=request.start_us+plan['interior_margin_us'];right=request.end_us-plan['interior_margin_us'];delta=plan['shift_us']
            true=[v.onset_us for v in truth if left<=v.onset_us<right]
            correct=[v['onset_us'] for v in predictions['correct']['events'] if left<=v['onset_us']<right]
            changed=[v['onset_us'] for v in predictions['translated']['events'] if left+delta<=v['onset_us']<right+delta]
            adjusted=[v-delta for v in changed]
            values={f'{name}_{ms}ms':match_f1(pred,target,ms*1000)[3] for name,pred,target in (
                ('correct_audio',correct,true),('translated_raw_clock',changed,true),('translated_corrected_clock',adjusted,true),
                ('self_raw_clock',changed,correct),('self_corrected_clock',adjusted,correct)) for ms in (10,50)}
            record=dict(index=index,component=panel.records[index]['component'],request=asdict(request),interior_us=[left,right],shift_us=delta,
                predictions=predictions,scores=values,true_interior_onsets=true,real_waveform_reextracted=True)
            report['rows'].append(record);write(ART/f'{index}.json',record);write(REPORT,report)
        report.update(status='completed',wall_s=time.monotonic()-started,checkpoint_unchanged=sha(weight)==plan['checkpoint_sha256'],
            mean_scores={k:float(np.mean([r['scores'][k] for r in report['rows']])) for k in report['rows'][0]['scores']},
            positive_translation_components=sum(r['scores']['translated_corrected_clock_10ms']>r['scores']['translated_raw_clock_10ms'] for r in report['rows']))
        write(REPORT,report)
    except BaseException as exc:
        report.update(status='stopped',failure=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(REPORT,report);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--encode',action='store_true');parser.add_argument('--after-concat',action='store_true');args=parser.parse_args()
    if args.encode:encode()
    elif args.after_concat:
        queue=ROOT/'reports/local-portfolio-v1/raw-translation-after-concat-v1.json'
        if queue.exists():raise FileExistsError('No duplicate raw translation tail')
        state=dict(status='waiting_for_concat_control',cloud_compute_calls=0,automatic_retries=0);write(queue,state);started=time.monotonic()
        try:
            while True:
                parent=json.loads((ROOT/'reports/local-portfolio-v1/concat-after-attention-v1.json').read_text())
                if parent['status']=='completed':break
                if parent['status']=='stopped':raise RuntimeError('Predecessor requires explicit review')
                if time.monotonic()-started>60000:raise TimeoutError('Wait bound')
                time.sleep(5)
            state['status']='running_raw_translation';write(queue,state);main()
            state.update(status='completed',experiment_report=str(REPORT.relative_to(ROOT)),wall_s=time.monotonic()-started);write(queue,state)
        except BaseException as exc:
            state.update(status='stopped',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);write(queue,state);raise
    else:main()
