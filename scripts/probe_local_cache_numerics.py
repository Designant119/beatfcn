"""Reproduce full-prefix/cache numerical differences without training or decoding changes."""
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from beatfcn.model.full_event_codec import FIELDS, EventRequest, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import request_inputs
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run():
    output = ROOT/'reports/d68_cache_numerics_v2.json'
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((ROOT/'bench/rhythm-v2/d68-cpu-generation-1000-v1.json').read_text())
    for source, expected in plan['inputs_sha256'].items():
        if sha(ROOT/source) != expected:
            raise ValueError('Frozen input changed: '+source)
    torch.set_num_threads(2)
    saved = torch.load(ROOT/plan['checkpoint'], map_location='cpu', weights_only=False)
    model = FullEventModel(FullEventConfig(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    features = LocalFeatureStore(ROOT,plan['features'])
    source = ROOT/'artifacts/d68-cpu-generation-1000-v1/phira-28312.json'
    song = json.loads(source.read_text())
    core = next(c for c in song['cores'] if c['start_us']==80_000_000)
    request = EventRequest('phira',core['start_us'],core['end_us'],song['duration_us'])
    controls = encode_controls('phira',{},(request.start_us,request.end_us),origin_us=request.start_us)
    inputs,_ = request_inputs(features,song['audio_sha256'],request,controls)
    inp,seq = collate([dict(inputs=inputs,sequence=teacher_sequence(core['tokens'],request))])
    inp = pad_input_shapes(inp,plan['input_shapes'])
    # The collator pads to a multiple of 128. The real decoder stops at EOS;
    # cached queries after EOS are not a valid comparison with padded teacher
    # queries, whose padding keys are masked out. Keep the full teacher tensor
    # but compare exactly the real prefix, including its final EOS decision.
    valid_length = int(seq['valid'].sum())
    if valid_length != len(core['tokens']) or not bool(seq['valid'][0,:valid_length].all()):
        raise ValueError('Expected one contiguous real prefix followed by padding')
    memory = model.encode(inp)
    keys=('tokens','fields','times','valid')
    report=dict(checkpoint_sha256=sha(ROOT/plan['checkpoint']),raw_sha256=sha(source),script_sha256=sha(Path(__file__)),
        optimizer_updates=0,new_cloud_compute_calls=0,cases=[],tolerance=1e-3,
        hypotheses=['Shape-dependent float32 SDPA kernel error; forcing math should remove it.',
                    'Other shape-dependent linear/RoPE numerical differences; math alone will not remove it.',
                    'A logical cache or masking error; errors persist in higher precision or alter legal decisions.'])
    for name,context in [('default',nullcontext),('math',lambda:sdpa_kernel(SDPBackend.MATH))]:
        started=time.monotonic()
        with context():
            full=model.decode(*(seq[k] for k in keys),memory)
            cache=model.start_cache(memory)
            max_error=legal_error=prob_error=0.
            worst=None;mismatch=0;by_field={}
            for i in range(valid_length):
                inc=model.decode(*(seq[k][:,i:i+1] for k in keys),memory,cache=cache)
                diff=(full[:,i:i+1]-inc).abs()
                err=float(diff.max())
                if err>max_error:
                    max_error=err;worst=dict(position=i,field=FIELDS[int(seq['fields'][0,i])])
                allowed=seq['allowed'][:,i:i+1]
                legal_error=max(legal_error,float(diff[allowed].max()))
                fa,ia=(v.masked_fill(~allowed,-1e9) for v in (full[:,i:i+1],inc))
                mismatch+=int((fa.argmax(-1)!=ia.argmax(-1)).sum())
                prob_error=max(prob_error,float((fa.softmax(-1)-ia.softmax(-1)).abs().max()))
                field=FIELDS[int(seq['fields'][0,i])]
                by_field[field]=max(by_field.get(field,0.),float(diff[allowed].max()))
        report['cases'].append(dict(backend=name,tokens=valid_length,padded_tokens=seq['tokens'].shape[1]-valid_length,max_logit_difference=max_error,
            max_legal_logit_difference=legal_error,max_probability_difference=prob_error,
            legal_argmax_mismatches=mismatch,worst=worst,legal_error_by_field=by_field,
            wall_s=time.monotonic()-started,passed=max_error<1e-3 and mismatch==0))
        write_json(output,report)
    report['complete']=True
    write_json(output,report)
    print(json.dumps(report['cases']))


if __name__=='__main__':
    run()
