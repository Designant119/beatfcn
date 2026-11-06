"""Export a deployable model file from a training checkpoint.

Keeps only what inference needs: config, vocabulary, the chosen weights (EMA by
default) and provenance (source checkpoint SHA, plan SHA, update count). No
optimizer state. Verifies the export loads into FullEventModel and reproduces
the source weights exactly.

  .venv/bin/python -m scripts.export_final_model --checkpoint step-X.pt --plan plan.json \
      --out final/beatfcn-final.pt [--weights ema]
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import torch
    from beatfcn.model.full_event_codec import EventVocabulary
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    a = p.parse_args()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if a.weights == 'ema' and not saved.get('ema'):
        raise SystemExit('Checkpoint has no EMA state')
    state = saved['ema']['model'] if a.weights == 'ema' else saved['model']
    config, vocabulary = saved['config'], saved.get('vocabulary', {})
    model = FullEventModel(FullEventConfig(**config), EventVocabulary(**vocabulary))
    model.load_state_dict(state, strict=True)
    plan = json.loads(a.plan.read_text())
    export = dict(
        schema='beatfcn.final-model.v1', config=config, vocabulary=vocabulary,
        weights=a.weights, model={k: v.detach().clone() for k, v in model.state_dict().items()},
        parameters=sum(t.numel() for t in model.parameters()),
        provenance=dict(source_checkpoint=str(a.checkpoint), source_sha256=sha(a.checkpoint),
                        plan=str(a.plan), plan_sha256=sha(a.plan), plan_id=plan.get('id'),
                        exposure_steps=saved.get('updates'),
                        optimizer_updates=saved.get('updates', 0) // int(plan.get('accum_steps', 1)),
                        ema=dict(decay=saved['ema'].get('decay'), updates=saved['ema'].get('updates'))
                        if saved.get('ema') else None,
                        tape=plan.get('tape'), mel_normalization=plan.get('mel_normalization')))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.out.with_suffix('.tmp')
    torch.save(export, temporary)
    temporary.replace(a.out)
    check = torch.load(a.out, map_location='cpu', weights_only=False)
    reloaded = FullEventModel(FullEventConfig(**check['config']), EventVocabulary(**check['vocabulary']))
    reloaded.load_state_dict(check['model'], strict=True)
    exact = all(torch.equal(reloaded.state_dict()[k], state[k].to(reloaded.state_dict()[k].dtype)) for k in state)
    if not exact:
        raise SystemExit('Exported weights do not reproduce the source state')
    print(json.dumps(dict(out=str(a.out), sha256=sha(a.out), bytes=a.out.stat().st_size,
                          parameters=export['parameters'], weights=a.weights,
                          optimizer_updates=export['provenance']['optimizer_updates']), indent=1))


if __name__ == '__main__':
    main()
