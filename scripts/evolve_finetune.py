"""Bounded fine-tune of an evolve candidate with the data recipe of the r5 plan.

A separate loop, not a plan for scripts/run_local_song_baseline.py: that entry pins code hashes and health reports and
has no per-field loss, KD, LayerDrop, freezing, absolute deadline or progress file. This one reuses its pieces
(training panel, exposure loader, regional compile, EMA, LR schedule, non-finite gradient guard).

--updates is the cumulative target, so rungs are `--updates 300`, then `--updates 900 --resume`, then `--updates 2400
--resume`; the LR schedule spans the plan's fixed horizon, so every rung is a prefix of the same trajectory.

  BEATFCN_ALLOW_MISSING_MERT=1 python -m scripts.evolve_finetune --plan PLAN.json --updates 300 --deadline-unix T

Writes OUT/progress.json atomically every 30 s from its own thread, also while the loop is stuck, and hard-exits (code
3) when stuck past the deadline. Exit 0 with phase 'done' or 'deadline' (resumable), 1 on failure.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from beatfcn.model.full_event_codec import FIELDS, EventVocabulary  # noqa: E402
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel  # noqa: E402
from beatfcn.train.counterfactual_exposure import build_exposure  # noqa: E402
from beatfcn.train.ema import ParameterEMA  # noqa: E402
from beatfcn.train.evolve_objective import evolve_loss, field_ce_sums, token_weight_table  # noqa: E402
from beatfcn.train.local_request_panel import event_request  # noqa: E402
from beatfcn.train.peer_rl import PeerRL  # noqa: E402
from beatfcn.train.exposure_loader import build_exposure_loader, make_vocabulary, plan_loader_settings  # noqa: E402
from beatfcn.train.training_objective import update_learning_rate  # noqa: E402
from scripts.evolve_surgery import file_sha, load_checkpoint_model  # noqa: E402

STUCK_GRACE_S = 600.
SAVE_MARGIN_S = 90.


def atomic_json(path, value):
    temporary = Path(str(path)+'.partial')
    temporary.write_text(json.dumps(value, allow_nan=False)+'\n')
    temporary.replace(path)


def load_recipe(spec, init_config):
    """The r5 plan, with the candidate's model config and the spec's overrides."""
    base_path = Path(spec['base_plan'])
    if spec.get('base_plan_sha256') and file_sha(base_path) != spec['base_plan_sha256']:
        raise ValueError(f'{base_path} is not the plan this candidate was written against')
    recipe = json.loads(base_path.read_text())
    recipe['config'] = dict(init_config)
    for key in ('seed', 'loader_workers', 'loader_prefetch', 'song_style'):
        if key in spec:
            recipe[key] = spec[key]
    return recipe


def missing_audio(panel):
    return set(getattr(panel, 'missing_audio', ()))


def ensure_exposure(spec, recipe, panel):
    """Fresh exposure over the train requests that have features; shared by candidates with the same path."""
    settings = spec['exposure']
    path = Path(settings['path'])
    steps = int(spec['schedule']['horizon_updates'])*int(recipe['accum_steps'])
    missing = missing_audio(panel)
    if path.exists():
        exposure = json.loads(path.read_text())
    else:
        train = [i for i, row in enumerate(panel.records) if row['split'] == 'train' and row['audio_sha256'] not in missing]
        tags = {int(k): v for k, v in json.loads(Path(settings['subdivision_tags']).read_text())['tags'].items()}
        counterfactual = recipe['counterfactual']
        exposure = build_exposure(train, steps=steps, seed=int(settings['seed']), mix=counterfactual['treatment_mix'],
                                  dropout=counterfactual['condition_dropout'],
                                  song_of={i: panel.records[i]['audio_sha256'] for i in train},
                                  song_chunk=int(recipe.get('exposure_song_chunk', 8)),
                                  unconditional_native=bool(settings['unconditional_native']),
                                  mode_weights=recipe['mode_weights'], mode_of=tags)
        exposure['excluded_missing_feature_songs'] = len(missing)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, exposure)
    if len(exposure['steps']) != steps:
        raise ValueError('Exposure length differs from horizon_updates x accum_steps')
    used = {panel.records[i]['audio_sha256'] for step in exposure['steps'] for i in step['indices']}
    if used & missing:
        raise ValueError(f'{len(used & missing)} exposed songs have no features')
    return exposure


def loader_workers(planned):
    """A container's nproc shows host cores; the cgroup quota is the real limit (24 threads once starved sshd)."""
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        return min(int(planned), max(int(quota)//int(period)-2, 1)) if quota != 'max' else int(planned)
    except (OSError, ValueError):
        return int(planned)


def layerdrop_keep(depth, p, rng):
    """Kept decoder layer indices of one forward; the first and last layer always stay, as in surgery."""
    return [i for i in range(depth) if i in (0, depth-1) or rng.random() >= p]


@contextmanager
def layer_dropped(model, keep):
    """Swap in the kept decoder blocks for one forward: encode, memory K/V and decode all read model.decoder, and
    parameters of dropped layers simply get no gradient. Needs no model definition change."""
    full = model.decoder
    model.decoder = torch.nn.ModuleList([full[i] for i in keep])
    try:
        yield
    finally:
        model.decoder = full


def freeze_encoders(model):
    """Everything encode() owns; decode side (tokens, fields, mode ids, blocks, output) keeps training."""
    trainable = ('decoder.', 'token_embedding.', 'field_embedding.', 'mode_ids.', 'final_norm.', 'output.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable))


def set_train_mode(model, frozen):
    model.train()
    if frozen:
        model.encoders.eval()


class Progress:
    """Latest run state; a daemon thread persists it, so a stall stays visible."""

    def __init__(self, path, *, every_s, deadline_unix, stuck_grace_s=STUCK_GRACE_S):
        self.path, self.every_s, self.deadline_unix = Path(path), float(every_s), deadline_unix
        self.stuck_grace_s = stuck_grace_s
        self.state = dict(phase='start', updates=0, started_unix=time.time(), last_update_unix=None,
                          deadline_unix=deadline_unix)
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.rate_mark = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)

    def write(self):
        with self.lock:
            s = dict(self.state)
            now = time.time()
            s['written_unix'] = now
            if s['last_update_unix'] is not None:
                s['since_last_update_s'] = now-s['last_update_unix']
            first, target = s.get('first_update_unix'), s.get('target_updates')
            done = s['updates']-s.get('start_updates', 0)
            if first is not None and done > 0 and now > first:
                s['rate_updates_per_s'] = done/(now-first)
                if self.rate_mark is not None and now > self.rate_mark[0]:
                    s['recent_updates_per_s'] = (s['updates']-self.rate_mark[1])/(now-self.rate_mark[0])
                self.rate_mark = (now, s['updates'])
                if target is not None:
                    s['eta_s'] = max(target-s['updates'], 0)/s['rate_updates_per_s']
            atomic_json(self.path, s)

    def _loop(self):
        while not self.stop.wait(self.every_s):
            self.write()
            if (self.deadline_unix is not None and time.time() > self.deadline_unix+self.stuck_grace_s
                    and self.state['phase'] not in ('done', 'deadline', 'failed')):
                self.update(phase='stuck_past_deadline')
                self.write()
                print(json.dumps(dict(phase='stuck_past_deadline')), flush=True)
                os._exit(3)

    def start(self):
        self.write()
        self.thread.start()

    def close(self, **fields):
        self.update(**fields)
        self.stop.set()
        self.write()


def checkpoint_path(out, updates):
    return out/f'step-{updates:06d}.pt'


def latest_checkpoint(out):
    found = sorted(out.glob('step-*.pt'))
    return found[-1] if found else None


def save_checkpoint(out, *, model, ema, optimizer, updates, spec, config, vocabulary, keep, save_optimizer):
    """``model`` holds the EMA weights in fp32 so benchmark and probe load it like an r5 export; ``raw`` is for resume."""
    path = checkpoint_path(out, updates)
    temporary = path.with_suffix('.partial')
    value = dict(schema='beatfcn.evolve-checkpoint.v1', id=spec['id'], config=config, vocabulary=vocabulary, weights='ema',
                 model=ema.model_state_dict(model), raw={k: v.detach().cpu() for k, v in model.state_dict().items()},
                 ema=dict(decay=ema.decay, updates=ema.updates), updates=updates, init=spec['init'])
    if save_optimizer:
        value['optimizer'] = optimizer.state_dict()
    torch.save(value, temporary)
    temporary.replace(path)
    for old in sorted(out.glob('step-*.pt'))[:-keep]:
        old.unlink(missing_ok=True)
    return path


def finetune(spec, *, updates, deadline_unix=None, resume=False, device='cuda', compile_regions=True, save_optimizer=False,
             panel=None, progress_every_s=30., log_every=10):
    """Returns the terminal phase ('done' or 'deadline'). ``panel`` is injected by tests, else built from the r5 plan."""
    out = Path(spec['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    horizon = int(spec['schedule']['horizon_updates'])
    if not 0 < updates <= horizon:
        raise ValueError(f'--updates must lie in [1, {horizon}], the plan horizon')
    previous = latest_checkpoint(out)
    if previous is not None and not resume:
        raise FileExistsError(f'{previous} exists; pass --resume or use a new out_dir')
    if previous is None and resume:
        raise FileNotFoundError(f'Nothing to resume in {out}')
    progress = Progress(out/'progress.json', every_s=progress_every_s, deadline_unix=deadline_unix)
    progress.update(id=spec['id'], target_updates=updates, horizon_updates=horizon)
    progress.start()

    def say(**fields):
        print(json.dumps(fields), flush=True)
    try:
        source = previous or spec['init']
        progress.update(phase='load_checkpoint', source=str(source))
        saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        config, vocabulary = saved['config'], saved.get('vocabulary', {})
        start_updates = int(saved.get('updates', 0)) if previous is not None else 0
        progress.update(updates=start_updates, start_updates=start_updates)
        recipe = load_recipe(spec, config)
        torch.set_num_threads(int(recipe.get('cpu_threads', 2)))
        torch.manual_seed(int(recipe['seed'])+start_updates)
        progress.update(phase='build_panel')
        if panel is None:
            from beatfcn.train.training_panel import build_training_panel
            panel = build_training_panel(Path('/'), recipe)
        progress.update(phase='build_exposure')
        exposure = ensure_exposure(spec, recipe, panel)
        accum = int(recipe['accum_steps'])
        microbatch = int(recipe.get('physical_microbatch_requests', 4))
        stride = max(1, microbatch//4)
        # Workers must fork before any CUDA context exists in this process.
        progress.update(phase='start_loader')
        workers = loader_workers(recipe.get('loader_workers', 8)) if device == 'cuda' else 0
        loader = build_exposure_loader(panel, exposure, microbatch=microbatch, num_workers=workers,
                                       prefetch_factor=int(recipe.get('loader_prefetch', 2)),
                                       start_update=start_updates*accum, seed=int(recipe['seed']),
                                       pin_memory=device == 'cuda', **plan_loader_settings(recipe))
        loader_iter = iter(loader)
        progress.update(phase='load_model', loader_workers=workers)
        c = FullEventConfig(**config)
        vocab = EventVocabulary(**vocabulary)
        if vocab != make_vocabulary(recipe.get('vocabulary')):
            raise ValueError('Checkpoint and base plan disagree on the token vocabulary')
        model = FullEventModel(c, vocab)
        model.load_state_dict({k: v.to(torch.float32) for k, v in saved.get('raw', saved['model']).items()}, strict=True)
        model = model.to(device)
        frozen = bool(spec.get('freeze_encoders'))
        if frozen:
            freeze_encoders(model)
        settings = recipe.get('optimizer', {})
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (decay if parameter.ndim >= 2 and 'input_scale' not in name else no_decay).append(parameter)
        trainable = decay+no_decay
        optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=float(settings.get('weight_decay', 0.))),
                                       dict(params=no_decay, weight_decay=0.)], lr=float(spec['learning_rate']),
                                      betas=tuple(settings.get('betas', (.9, .999))))
        rewarm = previous is not None and not saved.get('optimizer')
        if previous is not None and saved.get('optimizer'):
            optimizer.load_state_dict(saved['optimizer'])
        ema = ParameterEMA(model, float(spec['ema']['decay']))
        if previous is not None:
            ema.load_state_dict(dict(decay=saved['ema']['decay'], bias_corrected=True, updates=saved['ema']['updates'],
                                     model=saved['model']), model)
        del saved
        kd, teacher = spec.get('kd'), None
        if kd:
            progress.update(phase='load_teacher')
            teacher = load_checkpoint_model(kd['teacher'], device)
        table = token_weight_table(vocab, spec.get('field_weights'), spec.get('kind_class_weights'))
        layerdrop = float(spec.get('layerdrop', 0.))
        loss_settings = recipe['loss']
        amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(recipe.get('autocast'))
        if amp_dtype is torch.float16:
            raise ValueError('Fine-tune supports bf16 or fp32 only')
        amp = (lambda: torch.autocast('cuda', dtype=amp_dtype)) if amp_dtype is not None and device == 'cuda' else nullcontext
        if compile_regions and device == 'cuda' and recipe.get('compile_loss'):
            try:
                from beatfcn.train.compiled_training import enable_regional_compile
            except ImportError:
                raise SystemExit('compile_loss needs beatfcn/train/compiled_training.py (retired with the P1 line); drop it from the plan')
            progress.update(phase='compile')
            options = recipe['compile_loss'].get('options')
            enable_regional_compile(model, options=options)
            if teacher is not None:
                enable_regional_compile(teacher, options=options)
        with_aux = bool(c.onset_head)
        peer_rl, rl_counts = None, Counter()
        if spec.get('rl'):
            if compile_regions and recipe.get('compile_loss'):
                raise ValueError('RL replays have free lengths: run it with --no-compile')
            peer_rl = PeerRL(spec['rl'], vocab)
            if peer_rl.anchor_weight:
                # The student starts from the raw weights of init, so the teacher must too: KL starts at exactly zero.
                start = torch.load(spec['init'], map_location='cpu', weights_only=False, mmap=True)
                teacher_rl = FullEventModel(FullEventConfig(**start['config']), EventVocabulary(**start.get('vocabulary', {})))
                teacher_rl.load_state_dict({k: v.to(torch.float32) for k, v in start.get('raw', start['model']).items()}, strict=True)
                peer_rl.teacher = teacher_rl.to(device).eval().requires_grad_(False)
                del start, teacher_rl
        schedule = spec['schedule']
        current, phase, wait_s, wall_start = start_updates, 'done', 0., time.time()
        field_sum, field_count = torch.zeros(len(FIELDS), device=device), torch.zeros(len(FIELDS), device=device)
        losses = []
        last_log = (time.time(), start_updates)
        progress.update(phase='warmup', trainable_parameters=sum(p.numel() for p in trainable))
        say(phase='boot', id=spec['id'], start_updates=start_updates, target=updates, trainable=sum(p.numel() for p in trainable),
            depth=c.decoder_layers, decoder_ffn=c.decoder_ffn or c.ffn, loader_workers=workers, rewarm=rewarm,
            layerdrop=layerdrop, kd=bool(kd), frozen=frozen, weighted=table is not None)
        set_train_mode(model, frozen)
        while current < updates:
            if deadline_unix is not None and time.time() >= deadline_unix-SAVE_MARGIN_S:
                phase = 'deadline'
                break
            lr = update_learning_rate(optimizer, current, horizon, base_lr=float(spec['learning_rate']),
                                      warmup_updates=int(schedule['warmup_updates']), minimum_lr=float(schedule['minimum_lr']))
            if rewarm:
                # Adam moments restarted: a fresh warmup keeps the first steps small.
                lr *= min(1., (current-start_updates+1)/int(schedule['warmup_updates']))
                for group in optimizer.param_groups:
                    group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            for _ in range(0, accum, stride):
                tick = time.monotonic()
                item = next(loader_iter)
                wait_s += time.monotonic()-tick
                for ids, cpu_inputs, cpu_sequence in item['batches']:
                    inp = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in cpu_inputs.items()}
                    seq = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in cpu_sequence.items()}
                    weight = len(ids)/4/accum
                    keep = (layerdrop_keep(c.decoder_layers, layerdrop, np.random.default_rng([int(recipe['seed']), current, ids[0]]))
                            if layerdrop else range(c.decoder_layers))
                    with amp():
                        teacher_logits = None
                        if teacher is not None:
                            with torch.no_grad():
                                teacher_logits = teacher(inp, seq)
                        with layer_dropped(model, keep) if layerdrop else nullcontext():
                            logits, aux = model(inp, seq, return_aux=True) if with_aux else (model(inp, seq), None)
                        loss = evolve_loss(logits, seq, loss_settings, aux=aux, token_weights=table,
                                           teacher_logits=teacher_logits, kd=kd)
                    (loss*weight).backward()
                    total += loss.detach()*weight
                    summed, counted = field_ce_sums(logits, seq)
                    field_sum += summed
                    field_count += counted
                    del inp, seq, logits, aux, loss, teacher_logits
            if peer_rl is not None:
                ids, rl_inputs, rl_sequence = item['batches'][0]
                windows = int(spec['rl'].get('windows', 8))
                rl_rows = {k: (v[:windows].to(device) if torch.is_tensor(v) else v) for k, v in rl_inputs.items()}
                rl_requests = [event_request(panel.records[i]['request']) for i in ids[:windows]]
                rl_counts.update(peer_rl.accumulate(model, rl_rows, rl_requests, rl_sequence['valid'][:windows].sum(1).tolist(),
                                                    int(recipe['seed'])+current, amp, lambda: set_train_mode(model, frozen)))
                del rl_rows
            # The non-finite guard of the r5 entry: never step on a NaN or Inf gradient.
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            ema.update()
            current += 1
            now = time.time()
            losses.append(float(total))
            first = progress.state.get('first_update_unix') or now
            progress.update(phase='train', updates=current, lr=lr, last_update_unix=now, first_update_unix=first,
                            grad_norm=float(norm), data_wait_fraction=wait_s/max(now-wall_start, 1e-9))
            if current % log_every == 0 or current == updates:
                per_field = {name: float(s/n) for name, s, n in zip(FIELDS, field_sum.tolist(), field_count.tolist()) if n}
                objective = float(np.mean(losses))
                rate = (current-last_log[1])/max(now-last_log[0], 1e-9)
                progress.update(loss=objective, loss_per_field=per_field,
                                gpu_peak_mib=torch.cuda.max_memory_allocated()//2**20 if device == 'cuda' else None)
                say(phase='train', updates=current, target=updates, updates_per_s=round(rate, 4), loss=round(objective, 5),
                    grad=round(float(norm), 4), lr=lr, data_wait=round(wait_s/max(now-wall_start, 1e-9), 3),
                    fields={k: round(v, 4) for k, v in per_field.items()}, **({'rl': dict(rl_counts)} if peer_rl else {}))
                rl_counts.clear()
                last_log, losses = (now, current), []
                field_sum.zero_()
                field_count.zero_()
            if current % int(spec['light_checkpoint_every']) == 0 and current < updates:
                progress.update(phase='checkpoint')
                path = save_checkpoint(out, model=model, ema=ema, optimizer=optimizer, updates=current, spec=spec, config=config,
                                       vocabulary=vocabulary, keep=int(spec['light_checkpoint_keep']), save_optimizer=save_optimizer)
                progress.update(phase='train', checkpoint=str(path))
        if current > start_updates:
            progress.update(phase='checkpoint')
            path = save_checkpoint(out, model=model, ema=ema, optimizer=optimizer, updates=current, spec=spec, config=config,
                                   vocabulary=vocabulary, keep=int(spec['light_checkpoint_keep']), save_optimizer=save_optimizer)
            progress.update(checkpoint=str(path))
        progress.close(phase=phase, updates=current)
        say(phase=phase, updates=current, target=updates, checkpoint=progress.state.get('checkpoint'),
            wall_s=round(time.time()-wall_start, 1))
        del loader_iter, loader
        return phase
    except Exception as exc:
        progress.close(phase='failed', error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--updates', type=int, required=True, help='cumulative target, at most the plan horizon')
    a.add_argument('--deadline-unix', type=float, required=True, help='absolute stop; saves and exits before it')
    a.add_argument('--resume', action='store_true')
    a.add_argument('--save-optimizer', action='store_true', help='adds Adam state, about 2x the file size')
    a.add_argument('--no-compile', action='store_true', help='eager decoder blocks, if regional compile fails')
    a.add_argument('--device', default='cuda')
    args = a.parse_args()
    finetune(json.loads(args.plan.read_text()), updates=args.updates, deadline_unix=args.deadline_unix, resume=args.resume,
             device=args.device, compile_regions=not args.no_compile, save_optimizer=args.save_optimizer)


if __name__ == '__main__':
    main()
