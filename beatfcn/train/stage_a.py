"""Matched Phira context experiments. Preparation only; contains no job launcher.

The three arms share requests, rhythm labels, optimizer policy and source weights.
Stage A does not produce STATIC/BIND actions or establish their product value.
"""
from dataclasses import asdict, replace
import copy
import hashlib
import json
import numpy as np
import torch
from ..events.reference_policy import POLICY, bound_evidence
from ..events.requests import MASK_MIX
from ..model import features as F
from ..model.p1 import P1Config, P1Model
from ..spatial import project
from ..spatial.rpe import digest

ARMS = ('legacy', 'removed', 'operation')


def protocol():
    return {'schema': 'beatfcn.phira_stage_a.v1', 'status': 'prepared_training_paused',
            'launch_enabled': False, 'training_updates_executed': 0, 'paid_compute_jobs_started': 0,
            'campaign_cap_usd': 10.0, 'checkpoint': '/data/runs/s1-v21/best.pt', 'checkpoint_step': 3500,
            'checkpoint_sha256': '64a7797142fee342491ea269e4a105261f7399be32cfc19650a97395966ed1b4',
            'checkpoint_identity_source': 'reports/evolution_checkpoint_inspection.json; reverify bytes before use',
            'common': {'effective_updates_planned': 1000, 'optimizer': 'new_identical_state_all_arms',
                       'optimizer_settings': {'name': 'AdamW', 'betas': [.9, .95], 'weight_decay': .01,
                                              'clip': 1.0, 'fresh_state': True},
                       'learning_rate': {'peak': 2e-4, 'minimum': 2e-5, 'warmup_updates': 500,
                                         'schedule_total_updates': 8000, 'schedule_start': 3500},
                       'micro_batch': 8, 'accumulation': 4,
                       'request_mix': dict(MASK_MIX), 'control_drop_all_p': .20,
                       'lane_history_groups': 0, 'offset_sigma_ms': 0.0,
                       'selection': False, 'counterfactual': False, 'replay_fraction': 0.0,
                       'context_perturbation': dict(F.PERTURB), 'perturbation_seed': 'frozen_per_core_shared_across_arms',
                       'phira_location': False, 'execution': 'eager', 'reference_policy': POLICY,
                       'permission_quantization': 'crossing_events_observed_whole_v1',
                       'generated_prefix_space': 'unplaced_in_every_arm',
                       'memory_overflow': 'error', 'memory_capacity': 256,
                       'seeds': [20261024], 'budget_reconciliation_required': True},
            'arms': {arm: {'phira_context': arm, 'input': label} for arm, label in zip(ARMS,
                     ('authorized_legacy_line_side_x', 'legacy_spatial_fields_removed', 'authorized_operation_geometry'))},
            'comparisons': ['operation-minus-removed: geometry representation and encoder benefit', 'removed-minus-legacy: migration cost',
                            'operation-minus-legacy: total effect'],
            'interpretation': 'operation adds an encoder and memory rows; three arms alone do not isolate information from capacity',
            'negative_controls': ['no-scene from-scratch has no spatial input',
                                  'operation inference with geometry masked or mismatched; shared rhythm and permissions'],
            'acceptance': {'unit': 'song; paired confidence intervals', 'completion': 'all requests remain in denominator',
                           'hard_failures': ['hidden_input_leak', 'changed_locked_events', 'type_violation', 'budget_overrun'],
                           'quality_margins': {'note_f1_absolute': -.02, 'completion_delta': 0.0,
                                               'status': 'provisional_engineering_limits_not_human_calibrated'},
                           'promotion': 'paired lower CI above stated margin; primary gain must be positive; no automatic promotion',
                           'unknown_approximate_truncated': 'separate coverage denominators',
                           'geometry_support': 'report fully_supported, partially_supported, unsupported and no_spatial_input separately',
                           'spatial_output_quality': 'not_measured_in_stage_a',
                           'final_test': 'sealed; existing 40-song panel is development only'}}


def prepare_sample(sample, scene=None, *, shift_us=0, reference_sources=None, grants=(),
                   memory_overflow='error'):
    """Declare exact geometry grants and prepare a projection shared by all arms."""
    if not sample.ok:
        raise ValueError('quarantined model view cannot enter Stage A')
    evidence, budget = bound_evidence(sample.evidence)
    req = copy.deepcopy(sample.request)
    req['operation_space'] = {'version': 'context_v1', 'grants': copy.deepcopy(list(grants)),
                              'model_shift_us': shift_us}
    sources = reference_sources or {}
    for ref in req.get('references', []):
        source = sources.get(ref['reference_id'])
        ref['space_attributes'] = source is not None
        if source is not None:
            ref['spatial_source_sha256'] = source.source_sha256
    prepared = replace(sample, request=req, evidence=tuple(evidence),
                       recipe={**sample.recipe, 'reference_budget': sample.recipe.get('reference_budget', budget),
                               'stage_a_reference_recheck': budget, 'stage_a_version': 1})
    context = None
    if req['mode'] == 'phira':
        context = project(scene, req, evidence, shift_us=shift_us,
                          reference_scenes={v.source_sha256: v for v in sources.values()},
                          memory_overflow=memory_overflow)
    return prepared, context


def build_triplet(sample, chart_events, c0_us, context=None, *, perturb_seed=None):
    """Same rhythm labels, known event identities, negatives and controls in three views."""
    examples = {arm: F.build_core(sample, chart_events, c0_us, sample.request['audio_duration_us'],
                                  stage_a_arm=arm, spatial_context=context,
                                  perturb=None if perturb_seed is None else np.random.default_rng(perturb_seed)) for arm in ARMS}
    def targets(ex):
        return digest([{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in g.items()
                        if k != 'right_new'} for g in ex.groups])
    if len({targets(ex) for ex in examples.values()}) != 1:
        raise AssertionError('Stage A arms changed rhythm supervision')
    for ex in examples.values():
        issues = F.check_targets(ex)
        if issues:
            raise F.SkipExample('common Stage A target validation: ' + ','.join(issues))
        ex.provenance['stage_a_common'] = {'request_sha256': digest(sample.request),
                                          'evidence_sha256': digest(sample.evidence),
                                          'rhythm_targets_sha256': targets(ex), 'core_start_us': c0_us}
    return examples


def warm_start(checkpoint, arm, *, seed=20261024):
    """Instantiate and validate weights only. Never creates/steps an optimizer."""
    if arm not in ARMS:
        raise ValueError('unknown Stage A arm')
    base = P1Config(**checkpoint['config'])
    if base.phira_location:
        raise ValueError('Stage A source must be a rhythm-only checkpoint')
    cfg = replace(base, phira_context=arm, selection=False, lane_history_groups=0, offset_sigma_ms=0.0)
    if base.selection or base.lane_history_groups:
        raise ValueError('use the common pre-intervention checkpoint, not a silently stripped model')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = P1Model(cfg)
    for name, weight in checkpoint['model'].items():
        if torch.is_tensor(weight) and weight.is_floating_point() and not torch.isfinite(weight).all():
            raise ValueError(f'nonfinite source weight: {name}')
    result = model.load_state_dict(checkpoint['model'], strict=False)
    expected = {k for k in model.state_dict() if k.startswith('space_encoder.')} if arm == 'operation' else set()
    if result.unexpected_keys or set(result.missing_keys) != expected:
        raise ValueError(f'checkpoint architecture mismatch: {result}')
    return model


def tensor_fingerprint(batch):
    h = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            h.update(json.dumps([key, str(array.dtype), array.shape]).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def prepare_core_schedule(specs, load_chart, preparer, *, salient=None):
    """Build a caller-chosen common schedule, without sampling replacements.

    Each spec contains a full corpus row, seed, and optional kind/core_start_us.
    Rejections are shared by all arms and returned for review before freezing.
    This does not select songs, fetch data, load audio, or run the model.
    """
    from ..events.requests import sample_request
    from ..model.batch import MAX_NOTES
    accepted, rejected, fingerprints = [], [], []
    common = protocol()['common']
    for ordinal, spec in enumerate(specs):
        row, seed = spec['row'], int(spec['seed'])
        identity = {'ordinal': ordinal, 'chart_id': row['chart_id'], 'seed': seed}
        try:
            chart = load_chart(row['chart_id'])
            duration = round(row['audio_sec'] * 1e6)
            sample = sample_request(chart, seed, row['audio_sha256'], duration,
                                    mix={spec['kind']: 1.0} if spec.get('kind') else common['request_mix'],
                                    control_drop_all_p=common['control_drop_all_p'],
                                    salient_candidates_us=salient(row['audio_sha256']) if salient else None)
            sample, context = preparer(row, chart, sample)
            c0 = spec.get('core_start_us')
            if c0 is None:
                c0 = F.choose_core(sample, np.random.default_rng([seed, 1]), duration)
            examples = build_triplet(sample, chart.events, c0, context, perturb_seed=[seed, 2])
            if any(len(g['notes']) > MAX_NOTES for ex in examples.values() for g in ex.groups):
                raise F.SkipExample('simultaneous group exceeds MAX_NOTES; no event truncation')
            identity.update(kind=sample.recipe['kind'], **examples['legacy'].provenance['stage_a_common'])
            for ex in examples.values():
                ex.provenance.update(identity)
            accepted.append({'song': row['audio_sha256'], 'examples': examples})
            fingerprints.append(identity)
        except (ValueError, AssertionError) as exc:
            rejected.append({**identity, 'reason': f'{type(exc).__name__}:{exc}'})
    audit = {'accepted': len(accepted), 'rejected': rejected, 'schedule': fingerprints,
             'schedule_sha256': digest(fingerprints), 'independent_arm_resampling': False,
             'training_updates': 0}
    return accepted, audit


class GeometryPreparer:
    """Shared training/evaluation adapter backed by caller-provided local sources.

    load_source(row) returns {raw: bytes, info_offset_s: float}; missing geometry
    raises instead of silently turning the operation arm into an ablation arm.
    Compile once per chart, project anew for every request/permission change.
    """
    def __init__(self, load_source, *, memory_overflow='error', apply_codec_constant=True, retain_observed_lines=False):
        self.load_source, self.memory_overflow = load_source, memory_overflow
        self.apply_codec_constant = apply_codec_constant
        self.retain_observed_lines = retain_observed_lines
        self.compiled = {}

    def __call__(self, row, chart, sample):
        if chart.mode != 'phira':
            return prepare_sample(sample)
        from ..spatial import compile_rpe
        from .data import model_shift_us
        source = self.load_source(row)
        key = (hashlib.sha256(source['raw']).hexdigest(), source.get('info_offset_s', 0.0))
        if key[0] != chart.source_sha256:
            raise ValueError('native geometry does not match the canonical source fingerprint')
        if key not in self.compiled:
            self.compiled[key] = compile_rpe(source['raw'], info_offset_s=key[1])
        scene = self.compiled[key]
        grants = []
        if self.retain_observed_lines:
            grants = [{'kind': 'retained_line_scene', 'source_sha256': scene.source_sha256,
                       'line_ids': sorted({e['location']['line_id'] for e in sample.request['observed_events']}),
                       'span_us': [0, sample.request['audio_duration_us']]}]
        return prepare_sample(sample, scene, shift_us=model_shift_us(row, apply_codec_constant=self.apply_codec_constant),
                              reference_sources={r['reference_id']: scene for r in sample.evidence},
                              memory_overflow=self.memory_overflow, grants=grants)


def local_geometry_source(root):
    """Loader for hash-verified sources produced by fetch_stage_a_sources.py."""
    from pathlib import Path
    root = Path(root)
    def load(row):
        cid = str(row['native_id'])
        source = json.loads((root / f'{cid}.source.json').read_text())
        raw = (root / f'{cid}.chart.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['source_sha256'] or source['audio_sha256'] != row['audio_sha256']:
            raise ValueError('local geometry identity mismatch')
        return {'raw': raw, 'info_offset_s': source['info_offset_s']}
    return load


class FrozenArmSampler:
    """CoreSampler-compatible view of an already prepared common core schedule.

    Each draw returns (CoreExample, SongCache). No per-arm rejection/resampling is
    allowed; preparation must record shared failures before freezing the schedule.
    """
    def __init__(self, schedule, arm, song_cache):
        if arm not in ARMS or not schedule:
            raise ValueError('a supported arm and nonempty frozen schedule are required')
        self.schedule, self.arm, self.song_cache, self.cursor = schedule, arm, song_cache, 0
        self.skipped = {}  # preparation rejections belong to the shared schedule audit

    def draw(self, count):
        if count <= 0 or self.cursor + count > len(self.schedule):
            raise ValueError('frozen Stage A schedule exhausted; no silent resampling')
        items = self.schedule[self.cursor:self.cursor + count]
        self.cursor += count
        return [(item['examples'][self.arm], self.song_cache(item['song'])) for item in items]
