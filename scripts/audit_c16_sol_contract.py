"""Parent pre-audit of C16-S: byte identity and finite probability arithmetic only.

No project/array/ML imports, weights, model execution, new event outputs, or training.
This is not the production implementation or a training-admission certificate.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'reports/council-inventory-c16'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(name, value):
    with (FOLDER / name).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def softmax(scores):
    maximum = max(scores)
    weights = [math.exp(v - maximum) for v in scores]
    return [w / sum(weights) for w in weights]


def logsumexp(values):
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


def composition_dp(potentials, cap, allowed):
    """Log sum and lexicographically first max for each legal total."""
    log_z = [0.] + [-math.inf] * cap
    best = [(0., ())] + [None] * cap
    for row, enabled in zip(potentials, allowed):
        new_z, new_best = [], []
        for total in range(cap + 1):
            legal = range(total + 1) if enabled else (0,)
            new_z.append(logsumexp([
                log_z[total - count] + row[count] for count in legal
                if log_z[total - count] != -math.inf
            ]))
            choices = [
                (best[total - count][0] + row[count],
                 best[total - count][1] + (count,))
                for count in legal if best[total - count] is not None
            ]
            new_best.append(min(choices, key=lambda v: (-v[0], v[1])) if choices else None)
        log_z, best = new_z, new_best
    return log_z, best


def enumerated(potentials, cap, allowed):
    rows = []
    for vector in product(range(cap + 1), repeat=len(potentials)):
        if sum(vector) > cap or any(v and not a for v, a in zip(vector, allowed)):
            continue
        rows.append((vector, sum(potentials[k][n] for k, n in enumerate(vector))))
    return rows


def probability_checks():
    cap = 3
    potentials = [[.17 * (k + 1) * n - .13 * n * n for n in range(cap + 1)]
                  for k in range(4)]
    total_scores = [.2, .8, -.3, -.6, -2.]  # last class is explicit overflow
    q_total = softmax(total_scores)
    checks = []
    for allowed in [(True,) * 4, (True, False, True, False), (False, True, False, False)]:
        rows = enumerated(potentials, cap, allowed)
        log_z, best = composition_dp(potentials, cap, allowed)
        mass = {}
        for total in range(cap + 1):
            group = [(v, s) for v, s in rows if sum(v) == total]
            assert abs(log_z[total] - logsumexp([s for _, s in group])) < 1e-12
            assert best[total] == min([(s, v) for v, s in group], key=lambda x: (-x[0], x[1]))
            for vector, score in group:
                mass[vector] = q_total[total] * math.exp(score - log_z[total])
        total_mass = sum(mass.values()) + q_total[-1]
        assert abs(total_mass - 1.) < 1e-12
        for total in range(cap + 1):
            assert abs(sum(p for v, p in mass.items() if sum(v) == total) - q_total[total]) < 1e-12
        for k in range(4):
            marginal = [sum(p for v, p in mass.items() if v[k] == n) for n in range(cap + 1)]
            assert abs(sum(marginal) + q_total[-1] - 1.) < 1e-12
            if not allowed[k]:
                assert all(abs(v) < 1e-12 for v in marginal[1:])
        assert abs(sum(sum(v) * p for v, p in mass.items()) -
                   sum(n * q_total[n] for n in range(cap + 1))) < 1e-12
        # Exact N=0 cancellation, and finite-difference checks of conditional gradients.
        max_gradient_error = 0.
        for target in [(0, 0, 0, 0), next(v for v, _ in rows if sum(v) == 2)]:
            total = sum(target)
            group = [(v, s) for v, s in rows if sum(v) == total]
            for k in range(4):
                for n in range(cap + 1):
                    expected = sum(math.exp(s - log_z[total]) for v, s in group if v[k] == n)
                    analytic = expected - float(target[k] == n)
                    values = []
                    for epsilon in [-1e-5, 1e-5]:
                        changed = [r[:] for r in potentials]
                        changed[k][n] += epsilon
                        z, _ = composition_dp(changed, cap, allowed)
                        values.append(z[total] - sum(changed[j][v] for j, v in enumerate(target)))
                    numerical = (values[1] - values[0]) / 2e-5
                    error = abs(numerical - analytic)
                    max_gradient_error = max(max_gradient_error, error)
                    assert error < 1e-8
                    if total == 0 or sum(allowed) == 1:
                        assert abs(analytic) < 1e-12
            if total == 0:
                assert abs(log_z[total] - sum(potentials[k][0] for k in range(4))) < 1e-12
        # At fixed x, choosing q(N) equal to the old marginal recovers the old density.
        old_normalizer = logsumexp([s for _, s in rows])
        overflow = .07
        identity_error = max(abs(
            (1. - overflow) * math.exp(score - old_normalizer) -
            ((1. - overflow) * math.exp(log_z[sum(v)] - old_normalizer)) *
            math.exp(score - log_z[sum(v)])
        ) for v, score in rows)
        assert identity_error < 1e-12
        checks.append(dict(allowed=allowed, vectors=len(rows), normalized_mass=total_mass,
                           max_gradient_error=max_gradient_error,
                           old_density_factorization_error=identity_error))
    # Same probability distribution, different actions: vector MAP vs total-mode then MAP.
    bernoulli = [(v, .3 ** sum(v) * .7 ** (4 - sum(v)))
                 for v in product((0, 1), repeat=4)]
    old_mode = min(bernoulli, key=lambda x: (-x[1], x[0]))[0]
    marginal = [sum(p for v, p in bernoulli if sum(v) == n) for n in range(5)]
    n_mode = max(range(5), key=lambda n: marginal[n])
    hierarchical = min([(v, p) for v, p in bernoulli if sum(v) == n_mode],
                       key=lambda x: (-x[1], x[0]))[0]
    assert sum(old_mode) == 0 and sum(hierarchical) == 1
    return dict(
        scope='finite synthetic arithmetic only; no production model or new chart generation',
        cases=checks,
        policy_counterexample=dict(distribution='four independent Bernoulli(0.3)',
                                   vector_mode=old_mode, total_marginal=marginal,
                                   total_mode=n_mode, conditional_mode=hierarchical),
        conclusion='The proposed factorization can be coherent; the hierarchical decision alone '
                   'can increase output quantity without a new learned distribution.',
        limits=['Fixed-score factorization identity does not prove nesting of the neural model families.',
                'A/C audio controls do not isolate factorization from the decision-policy change.',
                'No production-scale gradient, MPS, real-target support, prior, resource, '
                'or learning admission was run.',
                'Typed marginals must integrate over predicted total; overflow remains explicit.'],
    )


def coverage_audit():
    result = {}
    for arm in ('A', 'C'):
        for task in ('draft', 'audio_shift', 'audio_foreign', 'audio_zero'):
            path = ROOT / f'artifacts/c15-typed-inventory-v1/{arm}-{task}.json'
            rows = read(path)
            values = {}
            for feature in ('local', 'mel', 'song'):
                values[feature] = dict(
                    core_min=min(r['control']['core_coverage'][feature]['fraction'] for r in rows),
                    full_input_min=min(r['control'].get('coverage', {}).get(feature, 1.) for r in rows),
                )
            result[f'{arm}-{task}'] = dict(
                path=str(path.relative_to(ROOT)), sha256=sha(path), requests=len(rows),
                components=len({r['component'] for r in rows}), coverage=values,
            )
    return dict(
        cases=result,
        supported='Raw-input dependence and paired count-probability evidence within the recorded support.',
        not_identified='Pure music semantics, mood, or equal-full-context content replacement; '
                       'foreign song-range coverage and shift edges also change.',
        old_evidence_changed=False, new_model_calls=0,
    )


def main():
    started = time.monotonic()
    outputs = ['parent-material-verification-S.json', 'parent-S-contract-audit.json']
    assert all(not (FOLDER / name).exists() for name in outputs), 'Never overwrite an earlier certificate'
    manifest_path = FOLDER / 'material-manifest.json'
    manifest = read(manifest_path)
    audit = read(FOLDER / 'sol-access-audit.json')
    audited = {row['path']: row for row in audit['objects']}
    verified, bytes_checked = [], 0
    assert len(audited) == len(manifest['files']) == 1071
    for row in manifest['files']:
        path = ROOT / row['path']
        assert path.resolve().is_relative_to(ROOT.resolve())
        size, digest = path.stat().st_size, sha(path)
        a = audited[row['path']]
        assert size == row['bytes'] == a['expected_bytes'] == a['actual_bytes']
        assert digest == row['sha256'] == a['expected_sha256'] == a['actual_sha256']
        assert a['verified']
        bytes_checked += size
        verified.append(dict(path=row['path'], bytes=size, sha256=digest))
    assert sha(manifest_path) == audit['manifest']['sha256']
    assert sha(FOLDER / 'sol.md') == audit['review_output']['sha256']
    terminal = read(FOLDER / 'terminal-S.json')['result']['structuredContent']
    assert terminal['status'] == 'completed' and not terminal['hasPendingChildRuns']
    assert terminal['model'] == 'gpt-6.1-sol'
    assert {'id': 'reasoningEffort', 'value': 'ultra'} in terminal['options']
    first = read(FOLDER / 'prompt-receipt-S.json')['result']['structuredContent']['items'][0]
    delivered = (FOLDER / 'delivered-prompt-S.txt').read_text()
    assert not first['textTruncated'] and first['text'] == delivered
    assert delivered == 'Act as the review sub-agent for this task.\n\n' + (FOLDER / 'prompt-S.txt').read_text()
    assert (audit['summary']['verified_objects'], audit['summary']['verified_bytes']) == (1071, bytes_checked)
    reading = Counter(row['highest_reading_level'] for row in audit['objects'])
    assert dict(reading) == audit['summary']['highest_reading_levels']
    write_new(outputs[0], dict(
        recorded_at=datetime.now(timezone.utc).isoformat(), status='identity_verified',
        manifest_sha256=sha(manifest_path), objects=len(verified), bytes=bytes_checked,
        mismatches=0, source_report_sha256=sha(FOLDER / 'sol.md'),
        reviewer_reading_levels=dict(reading),
        reading_scope='Parent rehashed all objects; this pass does not claim semantic reading of all objects.',
        delivered_prompt_exact=True, terminal_settings_verified=True,
        claimed_reviewer_execution_boundary=audit['execution_boundary_verification'],
        boundary_limit='Self-reported access audit and visible receipts, not OS sandbox attestation.',
        files=verified,
    ))
    write_new(outputs[1], dict(
        recorded_at=datetime.now(timezone.utc).isoformat(), recommendation='C16-S-R01',
        status='parent_math_preaudit_only', training_authorized=False,
        probability_arithmetic=probability_checks(), coverage_scope=coverage_audit(),
        elapsed_wall_s=time.monotonic() - started,
        new_neural_forwards=0, new_optimizer_updates=0, cloud_calls=0,
    ))
    print(dict(status='preaudit_passed_not_training_admission', objects=len(verified),
               bytes=bytes_checked, wall_s=round(time.monotonic() - started, 3),
               new_neural_forwards=0, new_optimizer_updates=0, cloud_calls=0))


if __name__ == '__main__':
    main()
