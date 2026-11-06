"""Read original fixed terminals; never choose checkpoints or sampling draws."""
import argparse
import json
from pathlib import Path

from scripts.run_local_portfolio_scale import sha, write


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'bench/rhythm-v2/local-portfolio-final-selection-v1.json'


def entries():
    # id, report, metric path, plan, checkpoint, architecture, arm, steps
    yield ('ssd-reference1000', 'local_portfolio_mamba_quality_v1',
        ('evaluations', '1000', 'mamba', 'draft'), 'local-portfolio-mamba-quality-v1',
        'local-portfolio-v1/mamba-quality/mamba/quality-1000.pt', 'mamba', None, 1000)
    for arm in ('small24', 'small64'):
        yield (arm + '-2000', 'local_portfolio_data_scale_v1',
            ('evaluations', '2000', arm, 'draft'), 'local-portfolio-data-scale-v1',
            f'local-portfolio-v1/data-scale/{arm}/quality-2000.pt', 'canvas', arm, 2000)
    for name in ('mel', 'package'):
        yield (name + '-2000', f'local_portfolio_{name}_quality_v1',
            ('evaluations', '2000', 'draft'), f'local-portfolio-{name}-quality-v1',
            f'local-portfolio-v1/{name}-quality/quality-2000.pt', name, None, 2000)
    for arm in ('global', 'mixed'):
        yield ('access-' + arm, 'local_access_quality_v1',
            ('arms', arm, 'evaluations', '1000', 'draft'), 'local-access-comparison-v1',
            f'local-access-comparison-v1/{arm}/quality-1000.pt', 'access', arm, 1000)
    for ident, report, factory in (
        ('local-whole-song-arrangement-v1', 'local_whole_song_arrangement_v1', 'arrangement'),
        ('local-grouped-chart-v2', 'local_grouped_chart_v2', 'grouped_ar'),
        ('local-lattice-complete-v1', 'local_lattice_complete_v1', 'lattice'),
        ('local-event-code-flow-v2', 'local_event_code_flow_v2', 'flow'),
        ('local-ssm-event-package-v1', 'local_ssm_event_package_v1', 'ssm_package'),
        ('local-canonical-package-quality-v2', 'local_canonical_package_quality_v2', 'canonical'),
        ('local-noisy-proposal-refiner-v2', 'local_noisy_proposal_refiner_v2', 'refiner'),
        ('local-group-diffusion-quality-v1', 'local_group_diffusion_quality_v1', 'diffusion'),
    ):
        yield (ident, report, ('evaluations', 'draft'), ident,
               f'{ident}/quality-1000.pt', factory, None, 1000)
    for arm in ('full', 'short32'):
        yield ('direct-song-' + arm, 'local_mert_context_pair_v1',
            ('arms', arm, 'evaluations', 'draft'), 'local-mert-context-pair-v1',
            f'local-mert-context-pair-v1/{arm}/quality-1000.pt', 'direct_song', arm, 1000)
    for name, factory in (('attention', 'canvas'), ('concat', 'mamba')):
        ident = f'local-{name}-control-quality-v1'
        yield (ident, f'local_{name}_control_quality_v1',
            ('arms', 'full', 'evaluations', 'draft'), ident,
            f'{ident}/full/quality-1000.pt', factory, 'full', 1000)
    for kind, arms, factory in (('depth', ('wide8', 'deep16'), 'depth_budget'),
                               ('ssl', ('supervised', 'matching'), 'ssl_context')):
        ident = f'local-music-{kind}-pair-v1'
        for arm in arms:
            yield (f'music-{kind}-{arm}', f'local_music_{kind}_pair_v1',
                ('arms', arm, 'evaluations', 'draft'), ident,
                f'{ident}/{arm}/quality-1000.pt', factory, arm, 1000)


def eligible(metrics, policy):
    gate = policy['eligibility']
    return dict(
        complete=metrics['attempted'] == metrics['complete'] == gate['complete'],
        nonempty=metrics['nonempty'] >= gate['min_nonempty'],
        count=gate['count_ratio'][0] <= metrics['density_ratio'] <= gate['count_ratio'][1],
        full=metrics['full_note_f1'] >= gate['full_note_f1'],
        onset=metrics['onset10'] >= gate['onset10'],
        null=metrics['onset10'] - metrics['shift_null10'] >= gate['onset10_minus_shift'],
        components=sum(v['onset10'] > v['shift_null10'] for v in metrics['components'].values())
                   >= gate['min_positive_components'],
    )


def catalog():
    policy = json.loads(POLICY.read_text())
    result = []
    for ident, report, keys, plan, weight, factory, arm, steps in entries():
        rp = ROOT / f'reports/{report}.json'
        item = dict(id=ident, report=str(rp.relative_to(ROOT)),
                    plan=f'bench/rhythm-v2/{plan}.json', checkpoint=f'artifacts/{weight}',
                    factory=factory, arm=arm, steps=steps)
        if not rp.exists():
            result.append(dict(item, status='not_run', eligible=False))
            continue
        source = json.loads(rp.read_text())
        current = source
        try:
            for key in keys:
                current = current[key]
        except KeyError:
            result.append(dict(item, status=source['status'], eligible=False))
            continue
        # The predesignated SSD1000 reference is a real fixed checkpoint even
        # though the old parent was interrupted later and resumed separately.
        if ident != 'ssd-reference1000' and source['status'] != 'completed':
            result.append(dict(item, status=source['status'], eligible=False))
            continue
        gates = eligible(current, policy)
        parameters = source.get('parameters')
        if isinstance(parameters, dict):
            parameters = parameters[arm]
        if parameters is None and arm:
            parameters = source['arms'][arm].get('parameters')
        if parameters is None:
            parameters = json.loads((ROOT / item['plan']).read_text()).get('parameters')
            if isinstance(parameters, dict):
                parameters = parameters[arm]
        item.update(status='completed_fixed_terminal', parameters=parameters,
                    metrics=current, eligibility=gates, eligible=all(gates.values()),
                    report_sha256=sha(rp), checkpoint_sha256=sha(ROOT / item['checkpoint']),
                    plan_sha256=sha(ROOT / item['plan']))
        result.append(item)
    return dict(id='local-portfolio-terminal-catalog-v1', policy_sha256=sha(POLICY),
                candidates=result, confirmation_scored=False, cloud_compute_calls=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', action='store_true')
    args = parser.parse_args()
    value = catalog()
    if args.select:
        queue = json.loads((ROOT / 'reports/local-portfolio-v1/remaining-pool-v4.json').read_text())
        if queue['status'] != 'completed':
            raise ValueError('All pending real trials must finish before selecting')
        if any(c['status'] != 'completed_fixed_terminal' for c in value['candidates']):
            raise ValueError('Incomplete candidate requires explicit root disposition')
        eligible_rows = [c for c in value['candidates'] if c['eligible']]
        eligible_rows.sort(key=lambda c: (-c['metrics']['full_note_f1'], -c['metrics']['onset10'],
                                           c['parameters'], c['id']))
        policy = json.loads(POLICY.read_text())
        finalists = eligible_rows[:policy['selection']['max_finalists']]
        selection = dict(policy_sha256=sha(POLICY), finalist_identities=finalists,
                         second_seed_by_rank=policy['second_seed']['seeds_by_rank'][:len(finalists)],
                         status='identities_frozen_before_replication_and_confirmation',
                         confirmation_scored=False, cloud_compute_calls=0)
        target = ROOT / 'reports/local_portfolio_finalist_selection_v1.json'
        if target.exists():
            raise FileExistsError('Never replace finalist selection after outcomes')
        write(target, selection)
    write(ROOT / 'reports/local_portfolio_terminal_catalog_v1.json', value)
    for c in value['candidates']:
        m = c.get('metrics', {})
        print(c['id'], c['status'], c['eligible'],
              m.get('full_note_f1'), m.get('onset10'))


if __name__ == '__main__':
    main()
