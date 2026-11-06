"""Close the two completed canonical-package trials from immutable raw results."""
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KEYS = ('full_note_f1', 'onset10', 'onset50', 'shift_null10', 'density_ratio',
        'complete', 'attempted', 'nonempty')


def main():
    for ident, report_name in (
        ('local-ssm-event-package-v1', 'local_ssm_event_package_v1'),
        ('local-canonical-package-quality-v2', 'local_canonical_package_quality_v2'),
    ):
        path = ROOT / f'reports/{report_name}.json'
        report = json.loads(path.read_text())
        if report['status'] != 'completed' or report['quality_updates'] != 1000:
            raise ValueError('A real fixed terminal is required')
        art = ROOT / 'artifacts' / ident
        records = json.loads((art / 'draft-final.json').read_text())
        predicted = Counter(e['onset_us'] % 40_000 // 1000
                            for r in records for e in r['prediction']['events'])
        target = Counter(e['onset_us'] % 40_000 // 1000
                         for r in records for e in r['target'])
        ev = report['evaluations']
        secondary = {k: sum(ev[f'ancestral_{s}'][k] for s in range(4)) / 4
                     for k in KEYS}
        sources = [path, ROOT / f'bench/rhythm-v2/{ident}.json',
                   art / 'quality-1000.pt', art / 'draft-final.json']
        result = dict(
            id=ident + '-review', status='closed_complete_recipe_failed',
            parameters=report['parameters'], health_updates=100,
            quality_updates=1000, health_weights_used=False,
            original_health_status=report.get('original_health_status'),
            admission_after_observation=report.get('admission_after_observation', False),
            primary={k: ev['draft'][k] for k in KEYS},
            local_screen=report['local_screen'],
            primary_per_type=ev['draft']['per_type'],
            secondary_four_T1_mean=secondary,
            completion={k: ev['completion'][k] for k in KEYS},
            reference_masked={k: ev['reference_masked'][k] for k in KEYS},
            all_acoustic_mismatch={k: ev['audio_swap'][k] for k in KEYS},
            primary_empty_events=sum(ev['empty']['predicted_types'].values()),
            individual_modes_nonempty=ev['individual_mode']['nonempty'],
            phase_ms_mod40=dict(predicted=predicted.most_common(), target=target.most_common()),
            training_wall_s=report.get('quality_training_wall_s', report.get('training_wall_s')),
            evaluation_wall_s=report['evaluation_wall_s'],
            total_quality_wall_s=report['total_quality_wall_s'],
            decision='Close this full recipe: both preregistered .20 placement/full-event screens fail. No extra steps, threshold search, best-draw selection, confirmation or cloud.',
            observations='Global cardinality recovers a plausible total, but greedy conditional allocation has strong phase/type concentration; all individual cell/type modes remain empty. T1 is archived as secondary and cannot replace the declared primary after scoring.',
            boundaries='No isolated objective/backbone/GRU causal conclusion. Audio mismatch is an input dependency check, not proof of rhythm, structure or emotion understanding. Original health failures are retained unchanged; the A2 admission revision was made after observing health, before its fresh quality trial.',
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources},
            cloud_compute_calls=0, confirmation_scored=False,
        )
        out = ROOT / f'reports/{report_name.replace("_v1", "_review_v1").replace("_v2", "_review_v2")}.json'
        if out.exists():
            raise FileExistsError(out)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        out.with_suffix('.md').write_text(
            f'# {ident} 完整终态结案\n\n'
            f'100 步健康与 fresh 1000 步质量分开。主读出 full F1 '
            f'{ev["draft"]["full_note_f1"]:.6f}，onset10 '
            f'{ev["draft"]["onset10"]:.6f}，数量比 '
            f'{ev["draft"]["density_ratio"]:.6f}。两个预注册 0.20 筛选均未通过。\n\n'
            f'四次 T1 的次要平均 full F1 {secondary["full_note_f1"]:.6f}、'
            f'onset10 {secondary["onset10"]:.6f}；不事后替换主读出。'
            f'两个空窗发出 {result["primary_empty_events"]} 个事件；'
            f'逐格众数的 32 个开发窗仍全空。相位与类型、参考屏蔽及完整声学错配均见同名 JSON。\n\n'
            '关闭本次完整配方，不追加步数，不以健康门或概率总量代替独立歌曲定位。'
            '该结论不单独否定 GRU、计数似然、Mamba 或 attention 方法族；'
            'A 的原健康失败及事后工程入场修订均保留。确认集未打开，云调用为零。\n'
        )
        print(out.relative_to(ROOT))


if __name__ == '__main__':
    main()
