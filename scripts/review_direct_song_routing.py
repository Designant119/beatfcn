"""Review the previously frozen same-weight routing contrast, without ML."""
import json
from pathlib import Path

from scripts.run_local_portfolio_scale import sha, write


ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/local_direct_song_routing_review_v1.json'
    if output.exists():
        raise FileExistsError('Preserve the original routing review')
    contract_path = ROOT / 'bench/rhythm-v2/local-direct-song-routing-v1.json'
    contract = json.loads(contract_path.read_text())
    specs = {
        'direct': ('local_mert_context_pair_v1', 'local-mert-context-pair-v1'),
        'concat': ('local_concat_control_quality_v1', 'local-concat-control-quality-v1'),
    }
    rows = {}
    identities = {}
    for arm, (report_name, ident) in specs.items():
        report_path = ROOT / f'reports/{report_name}.json'
        report = json.loads(report_path.read_text())
        row = report['arms']['full']
        if report['status'] != 'completed' or row['status'] != 'completed':
            raise ValueError('No review before actual fixed terminals')
        if row['health_updates'] != 100 or row['quality_updates'] != 1000:
            raise ValueError('Incomplete independent trials')
        if row['exposure']['windows'] != 4000:
            raise ValueError('Actual exposure differs')
        checkpoint = ROOT / f'artifacts/{ident}/full/quality-1000.pt'
        if sha(checkpoint) != row['final_weight_sha256']:
            raise ValueError('Checkpoint identity changed')
        if row['evaluations']['draft']['attempted'] != 32:
            raise ValueError('Missing primary requests')
        rows[arm] = row
        identities[arm] = dict(report=str(report_path.relative_to(ROOT)), report_sha256=sha(report_path),
            plan=f'bench/rhythm-v2/{ident}.json', plan_sha256=sha(ROOT / f'bench/rhythm-v2/{ident}.json'),
            checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint))
    for field in ('parameters', 'initial_weight_sha256', 'actual_supervision_and_common_mel_sha256'):
        if rows['direct'][field] != rows['concat'][field]:
            raise ValueError('Routing contrast mismatch: ' + field)
    d, c = [rows[k]['evaluations']['draft'] for k in ('direct', 'concat')]
    if set(d['components']) != set(c['components']):
        raise ValueError('Different evaluation components')
    gains = dict(full_note_f1=d['full_note_f1'] - c['full_note_f1'],
                 onset10=d['onset10'] - c['onset10'],
                 component_full_note_wins=sum(d['components'][k]['full_note_f1'] > v['full_note_f1']
                                              for k, v in c['components'].items()))
    gate = contract['gate']
    gates = dict(full=gains['full_note_f1'] >= gate['full_note_f1_gain'],
                 onset=gains['onset10'] >= -gate['onset10_max_regression'],
                 components=gains['component_full_note_wins'] >= gate['min_component_full_note_wins'])
    result = dict(id='local-direct-song-routing-review-v1', status='completed',
        contract_sha256=sha(contract_path), identities=identities,
        equal_parameters=True, equal_initial_tensors=True, equal_actual_supervision_and_common_mel=True,
        gains=gains, gates=gates, hypothesis_supported=all(gates.values()),
        metrics={k: r['evaluations']['draft'] for k, r in rows.items()},
        structure={k: r['structure'] for k, r in rows.items()},
        global_controls={k: r['global_controls'] for k, r in rows.items()},
        scope='One fixed-seed same-parameter routing contrast. Separate from the same-encoder full/32s extraction pair. Input dependence is not useful arrangement, emotion understanding or a human editing benefit; no isolated width or batch-size causal claim.',
        cloud_compute_calls=0)
    write(output, result)
    text = ['# 整曲读出路由配对', '',
            f'固定合同支持：{result["hypothesis_supported"]}；门：{gates}。', '',
            f'完整事件 F1 增益 {gains["full_note_f1"]:.6f}；onset10 增益 {gains["onset10"]:.6f}；'
            f'组件完整事件胜出 {gains["component_full_note_wins"]}/16。', '',
            '两臂参数、初始张量、4000 个实际监督请求及共同 mel 身份一致。'
            '整曲移除/错配和两秒数量轨迹单列于 JSON；不能把输入依赖写成音乐结构或情绪理解。', '']
    output.with_suffix('.md').write_text('\n'.join(text))
    print(json.dumps(dict(gains=gains, gates=gates, hypothesis_supported=result['hypothesis_supported'])))


if __name__ == '__main__':
    main()
