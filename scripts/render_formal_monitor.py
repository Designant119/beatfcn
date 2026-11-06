"""Render a self-refreshing HTML monitor for the formal MERT run + mel verification."""
import argparse
import ast
import html as html_lib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_RECEIPT = ROOT / 'reports/full-database-cloud-v1/mert-eager-formal-run-v7.json'
V2_LOG = ROOT / 'artifacts/full-database-mert-eager-formal-v7.log'
MEL_RECEIPT = ROOT / 'reports/full-database-cloud-v1/mel-array-content-receipt-v1.json'
MEL_PARTIAL = ROOT / 'inputs/full-database-cloud-v1/mel-array-content.jsonl.partial'
LEDGER = ROOT / 'reports/rhythm_v2_budget.json'
HISTORY = ROOT / 'artifacts/formal-monitor-history.json'
DEST = ROOT / 'artifacts/formal-monitor.html'
GRANT = 'owner-full-database-mert-pilot-20261030'
PLANNED_FRESH = 9817
SOFT_DEADLINE_S = 2050 + 80
THEME = dict(bg='#0d1117', card='#161b22', grid='#21262d', text='#c9d1d9',
    dim='#8b949e', accent='#58a6ff', good='#3fb950', warn='#d29922', bad='#f85149')


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def parse_milestones():
    found = []
    try:
        for line in V2_LOG.read_text().splitlines():
            line = line.strip()
            if line.startswith("{'stage': 'mert-eager'"):
                try:
                    found.append(ast.literal_eval(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return found


def tail_log(lines=40):
    try:
        rows = V2_LOG.read_text().splitlines()[-lines:]
    except FileNotFoundError:
        return '(log 尚未创建)'
    return html_lib.escape('\n'.join(rows))


def cpu_series(milestones, cores=12):
    """Per-window CPU% estimate: decode+write aggregate task time per wall second / cores."""
    points = []
    for previous, current in zip(milestones, milestones[1:]):
        dt = current.get('wall_s', 0) - previous.get('wall_s', 0)
        if dt <= 0.5:
            continue
        stages = current.get('stage_totals') or {}
        prior = previous.get('stage_totals') or {}
        work = ((stages.get('decode_s', 0) - prior.get('decode_s', 0))
            + (stages.get('write_s', 0) - prior.get('write_s', 0)))
        points.append(min(work / dt / cores * 100, 100))
    return points


def fmt_money(value):
    try:
        return f'${float(value):.3f}'
    except Exception:
        return '-'


def fmt_s(seconds):
    try:
        seconds = int(float(seconds))
        return f'{seconds // 60}m{seconds % 60:02d}s'
    except Exception:
        return '-'


def polyline_chart(series, *, width=560, height=200, threshold=None, ymax=None,
                   xlabel='', ylabel='', fill=True):
    """series: list of (index, value) or plain values; renders inline SVG."""
    values = [v for _, v in series] if series and isinstance(series[0], tuple) else list(series)
    svg = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto">']
    svg.append(f'<rect width="{width}" height="{height}" fill="{THEME["card"]}"/>')
    left, top, right, bottom = 44, 12, width - 10, height - 26
    for i in range(5):
        y = top + (bottom - top) * i / 4
        svg.append(f'<line x1="{left}" y1="{y:.0f}" x2="{right}" y2="{y:.0f}" '
                   f'stroke="{THEME["grid"]}" stroke-width="1"/>')
        label = (ymax * (1 - i / 4)) if ymax else None
        if label is None and values:
            label = (max(values) or 1) * (1 - i / 4)
        if label is not None:
            svg.append(f'<text x="{left - 6}" y="{y + 4:.0f}" fill="{THEME["dim"]}" '
                       f'font-size="10" text-anchor="end">{label:.0f}</text>')
    if not values:
        svg.append(f'<text x="{(left + right) // 2}" y="{(top + bottom) // 2}" '
                   f'fill="{THEME["dim"]}" font-size="12" text-anchor="middle">等待数据…</text>')
        svg.append('</svg>')
        return ''.join(svg)
    peak = ymax or max(values + ([threshold] if threshold else [])) or 1
    peak = peak * 1.05
    def x_at(i):
        return left + (right - left) * (i / max(len(values) - 1, 1))
    def y_at(v):
        return bottom - (bottom - top) * min(v / peak, 1.0)
    points = ' '.join(f'{x_at(i):.1f},{y_at(v):.1f}' for i, v in enumerate(values))
    if fill:
        svg.append(f'<polygon points="{left},{bottom} {points} {x_at(len(values) - 1):.1f},'
                   f'{bottom}" fill="{THEME["accent"]}" opacity="0.12"/>')
    color = THEME['good']
    if threshold is not None and values[-5:]:
        recent = sum(values[-5:]) / len(values[-5:])
        color = THEME['good'] if recent >= threshold else (
            THEME['warn'] if recent >= threshold * 0.8 else THEME['bad'])
    svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    if threshold is not None:
        ty = y_at(threshold)
        svg.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{right}" y2="{ty:.1f}" '
                   f'stroke="{THEME["bad"]}" stroke-width="1.5" stroke-dasharray="6 4"/>')
        svg.append(f'<text x="{right - 4}" y="{ty - 5:.1f}" fill="{THEME["bad"]}" '
                   f'font-size="10" text-anchor="end">{threshold}% 门</text>')
    svg.append(f'<text x="{left}" y="{height - 8}" fill="{THEME["dim"]}" '
               f'font-size="10">{xlabel}</text>')
    svg.append(f'<text x="{width - 10}" y="{height - 8}" fill="{THEME["dim"]}" '
               f'font-size="10" text-anchor="end">{ylabel} 最新 {values[-1]:.1f}'
               f' · 均值 {sum(values) / len(values):.1f}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def bars(segments, width=560, height=90):
    total = sum(v for _, v, _ in segments) or 1
    svg = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto">']
    svg.append(f'<rect width="{width}" height="{height}" fill="{THEME["card"]}" rx="8"/>')
    x = 8
    for label, value, color in segments:
        w = (width - 16) * value / total
        if w > 0.5:
            svg.append(f'<rect x="{x:.1f}" y="14" width="{w:.1f}" height="26" '
                       f'fill="{color}" rx="4"/>')
            svg.append(f'<text x="{x + w / 2:.1f}" y="31" fill="#0d1117" font-size="11" '
                       f'font-weight="bold" text-anchor="middle">{fmt_money(value)}</text>')
            svg.append(f'<text x="{x + w / 2:.1f}" y="60" fill="{THEME["dim"]}" '
                       f'font-size="10" text-anchor="middle">{label}</text>')
            x += w
    svg.append(f'<text x="8" y="82" fill="{THEME["dim"]}" font-size="10">'
               f'总 cap {fmt_money(sum(v for _, v, _ in segments))}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def card(title, value, sub='', color=None):
    color = color or THEME['text']
    return (f'<div class="card"><div class="ctitle">{title}</div>'
            f'<div class="cvalue" style="color:{color}">{value}</div>'
            f'<div class="csub">{sub}</div></div>')


def render():
    v2 = load_json(V2_RECEIPT)
    ledger = load_json(LEDGER)
    grant = (ledger.get('data_preparation_authorizations') or {}).get(GRANT, {})
    mel = load_json(MEL_RECEIPT)
    milestones = parse_milestones()
    now = time.time()

    history = load_json(HISTORY).get('points', [])
    live = None
    try:
        live = float(v2['usage']['live_upper_usd'])
    except Exception:
        pass
    completed = milestones[-1]['completed'] if milestones else 0
    try:
        mel_done = int(mel.get('verified') or 0) or sum(
            1 for _ in MEL_PARTIAL.open()) if MEL_PARTIAL.exists() else 0
    except Exception:
        mel_done = 0
    history.append(dict(t=now, live=live, completed=completed, mel=mel_done))
    history = history[-1500:]
    HISTORY.parent.mkdir(exist_ok=True)
    HISTORY.write_text(json.dumps(dict(points=history)))

    status = v2.get('status') or '启动中'
    started = v2.get('started_at_unix')
    elapsed = now - started if started else None
    summary = v2.get('result_summary') or {}
    last = milestones[-1] if milestones else {}
    gpu = last.get('gpu_monitor') or {}
    intervals = gpu.get('raw_interval_means') or []
    stage_totals = last.get('stage_totals') or {}
    throughput = summary.get('throughput') or {}
    asps = throughput.get('audio_seconds_per_second')

    eta = None
    if completed and elapsed and completed < PLANNED_FRESH:
        rate = completed / max(elapsed - 60, 1)
        eta = (PLANNED_FRESH - completed) / max(rate, 1e-9)

    committed = grant.get('committed_upper_usd')
    remaining = grant.get('remaining_usd')
    cap = grant.get('new_usd_cap')

    completed_series = [p['completed'] for p in history if p.get('completed') is not None]
    live_series = [p['live'] for p in history if p.get('live') is not None]
    mel_series = [p['mel'] for p in history if p.get('mel') is not None]

    header_color = {'formal_acceptance_passed': THEME['good'], 'failed': THEME['bad']}.get(
        status, THEME['accent'])
    cards = ''.join([
        card('运行状态', status,
            f"elapsed {fmt_s(elapsed or 0)} / budget {fmt_s(SOFT_DEADLINE_S)}", header_color),
        card('完成歌曲', f'{completed} / {PLANNED_FRESH}',
            f"ETA {fmt_s(eta) if eta else '-'}" + (f" · {asps:.0f} audio-s/s" if asps else '')),
        card('GPU 稳态均值',
            f"{gpu.get('stable_period_mean_utilization_percent', '-')}",
            f"last8 {gpu.get('last_eight_interval_mean_utilization_percent', '-')} · "
            f"过 480s 后强制 85%"),
        card('账本剩余', fmt_money(remaining),
            f"committed {fmt_money(committed)} / cap {fmt_money(cap)}"),
    ])
    stage_rows = ''.join(
        f'<tr><td>{name}</td><td>{fmt_s(value)}</td></tr>'
        for name, value in [
            ('decode 聚合', stage_totals.get('decode_s')),
            ('encode 聚合', stage_totals.get('encode_s')),
            ('write 聚合', stage_totals.get('write_s')),
            ('wall', last.get('wall_s'))] if value is not None)

    quarantine = summary.get('quarantined')
    cpu_points = cpu_series(milestones)
    log_tail = tail_log()
    html = f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>BeatFCN 正式跑监控</title><style>
body{{background:{THEME['bg']};color:{THEME['text']};font-family:-apple-system,'PingFang SC',sans-serif;
margin:0;padding:18px}} h1{{font-size:18px;margin:0 0 4px}} .sub{{color:{THEME['dim']};font-size:12px;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}}
.card{{background:{THEME['card']};border:1px solid {THEME['grid']};border-radius:10px;padding:12px}}
.ctitle{{color:{THEME['dim']};font-size:11px}} .cvalue{{font-size:22px;font-weight:700;margin:4px 0}}
.csub{{color:{THEME['dim']};font-size:11px}} .panel{{background:{THEME['card']};border:1px solid {THEME['grid']};
border-radius:10px;padding:12px;margin-bottom:12px}} .ptitle{{font-size:13px;font-weight:600;margin-bottom:8px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px}} td{{padding:3px 6px;border-bottom:1px solid {THEME['grid']}}}
.note{{color:{THEME['dim']};font-size:11px;margin-top:8px}}
pre{{margin:0;white-space:pre-wrap;word-break:break-all;font-size:10.5px;line-height:1.5;
color:{THEME['text']};max-height:340px;overflow-y:auto;font-family:ui-monospace,Menlo,monospace}}
</style></head><body>
<h1>BeatFCN 全库 MERT 正式跑 · v7 实时监控</h1>
<div class="sub">full-db-mert-eager-formal-v7 · A10G × 2 eager streams · 每 10s 自动刷新 ·
生成于 {time.strftime('%H:%M:%S')}</div>
<div class="grid">{cards}</div>
<div class="cols">
<div class="panel"><div class="ptitle">GPU 利用率（30s 区间均值 · nvidia 采样）</div>
{polyline_chart(intervals, threshold=85, ymax=100, xlabel='区间序号', ylabel='util %')}
<div class="note">红线 = 85% 验收门；480s 后双均值（总计 + last-8）均须 ≥85% · 当前区间数 {len(intervals)}</div></div>
<div class="panel"><div class="ptitle">CPU 利用率（decode+write 并行度 / 12 核折算）</div>
{polyline_chart(cpu_points, threshold=100, ymax=100, xlabel='里程碑窗口', ylabel='CPU %')}
<div class="note">decode worker 32 × ffmpeg 子进程近似满核即 ~100%；setup 阶段无里程碑数据</div></div>
<div class="panel"><div class="ptitle">完成歌曲数（累计）</div>
{polyline_chart(completed_series, xlabel='监控采样点', ylabel=f'首 · 目标 {PLANNED_FRESH}')}
<div class="note">每 16 首提交一次 Volume；resume 安全 · {completed and f'剩余 {PLANNED_FRESH - completed} 首' or '等待首里程碑'}</div></div>
<div class="panel"><div class="ptitle">v7 计费上界走势 / mel 校验进度</div>
{polyline_chart(live_series, xlabel='监控采样点', ylabel='USD')}
<div class="note">预留 {fmt_money(1.1223)} · mel verified {mel_done}/{mel.get('total', 10898)} ·
failures {len(mel.get('failures', []))}</div></div>
</div>
<div class="panel"><div class="ptitle">预算分布</div>
{bars([('已承诺(探针链+v1)', float(committed or 0), THEME['warn']),
       ('v7 已计费上界', live or 0, THEME['accent']),
       ('剩余', max(float(remaining or 0) - (live or 0), 0), THEME['good'])])}
<div class="note">cap {fmt_money(cap)} · 第四次 Owner 追加 +$1.5 已入账（15→16.5）</div></div>
<div class="panel"><div class="ptitle">阶段耗时（最新里程碑聚合）</div>
<table>{stage_rows or '<tr><td>等待首个里程碑…</td></tr>'}</table>
<div class="note">milestones {len(milestones)} · quarantine {quarantine if quarantine is not None else '(运行中)'} ·
ineligible {summary.get('ineligible', '(运行中)')}</div></div>
<div class="panel"><div class="ptitle">云端日志尾流（最近 40 行 · 实时）</div>
<pre>{log_tail}</pre>
</div>
</body></html>'''
    DEST.write_text(html)
    return dict(status=status, elapsed=elapsed, completed=completed,
        intervals=len(intervals), cpu_points=len(cpu_points), mel=mel_done, live=live)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--interval', type=int, default=10)
    arguments = parser.parse_args()
    if not arguments.serve:
        print(json.dumps(render(), ensure_ascii=False))
        return
    stop_file = ROOT / 'artifacts/formal-monitor.stop'
    while not stop_file.exists():
        try:
            render()
        except Exception as exc:
            with (ROOT / 'artifacts/formal-monitor-error.log').open('a') as stream:
                stream.write(f'{time.strftime("%H:%M:%S")} {type(exc).__name__}: {exc}\n')
        time.sleep(arguments.interval)


if __name__ == '__main__':
    main()
