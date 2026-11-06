"""Render paired bad cases without snapping, smoothing or removing events."""
import html
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'reports/local_portfolio_data_scale_review_v1.json'
DEST = ROOT / 'artifacts/local-portfolio-v1/data-scale/review'
COLORS = {'tap': '#1674cc', 'hold': '#cd8312', 'flick': '#c3426d', 'drag': '#23846d'}


def render(item, small24, small64, name):
    request = item['request']
    start, end = request['start_us'], request['end_us']
    x = lambda us: 110 + (us - start) / (end - start) * 1250
    groups = [('Human target', item['target']), ('small24', small24['prediction']['events']),
        ('small64', small64['prediction']['events'])]
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1050" viewBox="0 0 1400 1050">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263444;font-size:15px}</style>',
        f'<text x="30" y="32" style="font-size:19px">{html.escape(item["component"])} / window {start / 1e6:.3f}s</text>',
        '<text x="30" y="57">Native ms; all duplicate events retained; hold spans clipped only at the display boundary.</text>']
    for pane, (title, events) in enumerate(groups):
        top = 100 + pane * 300
        lines += [f'<text x="30" y="{top}" style="font-size:18px">{title}: {len(events)} notes</text>',
            f'<defs><clipPath id="p{pane}"><rect x="110" y="{top + 10}" width="1250" height="260"/></clipPath></defs>']
        for tick in range(9):
            px = 110 + tick / 8 * 1250
            lines.append(f'<path d="M {px} {top + 20} V {top + 270}" stroke="#dce2e9"/>')
            lines.append(f'<text x="{px}" y="{top + 290}" text-anchor="middle">{(end - start) * tick / 8 / 1e6:.1f}</text>')
        for kind, color in COLORS.items():
            y = top + 55 + tuple(COLORS).index(kind) * 55
            lines.append(f'<text x="35" y="{y + 5}">{kind}</text>')
            lines.append(f'<g clip-path="url(#p{pane})" fill="{color}" stroke="{color}">')
            for event in events:
                if event['type'] != kind:
                    continue
                px = x(event['onset_us'])
                if kind == 'hold':
                    lines.append(f'<path d="M {px:.3f} {y} H {x(event["end_us"]):.3f}" stroke-width="3"/>')
                lines.append(f'<circle cx="{px:.3f}" cy="{y}" r="3"/>')
            lines.append('</g>')
    lines.append('</svg>')
    path = DEST / f'{name}.svg'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))
    subprocess.run(['/usr/bin/qlmanage', '-t', '-s', '1400', '-o', str(DEST), str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=30)
    return str(path.with_suffix('.svg.png').relative_to(ROOT))


def main():
    review = json.loads(REVIEW.read_text())
    paths = []
    for arm, other in (('small24', 'small64'), ('small64', 'small24')):
        for rank, item in enumerate(review['bad_cases'][arm][:2]):
            paired = next(row for row in review['bad_cases'][other]
                if row['index'] == item['index']) if any(row['index'] == item['index'] for row in review['bad_cases'][other]) else None
            if paired is None:
                rows = json.loads((ROOT / f'artifacts/local-portfolio-v1/data-scale/{other}/quality-2000-draft.json').read_text())
                paired = next(row for row in rows if row['index'] == item['index'])
            first, second = (item, paired) if arm == 'small24' else (paired, item)
            paths.append(render(item, first, second, f'{arm}-bad-{rank}-index-{item["index"]}'))
    print(json.dumps(paths, indent=2))


if __name__ == '__main__':
    main()
