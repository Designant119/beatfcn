"""Native-time diagnostics, with no smoothing, snapping or event removal."""
import html
import json
from pathlib import Path
import subprocess

from scripts.run_local_typed_canvas import ROOT, ART


def main():
    old=json.loads((ROOT/'artifacts/local-bundle-v1/bundle/draft-final.json').read_text())
    new=json.loads((ART/'draft-final.json').read_text());paths=[]
    for j in (0,4,8,12):
        row=new[j];start=row['request']['start_us'];end=row['request']['end_us']
        groups=[('Human target',row['target']),('Previous bundle',old[j]['prediction']['events']),
                ('Typed canvas',row['prediction']['events'])]
        colors=['#1674cc','#cd8312','#c3426d','#23846d'];kinds=['tap','hold','flick','drag']
        x=lambda us:110+(us-start)/(end-start)*1250
        lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1400" viewBox="0 0 1400 1400">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<style>text{font-family:Arial,sans-serif;fill:#263444;font-size:15px}</style>',
               f'<text x="30" y="32" style="font-size:22px">{html.escape(row["chart_id"])} / window start {start/1e6:g}s</text>',
               '<text x="30" y="57">Native times, types and multiplicity. Hold spans clipped only for display; spatial layout excluded.</text>']
        for pane,(title,events) in enumerate(groups):
            top=145+pane*370
            lines += [f'<text x="30" y="{top}" style="font-size:18px">{title}: {len(events)} notes</text>',
                      f'<defs><clipPath id="p{pane}"><rect x="110" y="{top+20}" width="1250" height="300"/></clipPath></defs>']
            for s in range(9):
                px=110+s/8*1250;lines.append(f'<path d="M {px} {top+30} V {top+310}" stroke="#dce2e9"/>')
            for k,kind in enumerate(kinds):
                y=top+65+k*65;lines.append(f'<text x="35" y="{y+5}">{kind}</text>')
                lines.append(f'<g clip-path="url(#p{pane})" fill="{colors[k]}" stroke="{colors[k]}">')
                for e in events:
                    if e['type']!=kind:continue
                    px=x(e['onset_us'])
                    if kind=='hold':lines.append(f'<path d="M {px:.3f} {y} H {x(e["end_us"]):.3f}" stroke-width="3"/>')
                    lines.append(f'<circle cx="{px:.3f}" cy="{y}" r="3"/>')
                lines.append('</g>')
        for s in range(9):lines.append(f'<text x="{110+s/8*1250}" y="1260" text-anchor="middle">{s}</text>')
        lines += ['<text x="735" y="1305" text-anchor="middle">Seconds relative to validation window</text>','</svg>']
        path=ART/'review'/f'window-{j:02d}.svg';path.parent.mkdir(exist_ok=True);path.write_text('\n'.join(lines))
        subprocess.run(['/usr/bin/qlmanage','-t','-s','1400','-o',str(path.parent),str(path)],
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=30)
        paths.append(str(path.with_suffix('.svg.png').relative_to(ROOT)))
    print(json.dumps(paths),flush=True)


if __name__=='__main__':main()
