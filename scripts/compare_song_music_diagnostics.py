"""Compare already-exported whole-song music diagnostics, without changing gates."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scripts.run_local_song_baseline import sha, write_json

METRICS=('anchor_20ms_precision','anchor_20ms_strong_recall','anchor_20ms_shift_excess',
         'anchor_50ms_precision','anchor_50ms_shift_excess','density_flux_spearman')


def run(control, candidate, output):
    if output.exists():raise FileExistsError(output)
    reports=[json.loads(p.read_text()) for p in (control,candidate)]
    indexed=[{r['chart_id']:r for r in d['records']} for d in reports]
    if any(len(d['records'])!=20 for d in reports) or len(indexed[0])!=20 or indexed[0].keys()!=indexed[1].keys():
        raise ValueError('Require identical twenty-song panels')
    rows=[]
    for chart,a in indexed[0].items():
        b=indexed[1][chart]
        if ((a['audio_sha256'],a['stratum'],a['notes']['human']) !=
                (b['audio_sha256'],b['stratum'],b['notes']['human'])
                or a['capability']['audio'] != b['capability']['audio']
                or a['tap_hold_music']['human'] != b['tap_hold_music']['human']):
            raise ValueError('Audio/human reference changed: '+chart)
        values={label:{m:r['tap_hold_music'][kind][0][m] for m in METRICS}
            for label,r,kind in [('control',a,'model'),('candidate',b,'model'),('human',b,'human')]}
        rows.append(dict(chart_id=chart,stratum=b['stratum'],values=values,
            support={label:r['tap_hold_music'][kind][1] for label,r,kind in
                [('control',a,'model'),('candidate',b,'model'),('human',b,'human')]},
            complete=dict(control=a['complete'],candidate=b['complete'])))
    summary={}
    for label in ('control','candidate','human'):
        summary[label]={}
        for metric in METRICS:
            vals=[r['values'][label][metric] for r in rows if r['values'][label][metric] is not None]
            if not all(math.isfinite(x)for x in vals):raise ValueError('Nonfinite music diagnostic')
            summary[label][metric]=dict(supported_songs=len(vals),missing_songs=20-len(vals),
                mean=statistics.mean(vals) if vals else None,median=statistics.median(vals) if vals else None)
    write_json(output,dict(complete=True,optimizer_updates=0,new_cloud_compute_calls=0,
        source_sha256={str(p):sha(p)for p in [control,candidate,Path(__file__)]},
        rows=rows,summary=summary,quality_passed=False,
        interpretation='Descriptive tap/hold heads only. Artistic drag/flick excluded from this onset proxy, not from data/quality. Nulls stay missing. Shift excess compares onset anchoring to within-song shifted timings; music peaks are estimates, not unique correct chart choices. No new thresholds or promotion rule.'))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.control,a.candidate,a.output)
