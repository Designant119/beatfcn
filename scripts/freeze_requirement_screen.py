"""Freeze v2 case selection; no model allocation, GPU work or data mutations."""
import argparse
import hashlib
import json
from pathlib import Path


def freeze(source):
    samples=source['samples'];anchors=[]
    for mode in ('phira','mania_4k'):
        positive=[s for s in samples if s['mode']==mode and s['case_role']=='positive']
        if len(positive)<2:raise ValueError('Two independent positive songs per mode required')
        anchor=positive[0];foreign=next(s for s in positive[1:]if s['audio_sha256']!=anchor['audio_sha256'])
        anchors.append(dict(index=anchor['index'],foreign_index=foreign['index']))
    return dict(schema='requirement-screen-v2',tape_sha256=source['tape_sha256'],samples=samples,anchors=anchors,
        cell_us=100000,local_roll_s=2.,selection=source['selection'],
        semantics=dict(free='density unspecified: intensity_valid false',density_zero='explicit zero heads/s: valid true',
            low_high='same reference-density curve scaled0.5/2.0 as explicit requested condition',
            audio_zero='zero normalized features; not a real silent recording',
            local_roll='roll local MERT and mel by2s, keep global song context and clocks',
            audio_foreign='another held-out song; same event request/mode, acoustic streams replaced'),
        product_scope='fixed short-window requirement screen; whole-song and human play/edit remain required')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=freeze(json.loads(a.source.read_text()));value['source_panel_sha256']=hashlib.sha256(a.source.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(path=str(a.out),sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),cases=len(value['samples'])+14)))


if __name__=='__main__':main()
