"""Bounded read-only source retrieval for the four frozen local generation cases."""
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import modal
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'bench/rhythm-v2/d65-local-curve-2m-v1.json').read_text())
    panel=json.loads((ROOT/plan['panel']).read_text())
    catalog=torch.load(ROOT/'inputs/r09_positive_pairs/catalog.pt',map_location='cpu',weights_only=False)
    lookup={e['row']['chart_id']:e['row'] for e in catalog['validation_candidates']}
    output=ROOT/'inputs/d65_local/badcase_audio';output.mkdir(parents=True,exist_ok=True)
    receipt=ROOT/'reports/d65_badcase_audio.json'
    if receipt.exists():raise FileExistsError(receipt)
    report=dict(complete=False,records=[],errors=[],downloaded_bytes=0,new_cloud_compute_calls=0)
    volume=modal.Volume.from_name('beatfcn-data',create_if_missing=False)
    started=time.monotonic()

    def read(path):
        buffer=io.BytesIO()
        for chunk in volume.read_file(path):
            report['downloaded_bytes']+=len(chunk)
            if (time.monotonic()-started>240 or buffer.tell()+len(chunk)>64*2**20
                    or report['downloaded_bytes']>192*2**20):
                raise ValueError('Source transfer bound reached')
            buffer.write(chunk)
        return buffer.getvalue()

    for index in plan['generation_indices']:
        row=panel['records'][index];source=lookup[row['chart_id']];sha=row['audio_sha256']
        try:
            if source['mode']=='phira':
                remote=f"raw/phira/zips/{source['native_id']}.zip";archive=read(remote)
                with zipfile.ZipFile(io.BytesIO(archive)) as z:
                    names={}
                    for name in z.namelist():
                        key=name.lower().rsplit('/',1)[-1]
                        if key in names:raise ValueError('Ambiguous archive member basename')
                        names[key]=name
                    info=yaml.safe_load(z.read(names.get('info.yml') or names['info.yaml']).decode('utf-8-sig'))
                    member=names[str(info['music']).lower().rsplit('/',1)[-1]]
                    if z.getinfo(member).file_size>32*2**20:raise ValueError('Source audio too large')
                    audio=z.read(member)
                archive_sha=hashlib.sha256(archive).hexdigest()
            else:
                remote=f"raw/osu/audio/{sha}.{source['audio_ext']}";audio=read(remote)
                member=None;archive_sha=None
            if hashlib.sha256(audio).hexdigest()!=sha:raise ValueError('Raw audio identity differs from panel')
            path=output/f"{sha}.{source['audio_ext']}"
            with path.open('xb') as stream:stream.write(audio)
            report['records'].append(dict(index=index,chart_id=row['chart_id'],audio_sha256=sha,
                path=str(path.relative_to(ROOT)),remote=remote,archive_sha256=archive_sha,music_member=member,
                start_us=row['request']['start_us'],end_us=row['request']['end_us'],
                offset_decision=row['offset_decision']))
        except Exception as exc:
            report['errors'].append(dict(index=index,error=f'{type(exc).__name__}: {exc}'))
        report['wall_s']=time.monotonic()-started
        receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        if len(report['errors'])>=2:break
    report['complete']=len(report['records'])==len(plan['generation_indices'])
    receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(complete=report['complete'],records=len(report['records']),errors=report['errors'])))


if __name__=='__main__':main()
