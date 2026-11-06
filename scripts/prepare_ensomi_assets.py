"""Fetch a pinned, public external baseline; verify every model before inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'inputs/external_baselines/ensomi-r2-1.0.1'
REVISION = '0dbff5b22cd082d4d45cfe5b50c28b229b61259c'
BASE = f'https://huggingface.co/sed-i/ensomi-r2/resolve/{REVISION}/'
FILES = {
    'rows/model.pt': (9622005, '46b6ca37870120fda4d5de632949921fc26e020cf80abe44b4ba7f6f9557dda4'),
    'r2/model.pt': (9655631, '8dedefe4b11d238ad0b3186e6dfefe31f52cb333091aafe2e8efda8ad2ee3c6f'),
    'prior.json': (1958310, 'be5741d520120aaeb7ecf9126c2c12401015095b107b025584802f8e715590af'),
    'calibration.json': (443193, 'd39cd16fda451163d00d70cd6a913eb4bf64326ecd577feab08c25d8c22455ab'),
    'scopes.parquet': (2447837, '61f8648523cddb86e7316812b46896410f12b1a6ae179dcbf082c8b8615e3134'),
    'system.json': (209, '333d1c59a5ca199e07369b980992c2bb9683851246807426f1152260502e9718'),
    'LICENSE': (1068, '016f6c2a482f7c44720de5937f57f6d6286f651615d9074ca8778d7057475841'),
}


def fetch(name, url, size, digest, algorithm='sha256'):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_name(target.name + '.partial')
        started = time.monotonic()
        with urllib.request.urlopen(url, timeout=45) as response, partial.open('wb') as stream:
            while chunk := response.read(1 << 20):
                stream.write(chunk)
                if stream.tell() > size or time.monotonic() - started > 240:
                    raise RuntimeError(f'bounded download exceeded for {name}')
        partial.replace(target)
    raw = target.read_bytes()
    if len(raw) != size or hashlib.new(algorithm, raw).hexdigest() != digest:
        raise ValueError(f'asset integrity failed: {name}')
    result = dict(path=str(target.relative_to(ROOT)), url=url, bytes=len(raw),
                  sha256=hashlib.sha256(raw).hexdigest(), verified_against=algorithm,
                  official_expected_digest=digest)
    print(json.dumps(result), flush=True)
    return result


def main():
    started = time.time()
    results = [fetch(name, BASE + name, *identity) for name, identity in FILES.items()]
    results.append(fetch('beatthis/final0.ckpt',
        'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt',
        81058141, 'e1506282faf66ca10e8ab50ee26bd542b7b9ff0a', 'sha1'))
    # Keep the release's system.json untouched. The separate offline config changes
    # only checkpoint location, never learned parameters or decoding controls.
    config = json.loads((DEST / 'system.json').read_text())
    config['beatthis'] = 'beatthis/final0.ckpt'
    (DEST / 'system-offline.json').write_text(json.dumps(config, indent=2) + '\n')
    report = dict(id='D47-assets', complete=True, hf_revision=REVISION, files=results,
        downloaded_bytes=sum(x['bytes'] for x in results), started_at_unix=started,
        ended_at_unix=time.time(), new_cloud_spend_usd=0,
        code_revision='9e380a15f7c193476ff2063dfc10043e8484186a',
        disclaimer='External pretrained baseline. No BeatFCN training or quality claim.')
    (ROOT / 'reports/d47_external_assets.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
