"""Request panel backed by MERT receipts instead of the legacy Pupu cache."""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.mert_features import MertFeatureStore


class MertRequestPanel(LocalRequestPanel):
    """Same request/target contract as LocalRequestPanel; feature windows come from MERT."""
    worker_cache_specs = (('_arrays', '_read_arrays', 4), ('sequence', '_sequence', 8))

    def __init__(self, root, tape, features_root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / tape).read_text())
        if not self.manifest['complete'] or self.manifest['failures']:
            raise ValueError('Request tape is incomplete or failed')
        self.features = MertFeatureStore(self.root, features_root)
        self.charts = self.manifest['charts']
        self.records = []
        for chart_index, chart in enumerate(self.charts):
            for request in chart['requests']:
                self.records.append(dict(request, chart_index=chart_index,
                    chart_id=chart['chart_id'], audio_sha256=chart['audio_sha256'],
                    component=chart['component'], split=chart['split'], mode=chart['mode'],
                    stratum=chart['stratum']))
        audios = sorted({record['audio_sha256'] for record in self.records})
        # Network-volume metadata reads dominate start-up on the full database
        # (~10k songs): check and load receipts concurrently, decide serially.
        from concurrent.futures import ThreadPoolExecutor

        def load(audio):
            if not (Path(features_root) / audio / 'receipt.json').is_file():
                return audio, False
            self.features.ensure(audio)
            return audio, True
        with ThreadPoolExecutor(max_workers=32) as pool:
            present = dict(pool.map(load, audios))
        missing = [audio for audio in audios if not present[audio]]
        # Opt-in for evaluating a SUBSET on a machine that holds only those songs' features (strict by default):
        # songs without a receipt are recorded and must never be requested.
        self.missing_audio = set(missing) if os.environ.get('BEATFCN_ALLOW_MISSING_MERT') == '1' else set()
        if missing and not self.missing_audio:
            raise ValueError(f'{len(missing)} panel songs lack a MERT receipt')
        # Trust the tape's admitted clock exactly; the receipt's integer division
        # can differ by one microsecond and the request validator is strict.
        for chart in self.charts:
            audio = chart['audio_sha256']
            if audio in self.missing_audio:
                continue
            if abs(self.features.records[audio]['duration_us']-int(chart['duration_us'])) > 1:
                raise ValueError('Target tape and decoded audio clocks disagree')
            self.features.records[audio]['duration_us'] = int(chart['duration_us'])
        self._arrays = lru_cache(maxsize=4)(self._read_arrays)
        self.sequence = lru_cache(maxsize=8)(self._sequence)
