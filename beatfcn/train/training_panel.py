"""One versioned data entry point for training, validation, and loader benchmarks."""
import hashlib
import json

from .event_density_drag_filter import RULE
from .local_request_panel import LocalRequestPanel
from .mert_panel import MertRequestPanel
from .intensity_panel import IntensityPanel
from .reference_intensity import ReferenceIntensityPanel
from .mel_normalization import MelNormalizedFeatures


def build_training_panel(root, plan):
    version = plan.get('data_contract_version')
    if version not in (1, 2) or (version == 1 and not plan.get('legacy_replay')):
        raise ValueError('Explicit data_contract_version=2 or isolated legacy_replay is required')
    if plan.get('mert_features'):
        panel = MertRequestPanel(root, plan['tape'], plan['mert_features'])
    else:
        panel = LocalRequestPanel(root, plan['tape'], plan['features'])
    if version == 2:
        if plan.get('config', {}).get('intensity_dim', 0) not in (0, 3):
            raise ValueError('v2 density requires intensity_dim=3: density/start/end')
        if (panel.manifest.get('clock_contract') != 'native-v1'
                or panel.manifest.get('drag_policy_id') != RULE['id']):
            raise ValueError('v2 training requires a rebuilt native-clock strict-drag-v2 tape')
        if plan.get('drag_filter'):
            raise ValueError('v2 tape is already filtered; legacy runtime drag_filter is forbidden')
        splits = {}
        for row in panel.records:
            for key in (row['audio_sha256'], row['component']):
                if key in splits and splits[key] != row['split']:
                    raise ValueError('Song/audio leaks across train and evaluation')
                splits[key] = row['split']
        norm = plan.get('mel_normalization')
        if norm is None:
            raise ValueError('Declare mel_normalization explicitly: none or train-only statistics')
        if norm != 'none':
            path = root/norm['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != norm['sha256']:
                raise ValueError('Mel normalization identity changed')
            stats = json.loads(path.read_text())
            train = {r['audio_sha256'] for r in panel.records if r['split'] == 'train'}
            other = {r['audio_sha256'] for r in panel.records if r['split'] != 'train'}
            panel.features = MelNormalizedFeatures(panel.features, stats,
                train_audios=train, forbidden_audios=other)
    if plan.get('reference_intensity'):
        panel = ReferenceIntensityPanel(panel, **dict(plan['reference_intensity'], version=version))
    elif plan.get('intensity'):
        panel = IntensityPanel(panel, **dict(plan['intensity'], version=version))
    if version == 1 and plan.get('drag_filter'):
        from .drag_density_filter import DragFilteredPanel
        panel = DragFilteredPanel(panel, **plan['drag_filter'])
    return panel
