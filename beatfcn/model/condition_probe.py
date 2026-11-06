"""Predeclared control requests without looking up the hidden target's density."""
from copy import deepcopy
from dataclasses import replace
import math

from ..events import controls as C
from ..events.canonical import MODE_TYPES

# These are exploratory engineering requests, not human-calibrated difficulty bands.
# Freeze them before generating outputs; never search for a band that matches truth.
SWEEP_BANDS = {
    'density_notes_per_second': [[1., 1.5], [3., 4.], [6., 8.]],
    'peak_notes_per_second': [[2., 3.], [6., 8.], [12., 16.]],
    'hold_head_ratio': [[0., .02], [.2, .3], [.6, .8]],
    'hold_occupancy_ratio': [[0., .02], [.2, .3], [.6, .8]],
    'chord_ratio': [[0., .02], [.2, .3], [.6, .8]],
    'jack_ratio': [[0., .02], [.2, .3], [.6, .8]],
    'flick_ratio': [[0., .02], [.2, .3], [.6, .8]],
    'drag_ratio': [[0., .02], [.2, .3], [.6, .8]],
}


def control_views(sample, field, bands=None):
    """Same task/truth/reference; only one numeric control is present in each view.

    Remove other numeric controls to avoid contradictory requests and inadvertent
    target-derived conditioning. Keep the original allowed-type whitelist.
    """
    mode = sample.request['mode']
    if field not in C.MODE_FIELDS[mode]:
        raise ValueError('control does not apply to this mode')
    bands = SWEEP_BANDS[field] if bands is None else bands
    if not bands or any(not C.is_table_band(field, b) for b in bands):
        raise ValueError('control sweep needs supported table bands')
    result = []
    for band in bands:
        request = deepcopy(sample.request)
        allowed = list(request.get('controls', {}).get('allowed_types', MODE_TYPES[mode]))
        request['controls'] = {'allowed_types': allowed, field: list(band)}
        result.append(replace(sample, request=request,
                              recipe={**sample.recipe, 'control_probe': {'field': field, 'band': list(band),
                                                                       'source': 'fixed_non_oracle_v1'}}))
    return tuple(result)


def condition_inventory():
    return {'numeric_fields': {name: {'modes': [mode for mode, fs in C.MODE_FIELDS.items() if name in fs],
                                     'table_edges': list(edges), 'probe_bands': SWEEP_BANDS[name],
                                     'enforcement': 'learned_soft_condition'} for name, edges in C.BAND_EDGES.items()},
            'hard_constraints': ['mode', 'allowed_types', 'editable_and_locked_regions', 'observed_event_preservation'],
            'learned_evidence': ['local_audio', 'whole_song_audio_and_known_chart_summary',
                                 'observed_events_and_anchors', 'generated_prefix',
                                 'reference_selected_and_explicit_unselected_evidence'],
            'experimental_input': ['authorized_phira_operation_geometry'],
            'not_implemented_as_numeric_inputs': ['difficulty_label', 'intensity_curve', 'reference_strength',
                                                  'max_active_contacts'],
            'counterfactual_implementation': 'reference support selection swaps; complete-window density supervision and incompatible-density ranking are implemented; generated control benefit still requires benchmark validation',
            'capability_warning': 'Input availability is not evidence of effective control. Hard constraints do not prove learned understanding.'}


def screen_control(field, mode, records, expected_songs, *, minimum_songs=4):
    """Conservative small-panel rejection gate, applicable to every numeric control.

    Rows carry song, requested, achieved, complete, legality and feasible. An
    explicit feasible=True is required: a locked chart or whitelist can make a
    requested band impossible. Missing/undefined/illegal rows retain coverage.
    Zero strictly increasing sweeps on >=4 songs stops expansion for futility.
    This is an engineering rejection rule, not a population significance test;
    a positive response only permits quality review, never automatic promotion.
    """
    if field not in C.MODE_FIELDS[mode]:
        raise ValueError('control does not apply to this mode')
    songs = list(expected_songs)
    if len(songs) != len(set(songs)) or minimum_songs < 1:
        raise ValueError('unique expected songs and positive minimum required')
    bands = SWEEP_BANDS[field]
    indexed = {}
    for row in records:
        key = (row['song'], tuple(row['requested']))
        if row['song'] not in songs or list(row['requested']) not in bands or key in indexed:
            raise ValueError('unexpected or duplicate control record')
        indexed[key] = row
    values = []
    for song in songs:
        sweep = [indexed.get((song, tuple(b))) for b in bands]
        if not all(r and r.get('complete') and not r.get('legality') and r.get('feasible') is True
                   and r.get('achieved') is not None and math.isfinite(r['achieved']) for r in sweep):
            continue
        values.append([r['achieved'] for r in sweep])
    monotonic = sum(a < b < c for a, b, c in values)
    in_band = sum(C.band_of(field, x) == band for vs in values for x, band in zip(vs, bands))
    enough = len(values) >= minimum_songs
    return {'field': field, 'mode': mode, 'expected_songs': len(songs), 'recorded': len(records),
            'not_recorded': len(songs) * len(bands) - len(records),
            'complete_feasible_sweeps': len(values), 'strict_monotonic_songs': monotonic,
            'in_band': in_band, 'in_band_denominator': len(values) * len(bands),
            'decision': ('stop_futility' if monotonic == 0 else 'quality_review_required') if enough else 'insufficient_coverage',
            'promotion': False,
            'reason': 'zero_strict_response_on_small_panel' if enough and monotonic == 0 else 'no_automatic_pass'}
