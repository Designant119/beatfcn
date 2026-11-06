"""Reference-only adapter for joint candidates, including known empty regions."""
from dataclasses import asdict

from beatfcn.model.local_song_rollout import canonical
from .event_reference import VisibleReference, SOURCE_COLUMN, encode_reference
from .reference_completion import sparse_reference


def joint_reference_inputs(request, audio_sha256, visible, regions, *, source_id, role='human_input'):
    """Consumes only caller-visible content/regions; no hidden-target argument.

    Regions mean a complete authorized view, even if visible has zero events.
    Without regions, supplied events are positive-only and imply no silence.
    """
    visible=tuple(visible);regions=tuple(regions)
    if role not in SOURCE_COLUMN or not source_id:raise ValueError('Explicit reference provenance required')
    sources=()
    if regions:
        sources=(VisibleReference(source_id=source_id,mode=request.mode,audio_sha256=audio_sha256,
            role=role,regions=regions,events=tuple(canonical(visible,request.mode))),)
    result,provenance=encode_reference(mode=request.mode,audio_sha256=audio_sha256,
        duration_us=request.duration_us,origin_us=request.start_us,
        unknown_regions=((request.start_us,request.end_us),),sources=sources)
    if not regions and visible:
        if any(not 0<=e.onset_us<=e.end_us<=request.duration_us or
            (e.type=='hold' and e.end_us<=e.onset_us) or
            (e.type!='hold' and e.end_us!=e.onset_us) or e.lane!=-1 for e in visible):
            raise ValueError('Invalid sparse visible event')
        canonical(visible,request.mode)  # Validate visible event types/native adapter.
        result=sparse_reference(visible,request)
        result['reference'][:,11:14]=0
        result['reference'][1:,SOURCE_COLUMN[role]]=1
        provenance=[dict(role='unknown',interval_us=[request.start_us,request.end_us])]+[
            dict(role=role,source_id=source_id,positive_only=True,event=asdict(e)) for e in visible]
    return result,provenance
