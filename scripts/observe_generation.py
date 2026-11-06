"""Read-only stopping-decision observation; no sampling or model output changes."""
from contextlib import contextmanager
from unittest.mock import patch

import torch


def pointer_summary(logits, mask, row_start):
    scores = logits.detach().float().reshape(-1, logits.shape[-1])
    valid = mask.detach().reshape_as(scores)
    if scores.shape[0] != 1:
        raise ValueError('observer expects one generation query')
    probability = scores.masked_fill(~valid, -1e9).softmax(-1)[0].cpu()
    if not bool(torch.isfinite(probability).all()):
        return dict(row=int(row_start), finite=False)
    top, indices = probability.topk(min(4, probability.numel()))
    end = float(probability[0]); choice = int(probability.argmax())
    return dict(row=int(row_start), finite=True, end_probability=end,
                any_note_probability=float(probability[1:].sum()),
                best_note_probability=float(probability[1:].max()),
                greedy_choice=choice, greedy_end=choice == 0,
                legal_note_choices=int(valid[0, 1:].sum()),
                end_beats_individual_but_not_total_note_mass=bool(choice == 0 and end < .5),
                top_choices=[dict(index=int(i), probability=float(p)) for i, p in zip(indices, top)])


@contextmanager
def observe_generation():
    """Patch only for this single-threaded diagnostic and restore even on failure."""
    from beatfcn.model import sample
    original_generate = sample.generate_core
    summaries = []
    def generate(model, *args, **kwargs):
        original_pointer = model.time_logits
        trace = []
        def pointer(*p_args, **p_kwargs):
            result = original_pointer(*p_args, **p_kwargs)
            trace.append(pointer_summary(result[0], result[1], p_kwargs.get('row_start', 0)))
            return result
        with patch.object(model, 'time_logits', new=pointer):
            result = original_generate(model, *args, **kwargs)
        summary = dict(calls=len(trace), first=trace[0] if trace else None,
                       last=trace[-1] if trace else None,
                       end_steps=[r for r in trace if r.get('greedy_end')],
                       finite=all(r['finite'] for r in trace),
                       output_rule_changed=False)
        diagnostics = kwargs.get('diagnostics')
        if diagnostics is not None:
            diagnostics['time_pointer_observation'] = summary
        summaries.append(summary)
        return result
    with patch.object(sample, 'generate_core', new=generate):
        yield summaries
