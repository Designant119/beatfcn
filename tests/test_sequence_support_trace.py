import unittest
import torch

from beatfcn.model.full_event_codec import EventRequest, EventState
from scripts.audit_sequence_support_trace import static_head_support, rank


class SupportTraceTest(unittest.TestCase):
    def test_shadow_preserves_actual_state_and_coarse_block(self):
        state=EventState(EventRequest('phira',0,8000000,10000000))
        state.last_onset=2025000;state.phase='head_ms';state.head_block=1;state.head_cell=0
        actual=state.allowed_tokens();shadow=static_head_support(state)
        self.assertEqual(state.last_onset,2025000)
        self.assertEqual(state.allowed_tokens(),actual)
        self.assertEqual([state.vocab.decision(t)[1] for t in actual],list(range(26,40)))
        self.assertEqual([state.vocab.decision(t)[1] for t in shadow],list(range(40)))

    def test_rank_ties_use_original_support_order(self):
        result=rank(torch.tensor([7.,5.,7.]),(2,0,1))
        self.assertEqual(result['top_token'],2)
        self.assertEqual(result['tied_top_count'],2)
        self.assertEqual(result['margin'],0)


if __name__=='__main__':unittest.main()
