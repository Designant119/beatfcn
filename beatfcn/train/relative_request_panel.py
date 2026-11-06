"""Opt-in D88 supervision; audio, requests and references use the original loader."""
from beatfcn.model.relative_event_codec import encode_events, teacher_sequence
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request


class RelativeRequestPanel(LocalRequestPanel):
    def _sequence(self, index):
        row = self.records[index]
        request = event_request(row['request'])
        tokens = encode_events(self.targets(index), request)
        if len(tokens) != row['tokens']:
            raise ValueError('Relative supervision changed the frozen token count')
        return teacher_sequence(tokens, request)
