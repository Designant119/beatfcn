import copy
import unittest
from beatfcn.train.audit import refresh_offsets, fingerprint
from beatfcn.train.loop import CoreSampler
from test_ingest import row, diag
from events_helpers import synthetic_mania, DURATION


class AuditTests(unittest.TestCase):
    def test_migration_preserves_split_and_constant_and_other_rejections(self):
        old = row('osu:a', 'osu', 'song', 'sha', 60)
        old.update(split='train', kept=False, approved=1,
                   p0_quarantine=['some_other_error'], quarantine_reasons=['some_other_error', 'offset_low_agree_after'],
                   offset_decision={'action':'quarantine', 'shift_ms':0, 'c_ms':14, 'r_ms':46})
        snapshot = copy.deepcopy(old)
        new, report = refresh_offsets([old], lambda _: diag(60))
        self.assertEqual(old, snapshot)
        self.assertEqual(new[0]['split'], 'train')
        self.assertEqual(new[0]['offset_decision']['c_ms'], 14)
        self.assertEqual(new[0]['offset_decision']['action'], 'shift')
        self.assertFalse(new[0]['kept'])
        self.assertEqual(new[0]['quarantine_reasons'], ['some_other_error'])
        self.assertNotEqual(report['before_sha256'], report['after_sha256'])
        self.assertEqual(fingerprint(new), report['after_sha256'])

    def test_missing_full_diagnostic_fails(self):
        r = row('osu:a','osu','a','s',60)
        r['offset_decision']={'c_ms':14}
        with self.assertRaisesRegex(ValueError, 'full diagnostic'):
            refresh_offsets([r], lambda _: {k:v for k,v in r['offset'].items() if k!='agree_counts'})

    def test_sampler_audit_records_rejected_and_accepted_requests(self):
        records=[]
        r={'chart_id':'a','song':'s','mode':'mania_4k','duration_us':DURATION}
        chart=synthetic_mania()
        sampler=CoreSampler([r],lambda _:chart,None,0,audit=records.append)
        ex=sampler.core(r,0)
        self.assertTrue(records[-1]['accepted'])
        self.assertEqual(ex.provenance['seed'],0)
        sampler.limits={'G':0}
        with self.assertRaises(ValueError):
            sampler.core(r,0)
        self.assertFalse(records[-1]['accepted'])
        self.assertEqual(records[-1]['stage'],'bounds')
        self.assertEqual(records[0]['c0_us'],records[1]['c0_us'])
