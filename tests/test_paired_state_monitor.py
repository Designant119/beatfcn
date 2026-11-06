from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts import train_monitor as monitor


class PairedStateMonitorTests(unittest.TestCase):
    def test_result_is_isolated_from_historical_model_and_stale_scores_are_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            result=directory/'fit.json';result.write_text('{"passed":false}')
            summary={arm:dict(loss=.06,mania_f1=1.,phira_f1=.98,phira_hold_matched=8,phira_hold_true=9,phira_drag_matched=32,phira_drag_true=32) for arm in ['A','B']}
            review=dict(complete=True,passed=False,summary=summary,optimizer_updates=200,scope='two fixed cores',bounded_process_s=13.8,peak_rss_gib=.414,new_cloud_spend_usd=0,source_sha256={'reports/fit.json':sha256(result.read_bytes()).hexdigest()})
            review['interpretation_note']='收益门槛不可达；原始结果保留。'
            (directory/'review.json').write_text(json.dumps(review))
            target={'paired_state_probe':dict(review='review.json',result='fit.json')}
            snapshot=monitor.Monitor().snapshot('old',now=20)
            before=dict(snapshot['metrics'])
            monitor.paired_state_probe_status(snapshot,target,directory)
            self.assertEqual(snapshot['paired_state_probe']['status'],'failed')
            self.assertFalse(snapshot['paired_state_probe']['model_promotion'])
            self.assertEqual(snapshot['metrics'],before)
            text=monitor.render_text(snapshot)
            self.assertIn('Phira释放错误仍在',text)
            self.assertIn('8/9',text)
            self.assertIn(review['interpretation_note'],text)
            result.write_text('{}')
            monitor.paired_state_probe_status(snapshot,target,directory)
            self.assertEqual(snapshot['paired_state_probe']['status'],'evidence_incomplete')
            self.assertNotIn('summary',snapshot['paired_state_probe'])
            self.assertNotIn('98.0%',monitor.render_text(snapshot))
            self.assertNotIn(review['interpretation_note'],monitor.render_text(snapshot))


if __name__=='__main__':unittest.main()
