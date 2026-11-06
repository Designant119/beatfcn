import unittest

from scripts.monitor_600m import State
from scripts.monitor_600m_dashboard import snapshot, trend


class DashboardTests(unittest.TestCase):
    def test_directions_are_inverted_for_losses_and_scores(self):
        self.assertEqual(trend([2.,1.],-1)['status'],'改善')
        self.assertEqual(trend([.1,.3],1)['status'],'改善')
        self.assertEqual(trend([1.,2.],-1)['status'],'变差')
        self.assertEqual(trend([10.,1.],-1,False)['status'],'不可判')

    def test_owner70percent_threshold_and_safety_range(self):
        state=State();state.apply(dict(phase='train',step=16,updates=1,elapsed=10,
            gpu_util_rolling=83.,gpu_mem_mib=1024,tok_ce=2.))
        view=snapshot(state,dict(operator_override=dict(minimum_util=70)))
        gpu=view['resources'][0]
        self.assertEqual(gpu['target'],70)
        self.assertEqual(gpu['status'],'达标')
        self.assertEqual(view['train']['status'],'不可判')

    def test_fewer_censored_events_cannot_be_marked_improvement(self):
        def checkpoint(opt,count):
            return dict(optimizer_updates=opt,sha256=str(opt),
                records=[dict(case_role='positive',teacher_joint_ce=2./opt,
                              prediction=dict(complete=True,events=[])),
                         dict(case_role='natural_silence',teacher_joint_ce=2./opt,
                              prediction=dict(complete=False,events=[{}]*count))],
                summary=dict(expected=2,hold_joint_f1_50ms=0.,silence_false_events=count,per_type={}))
        state=State();state.evaluation=dict(protocol_sha256='fixed',checkpoints=[checkpoint(1,20),checkpoint(2,2)])
        view=snapshot(state,dict(quality_eval_protocol=dict(panel_sha256='fixed')))
        scores={q['key']:q for q in view['quality']}
        self.assertEqual(scores['teacher_positive']['status'],'改善')
        self.assertEqual(scores['silence_false']['status'],'不可判')
        self.assertEqual(scores['useful_complete']['value'],0.)

    def test_other_protocol_results_are_not_compared(self):
        state=State();state.evaluation=dict(protocol_sha256='foreign',checkpoints=[dict(optimizer_updates=20)])
        view=snapshot(state,dict(quality_eval_protocol=dict(panel_sha256='fixed')))
        self.assertEqual(view['quality'],[])
