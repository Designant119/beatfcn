import unittest

from scripts.build_round4_plan import WEIGHTS, round4_plan


class Round4PlanTest(unittest.TestCase):
    def test_fields(self):
        base = dict(id='r3', accum_steps=16, exposure='/data/r3/exposure.json', checkpoint_every=16000,
                    inputs_sha256={'/data/r3/exposure.json': 'old', '/data/tape.json': 't'},
                    alternative_context=dict(p=.7), learning_rate_schedule=dict(warmup_updates=300, total_updates=12000,
                    minimum_lr=5e-6), benchmark_v2=dict(every_optimizer_updates=4000, history_dir='x'), relaunch={})
        p = round4_plan(base, run_id='r4', exposure_key='/data/r4/exposure.json', exposure_sha='new', init_key='/i.pt',
                        init_sha='i', updates=3000, lr=3e-5, seed=7, weights=WEIGHTS, tags_key='/t.json', tags_sha='g')
        self.assertEqual(p['steps'], 48000)
        self.assertEqual(p['checkpoints'], [16000, 32000, 48000])
        self.assertNotIn('alternative_context', p)
        self.assertNotIn('/data/r3/exposure.json', p['inputs_sha256'])
        self.assertEqual(p['inputs_sha256']['/data/r4/exposure.json'], 'new')
        self.assertEqual(p['init_weights'], '/i.pt')
        self.assertEqual(p['learning_rate_schedule'], dict(warmup_updates=100, total_updates=3000, minimum_lr=5e-6))
        self.assertEqual(p['benchmark_v2']['every_optimizer_updates'], 1500)
        self.assertEqual(p['output'], 'reports/data-contract-v2-final/r4-training.json')
        self.assertEqual(base['id'], 'r3')  # source untouched


if __name__ == '__main__':
    unittest.main()
