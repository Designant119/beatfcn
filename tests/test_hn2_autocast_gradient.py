import ast
import copy
import json
from pathlib import Path
import unittest

import torch

from beatfcn.model.hn2_native import HN2Native, HN2Config
from beatfcn.train.hn2_native import training_step
from beatfcn.train.hn2_native_fixtures import native_fixtures


ROOT = Path(__file__).resolve().parents[1]
HEADS = {'head_query.weight', 'type_read.0.weight', 'type_read.2.weight',
         'tail_query.weight', 'tail_key.weight'}


class AutocastTests(unittest.TestCase):
    def test_same_real_model_cached_vs_uncached_gradients_and_unchanged_forward_loss(self):
        torch.set_num_threads(4);torch.manual_seed(17)
        config = HN2Config(d=64,heads=4,ffn=128,fine=32,local_layers=1,
                           global_layers=1,relation_layers=1,decoder_layers=1)
        original = HN2Native(config);weights = copy.deepcopy(original.state_dict())
        batch = native_fixtures()[0]
        prior = json.loads((ROOT/'artifacts/c17-masked-native-events-v1/priors.json').read_text())['duration']
        losses = {}
        for name, enabled, cache in [('fp32',False,True),('bad_cache',True,True),('fixed',True,False)]:
            model=HN2Native(config);model.load_state_dict(weights)
            with torch.autocast('cpu',dtype=torch.bfloat16,enabled=enabled,cache_enabled=cache):
                loss,*_=training_step(model,batch,prior,generator=torch.Generator().manual_seed(123))
            loss.backward();missing={n for n,p in model.named_parameters() if p.grad is None}
            self.assertEqual(missing, HEADS if name=='bad_cache' else set(), name)
            losses[name]=float(loss.detach())
        self.assertAlmostEqual(losses['bad_cache'],losses['fixed'],places=5)

    def test_deployed_runtime_disables_weight_cache_in_every_bf16_region(self):
        tree=ast.parse((ROOT/'beatfcn/train/hn2_uncached_amp_runtime.py').read_text())
        contexts=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                  and n.func.attr=='autocast']
        self.assertEqual(len(contexts),8)
        for n in contexts:
            option=next((k.value for k in n.keywords if k.arg=='cache_enabled'),None)
            self.assertIsInstance(option,ast.Constant)
            self.assertIs(option.value,False)


if __name__=='__main__':
    unittest.main()
