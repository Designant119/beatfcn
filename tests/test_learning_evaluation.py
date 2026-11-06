import unittest

from beatfcn.train.learning_evaluation import metrics, summarize


class LearningEvaluationTests(unittest.TestCase):
    def test_type_counts_cannot_replace_temporal_matching(self):
        truth=[dict(onset_us=0,end_us=0,type='tap',lane=0)]
        prediction=dict(complete=True,events=[dict(onset_us=500000,end_us=500000,type='tap',lane=0)])
        result=metrics(prediction,truth,'mania_4k')
        self.assertEqual(result['per_type']['tap']['tp'],0)
        self.assertEqual(result['per_type']['tap']['f1'],0.)

    def test_incomplete_generation_is_not_a_complete_case(self):
        truth=[dict(onset_us=0,end_us=100000,type='hold',lane=0)]
        prediction=dict(complete=False,events=truth)
        row=dict(case_role='positive',teacher_joint_ce=1.,prediction=prediction,
                 metrics=metrics(prediction,truth,'mania_4k'))
        result=summarize([row],1)
        self.assertEqual(result['complete_cases'],0)
        self.assertEqual(result['hold_joint_f1_50ms'],1.)


class PhysicalBatchObjectiveTests(unittest.TestCase):
    def test_full_model_eight_matches_two_four_request_gradients(self):
        import torch
        from tests.test_full_event_model import EventModelTests
        from beatfcn.train.training_objective import training_loss, exposure_group_loss
        fixture=EventModelTests();fixture.setUp()
        model=fixture.model
        inputs={k:v.repeat((8,)+(1,)*(v.ndim-1)) for k,v in fixture.inputs.items()}
        sequence={k:v.repeat((8,)+(1,)*(v.ndim-1)) for k,v in fixture.seq.items()}
        inputs['local']=inputs['local']+torch.arange(8).view(8,1,1,1,1)*.01
        settings=dict(type_weight=.1,z_weight=.0001,first_token_weight=5.)
        actual=exposure_group_loss(model(inputs,sequence),sequence,settings)*8/64
        actual_grad=torch.autograd.grad(actual,tuple(model.parameters()))
        expected=sum(training_loss(model({k:v[i:i+4]for k,v in inputs.items()},
            {k:v[i:i+4]for k,v in sequence.items()}),
            {k:v[i:i+4]for k,v in sequence.items()},settings)*4/64 for i in (0,4))
        expected_grad=torch.autograd.grad(expected,tuple(model.parameters()))
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        for a,b in zip(actual_grad,expected_grad):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)

    def test_grouped_auxiliary_loss_preserves_sparse_type_denominators(self):
        import torch
        from beatfcn.model.full_event_codec import FIELDS
        from beatfcn.train.training_objective import training_loss, exposure_group_loss
        torch.manual_seed(8)
        logits=torch.randn(8,3,5,requires_grad=True)
        fields=torch.full((8,3),FIELDS.index('head_ms'))
        fields[0,:]=FIELDS.index('kind')
        fields[4:,:]=FIELDS.index('kind')
        sequence=dict(valid=torch.ones(8,3,dtype=torch.bool),fields=fields,
            targets=torch.zeros(8,3,dtype=torch.int64),allowed=torch.ones(8,3,5,dtype=torch.bool))
        settings=dict(type_weight=2.,z_weight=.0001,first_token_weight=5.)
        expected=sum(training_loss(logits[i:i+4],{k:v[i:i+4]for k,v in sequence.items()},settings)
                     for i in (0,4))/2
        actual=exposure_group_loss(logits,sequence,settings)
        torch.testing.assert_close(actual,expected)
        self.assertGreater(abs(float((training_loss(logits,sequence,settings)-expected).detach())),1e-5)
        a=torch.autograd.grad(actual,logits,retain_graph=True)[0]
        b=torch.autograd.grad(expected,logits)[0]
        torch.testing.assert_close(a,b)
