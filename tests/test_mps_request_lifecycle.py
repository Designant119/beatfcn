"""A real backend regression: cleanup cannot erase a partial accumulation."""
import torch

from beatfcn.train.mps_request_lifecycle import clear_request_cache


def probe():
    if not torch.backends.mps.is_available():
        raise RuntimeError('Actual MPS is required')
    cpu = torch.nn.Parameter(torch.tensor([.2, -.3, .5]))
    mps = torch.nn.Parameter(cpu.detach().to('mps'))
    opts = [torch.optim.AdamW([p], lr=.003, weight_decay=.01) for p in (cpu,mps)]
    gradient_diffs = []
    parameter_diffs = []
    retained = []
    for step in range(2):
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        for i in range(4):
            vector = torch.tensor([i+.5, 1.-i, step+.25])
            target = torch.tensor(.3*i-.1*step)
            for p in (cpu,mps):
                loss = ((p*vector.to(p.device)).sum()-target.to(p.device)).square()/4
                loss.backward()
                del loss
            before = mps.grad.detach().cpu().clone()
            clear_request_cache()
            retained.append(torch.equal(before,mps.grad.cpu()))
            gradient_diffs.append(float((cpu.grad-mps.grad.cpu()).abs().max()))
        for opt in opts:
            opt.step()
        parameter_diffs.append(float((cpu.detach()-mps.detach().cpu()).abs().max()))
    assert all(retained), 'Cleanup erased or changed partial accumulated gradients'
    assert max(gradient_diffs) <= 1e-5
    assert max(parameter_diffs) <= 1e-5
    return dict(status='completed',runtime=torch.__version__,retained_gradients=retained,
                gradient_max_diff=max(gradient_diffs),parameter_max_diff=max(parameter_diffs),
                toy_optimizer_updates_cpu=2,toy_optimizer_updates_mps=2,
                scope='New lifecycle helper only;not new BeatFCN health or quality')


def test_actual_accumulation_survives_cleanup():
    probe()
