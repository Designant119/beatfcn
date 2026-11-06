"""Cold compiled-loss preparation with RNG rollback and real kernel traces."""
from contextlib import contextmanager
import json
import signal
from pathlib import Path
import statistics
import time

import torch


@contextmanager
def preparation_deadline(seconds, label='Compiled training cold preparation'):
    previous = signal.getsignal(signal.SIGALRM)
    def expired(*_):
        raise TimeoutError(label+' exceeded its bound')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def profile_pass(function, inputs, sequence, output):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        loss = function(inputs, sequence)
        loss.backward()
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-tick
    profiler.export_chrome_trace(str(output))
    cpu = sorted(profiler.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    kernels = [e for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    intervals = sorted((e.time_range.start, e.time_range.end) for e in kernels)
    busy, end = 0., 0.
    for start, stop in intervals:
        busy += max(0., stop-max(start, end))
        end = max(end, stop)
    result = dict(wall_s=elapsed, cuda_kernel_count=len(kernels), kernel_busy_s=busy/1e6,
        top_cpu=[dict(name=e.key, self_us=e.self_cpu_time_total, count=e.count) for e in cpu[:12]],
        loss=float(loss.detach()), trace=str(output))
    del profiler, kernels, intervals, loss
    return result


def enable_regional_compile(model, *, backend='inductor', options=None):
    blocks = [block for group in model.encoders for block in group] + list(model.decoder)
    kwargs = dict(backend=backend, dynamic=True, fullgraph=True)
    if options:
        kwargs['options'] = options
    for block in model.decoder:
        block.cross_attn.project_kv = torch.compile(block.cross_attn.project_kv, **kwargs)
    for block in blocks:
        block.compile(**kwargs)
    return len(blocks) + len(model.decoder)


def probe_training_memory(function, model, optimizer, dataset, protocol, checkpoint, device):
    results = []
    total_memory = torch.cuda.get_device_properties(device).total_memory
    expected_steps = [float(state['step']) for state in optimizer.state.values() if 'step' in state]
    for case in protocol['cases']:
        item = dataset[(case['start_exposure']-1)//2]
        if item['step'] != case['end_exposure']:
            raise ValueError('Memory probe exposure identity mismatch')
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        try:
            requests = sum(len(ids) for ids, _, _ in item['batches'])
            for ids, cpu_inputs, cpu_sequence in item['batches']:
                inputs = {k: v.to(device) for k, v in cpu_inputs.items()}
                sequence = {k: v.to(device) for k, v in cpu_sequence.items()}
                loss = function(inputs, sequence)*len(ids)/requests
                loss.backward()
                del inputs, sequence, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            result = dict(**case, physical_sizes=[len(ids) for ids, _, _ in item['batches']],
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                total_device_bytes=total_memory, margin_fraction=1-max(allocated,reserved)/total_memory,
                discarded_probe_optimizer_updates=1)
            results.append(result)
            print(json.dumps(dict(phase='memory_probe', **result)), flush=True)
            if max(allocated, reserved) > .8*total_memory:
                raise MemoryError('Real worst-case probe left less than 20 percent device margin')
        finally:
            optimizer.zero_grad(set_to_none=True)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            del saved
        steps = [float(state['step']) for state in optimizer.state.values() if 'step' in state]
        result['restored_optimizer_step_min'] = min(steps)
        result['restored_optimizer_step_max'] = max(steps)
        if steps != expected_steps:
            raise RuntimeError('Discarded memory probe changed optimizer continuation state')
    return results


def prepare_compiled_loss(eager, model, optimizer, inputs, sequence, *, output, options=None, mode='regions', memory_probe=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = inputs['local'].device
    print(json.dumps(dict(phase='compile_preflight_start', device=str(device))), flush=True)
    if mode not in ('full', 'regions'):
        raise ValueError('Unknown compiled training mode')
    def eager_measure(inp, seq):
        with torch.compiler.set_stance('force_eager'):
            return eager(inp, seq)
    compiled_regions = 0
    timings = dict(eager=[], compiled=[])
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone()
    with preparation_deadline(360), torch.random.fork_rng(devices=[device.index or 0]):
        optimizer.zero_grad(set_to_none=True)
        eager(inputs, sequence).backward()
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        baseline = profile_pass(eager_measure, inputs, sequence, output/'eager.json')
        if mode == 'regions':
            compiled_regions = enable_regional_compile(model, options=options)
            compiled = eager
        else:
            compiled = torch.compile(eager, backend='inductor', dynamic=True, fullgraph=True, options=options or {})
        optimizer.zero_grad(set_to_none=True)
        compile_started = time.monotonic()
        compiled(inputs, sequence).backward()
        torch.cuda.synchronize()
        compile_s = time.monotonic()-compile_started
        print(json.dumps(dict(phase='compile_warmup_done', seconds=compile_s)), flush=True)
        optimizer.zero_grad(set_to_none=True)
        candidate = profile_pass(compiled, inputs, sequence, output/'compiled.json')
        for _ in range(5):
            for label, function in [('eager', eager_measure), ('compiled', compiled)]:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                loss = function(inputs, sequence)
                loss.backward()
                torch.cuda.synchronize()
                timings[label].append(time.perf_counter()-tick)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Compiled training preflight produced a nonfinite loss')
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Compiled training preflight produced missing or nonfinite gradients')
        finite_parameters = len(gradients)
        del gradients, loss
        optimizer.zero_grad(set_to_none=True)
        memory_results = memory_probe(compiled) if memory_probe is not None else []
    if (not torch.equal(cpu_rng, torch.get_rng_state())
            or not torch.equal(gpu_rng, torch.cuda.get_rng_state(device))):
        raise RuntimeError('Cold preparation changed continuation RNG state')
    medians = {name: statistics.median(samples) for name, samples in timings.items()}
    result = dict(mode=mode, compiled_regions=compiled_regions,
        scope='same real batch forward/backward; excludes optimizer and loader',
        optimizer_updates=0, rng_restored=True, compile_s=compile_s,
        cold_preparation_s=time.monotonic()-started, samples_s=timings, median_s=medians,
        baseline=baseline, compiled=candidate, finite_grad_parameters=finite_parameters, memory_probes=memory_results)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(phase='compile_preflight_done', median_s=medians,
        baseline_kernels=baseline['cuda_kernel_count'], compiled_kernels=candidate['cuda_kernel_count'])), flush=True)
    return compiled, result
