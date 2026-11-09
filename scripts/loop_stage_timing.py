"""Opt-in stage diagnostics around the frozen benchmark; never formal throughput.

Host scopes measure exclusive wall time (including waits), not CPU utilization.
CUDA events measure stream spans, which can include host submission gaps, not
kernel-active time. Host and device measurements must never be added together.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, ExitStack
from functools import wraps
import json
from pathlib import Path
import time
from unittest.mock import patch


class StageClock:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.host = Counter()
        self.calls = Counter()
        self.stack = []
        self.events = []
        self.gpu_depth = 0

    @contextmanager
    def scope(self, name, gpu=False):
        import torch
        record_gpu = gpu and self.gpu_depth == 0
        if record_gpu:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        if gpu:
            self.gpu_depth += 1
        frame = [self.clock(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame[0]
            self.stack.pop()
            self.host[name] += elapsed - frame[1]
            self.calls[name] += 1
            if self.stack:
                self.stack[-1][1] += elapsed
            if gpu:
                self.gpu_depth -= 1
            if record_gpu:
                end_event.record()
                self.events.append((name, start_event, end_event))

    def wrap(self, fn, label, gpu=False):
        @wraps(fn)
        def measured(*args, **kwargs):
            name = label(*args, **kwargs) if callable(label) else label
            with self.scope(name, gpu=gpu):
                return fn(*args, **kwargs)
        return measured

    def result(self, elapsed):
        import torch
        torch.cuda.synchronize()
        device, count = Counter(), Counter()
        for name, start, end in self.events:
            device[name] += start.elapsed_time(end) / 1000
            count[name] += 1
        residual = elapsed - sum(self.host.values())
        if residual < -1e-6:
            raise ValueError('exclusive host intervals exceed measurement window')
        host = dict(self.host, scheduler_and_unattributed=max(0.0, residual))
        return dict(host_exclusive_wall_s=host, host_calls=dict(self.calls),
                    cuda_stream_span_s=dict(device), cuda_event_pairs=dict(count),
                    diagnostic_wall_s=elapsed,
                    host_accounting_error_s=sum(host.values())-elapsed)


def operation_label(name):
    return {'advance': 'recurrent', 'readout': 'combined_readout',
            'readout_top': 'combined_readout', 'embed': 'embedding',
            'prefill': 'prefill', 'synchronize': 'synchronization'}[name]


@contextmanager
def instrument(timer):
    import llm_memory_editability.loop_batch_decode as batch
    import llm_memory_editability.loop_optimized_decode as optimized
    import llm_memory_editability.loop_budget_decode as budget
    from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
    from llm_memory_editability.loop_tree_decode import DraftTree, LoopKV

    with ExitStack() as patches:
        def wrap(owner, name, label, gpu=False):
            patches.enter_context(patch.object(
                owner, name, timer.wrap(getattr(owner, name), label, gpu)))

        # Executor host scope includes metadata, readback and blocking waits.
        wrap(batch.FusedLoopExecutor, 'execute',
             lambda self, name, jobs: operation_label(name) + '_host')
        # CUDA event scope is the actual packed forward, excluding metadata setup.
        wrap(optimized.DecodeGraphs, 'run',
             lambda self, key, inputs, fn: operation_label(key[0]) + '_forward', True)
        wrap(optimized, 'prefill_chunk', 'prefill', True)
        for method in ['start_request', 'add_active_response', 'stop_request', 'abort_request']:
            wrap(CompletionSuffixCache, method, 'suffix_update')
        wrap(CompletionSuffixCache, 'speculate', 'suffix_query')
        wrap(budget, 'select_candidates', 'tree_selection')
        for method in ['add', 'release']:
            wrap(budget.RecyclingTree, method, 'tree_allocate_prune')
        for method in ['add', 'kill']:
            wrap(DraftTree, method, 'tree_allocate_prune')
        wrap(LoopKV, 'upload', 'metadata_upload_and_wait')
        wrap(LoopKV, 'commit', 'kv_commit', True)
        yield


def main():
    import loop_batch_bench as bench
    args = bench.parser().parse_args()
    out = Path(args.out)
    original = bench.batch_decode
    measured_calls = 0

    def measured(*a, **kw):
        nonlocal measured_calls
        engine = kw.get('executor')
        if engine is None or engine.graphs.warming:
            return original(*a, **kw)
        timer = StageClock()
        with instrument(timer):
            a[1][0].synchronize()
            begin = time.perf_counter()
            records, summary = original(*a, **kw)
            a[1][0].synchronize()
            elapsed = time.perf_counter() - begin
        report = timer.result(elapsed)
        report.update(instrumented=True, formal_throughput=False,
                      output_tokens=sum(r['n_tokens'] for r in records),
                      decode_tokens=sum(r['stats']['tokens'] for r in records),
                      benchmark_wall_s=summary['wall_s'],
                      definitions=__doc__,
                      merged_stages='Shallow and full-depth readouts share fused calls; '
                      'combined_readout includes coda, LM head, and full-depth verification. '
                      'Recurrent includes attention KV reads/writes. No per-row time allocation.')
        prefix = f'{a[2].name}-b{len(a[1])}-seed0'
        (out / f'{prefix}-stages.json').write_text(json.dumps(report, indent=2))
        summary.update(diagnostic=True, stage_profiled=True, formal_throughput=False)
        measured_calls += 1
        return records, summary

    bench.batch_decode = measured
    bench.main()
    if measured_calls != 1:
        raise RuntimeError(f'expected exactly one measured diagnostic stream: {measured_calls}')
    metadata_path = out / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(stage_profiled=True, formal_throughput=False)
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
