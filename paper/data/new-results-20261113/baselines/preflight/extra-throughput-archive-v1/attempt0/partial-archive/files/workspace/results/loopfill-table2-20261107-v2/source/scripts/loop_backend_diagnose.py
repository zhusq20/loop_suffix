"""Short, instrumented backend diagnostics; forwards CLI options to loop_batch_bench.

Writes CPU profiles and measured/warmup CUDA graph shape frequencies. These runs
are diagnostic artifacts, not replacements for uninstrumented batch results.
"""
from __future__ import annotations

from collections import Counter
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

import torch

import loop_batch_bench as bench


class KernelCheckComplete(Exception):
    pass


@torch.no_grad()
def check_kernels(engine, functions, out, label):
    """Compare identical captured inputs, outside any end-to-end measurement."""
    graphs = engine.graphs
    keys = sorted((k for k in graphs.entries if k[0] == 'advance'),
                  key=lambda k: graphs.observed[k], reverse=True)[:3]
    if not keys:
        raise RuntimeError('warmup captured no recurrent graph to check')
    rows = []
    for key in keys:
        static, graph, output = graphs.entries[key]
        fn = functions[key]
        expected = fn(static).clone()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        row = dict(key=key, warmup_calls=graphs.observed[key], exact_output_match=True)
        for name, call in [('eager', lambda: fn(static)), ('graph', graph.replay)]:
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin = time.perf_counter()
            start.record()
            for _ in range(20):
                call()
            end.record()
            end.synchronize()
            row[name] = dict(wall_ms_per_call=1000 * (time.perf_counter() - begin) / 20,
                             stream_ms_per_call=start.elapsed_time(end) / 20)
        rows.append(row)
    (out / f'{label}-kernels.json').write_text(json.dumps({
        'purpose': 'Identical recurrent operator inputs, 20 calls each; excludes metadata and scheduling',
        'formal_throughput': False, 'grad_enabled': torch.is_grad_enabled(), 'checks': rows,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostic-requests', type=int)
    parser.add_argument('--diagnostic-kernels-only', action='store_true')
    diagnostic, remaining = parser.parse_known_args()
    if diagnostic.diagnostic_requests is not None and diagnostic.diagnostic_requests < 1:
        parser.error('--diagnostic-requests must be positive')
    sys.argv[1:] = remaining
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Preserving existing diagnostic artifacts: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'DIAGNOSTIC.json').write_text(json.dumps({
        'instrumented': True,
        'purpose': 'CPU hotspots and CUDA graph shape coverage; not formal throughput',
        'argv': sys.argv[1:],
        'measured_request_limit': diagnostic.diagnostic_requests,
        'kernels_only': diagnostic.diagnostic_kernels_only,
    }, indent=2))
    original = bench.batch_decode
    functions = {}
    if diagnostic.diagnostic_kernels_only:
        from llm_memory_editability.loop_optimized_decode import DecodeGraphs
        freeze = DecodeGraphs.freeze

        def remember(self):
            functions.update({k: fn for k, (_, fn) in self.candidates.items()})
            return freeze(self)

        DecodeGraphs.freeze = remember

    def measured(*args, **kwargs):
        engine = kwargs.get('executor')
        if engine is None or engine.graphs.warming:
            return original(*args, **kwargs)
        if diagnostic.diagnostic_kernels_only:
            check_kernels(engine, functions, out, f'{args[2].name}-b{len(args[1])}')
            raise KernelCheckComplete()
        graphs = engine.graphs
        calls, hits = Counter(), Counter()
        run = graphs.run

        def tracked(key, inputs, fn):
            calls[key] += 1
            hits[key] += int(key in graphs.entries)
            return run(key, inputs, fn)

        graphs.run = tracked
        profiler = cProfile.Profile()
        config = args[2]
        label = f'{config.name}-b{len(args[1])}'
        if diagnostic.diagnostic_requests is not None:
            args = (args[0][:diagnostic.diagnostic_requests], *args[1:])
        try:
            profiler.enable()
            result = original(*args, **kwargs)
            result[1].update(diagnostic=True, cpu_profiled=True)
        finally:
            profiler.disable()
            graphs.run = run
            profiler.dump_stats(str(out / f'{label}-cpu.pstats'))
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats('tottime').print_stats(60)
            stats.sort_stats('cumtime').print_stats(60)
            (out / f'{label}-cpu.txt').write_text(stream.getvalue())
            (out / f'{label}-shapes.json').write_text(json.dumps({
                'key_fields': ['operation', 'real_rows', 'groups_bucket', 'query_bucket', 'tree_bucket'],
                'captured_count': len(graphs.entries),
                'warmup_unique': len(graphs.observed),
                'measured_unique': len(calls),
                'shapes': [dict(key=k, calls=n, hits=hits[k], warmup=graphs.observed[k],
                                captured=k in graphs.entries)
                           for k, n in calls.most_common()],
                'warmup_shapes': [dict(key=k, calls=n, captured=k in graphs.entries)
                                  for k, n in graphs.observed.most_common()],
            }, indent=2))
        return result

    bench.batch_decode = measured
    try:
        bench.main()
    except KernelCheckComplete:
        print('Kernel check complete; no measured request stream was run.', flush=True)


if __name__ == '__main__':
    main()
