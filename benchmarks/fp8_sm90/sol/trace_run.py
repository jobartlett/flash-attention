#!/usr/bin/env python
"""Run one configuration of the SM90 FP8 decode kernel for the speed-of-light study.

One process is one kernel build: the trace switch, the variant and the stage override are read
from the environment when the kernel is constructed (flash_attn/cute/sol_trace.py in the trace
worktree), so they are set here before the first call. Each run writes, under OUT/LABEL/:
  meta.json   shape, knobs, correctness, CUDA-graph and kernel timings, GPU info
  trace.npy   uint32 [iters, n_cta, 256 slots, (globaltimer_lo, clock)]  (with --trace)

  python benchmarks/fp8_sm90/sol/trace_run.py --root ../fa4-dtrace --case gemma-swa-p16 \
      --label base-trace --trace --out-dir sol-results/runs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fp8_compare as fc  # noqa: E402

TRACE_CTA_BYTES = 2048
TRACE_MAGIC = 0x7ACEFA4D
MAX_CTAS = 4096


def gpu_clocks():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,"
             "power.draw", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception as e:  # nvidia-smi missing in the container is not fatal
        return f"unavailable: {e}"


def build_case(args):
    c = fc.CASES[args.case]
    over = {k: getattr(args, k) for k in ("batch", "hq", "hk", "sk", "window", "page")
            if getattr(args, k) is not None}
    return replace(c, name=f"{c.name}+{args.label}", **over) if over else c


def capture_graph(fn):
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", required=True, help="trace worktree to import flash_attn.cute from")
    p.add_argument("--case", default="gemma-swa-p16", choices=sorted(fc.CASES))
    p.add_argument("--label", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--trace", action="store_true", help="build the timestamped kernel")
    p.add_argument("--variant", default="", help="FA4_SOL_VARIANT: noobs,nocal,hold,prewait,"
                   "stream,nomerge (comma list)")
    p.add_argument("--stages", type=int, default=0, help="FA4_SOL_STAGES (0: kernel default)")
    p.add_argument("--page-order", choices=["random", "identity"], default="random")
    p.add_argument("--aligned-seqlens", action="store_true",
                   help="round each request's KV length down to a multiple of 64")
    for k in ("batch", "hq", "hk", "sk", "window", "page"):
        p.add_argument(f"--{k}", type=int, default=None, help=f"override the case's {k}")
    p.add_argument("--num-splits", type=int, default=0)
    p.add_argument("--max-seqlen-k", type=int, default=0)
    p.add_argument("--flush-mb", type=int, default=128, help="L2 flush before each call; 0 = warm")
    p.add_argument("--iters", type=int, default=200, help="CUDA-graph timing iterations")
    p.add_argument("--prof-iters", type=int, default=50, help="torch-profiler kernel iterations")
    p.add_argument("--trace-iters", type=int, default=40, help="traced replays to keep")
    args = p.parse_args()

    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"
    os.environ["FA4_SOL_VARIANT"] = args.variant
    os.environ["FA4_SOL_STAGES"] = str(args.stages) if args.stages else ""
    os.environ.pop("FA4_TRACE_ADDR", None)

    import numpy as np
    import torch

    fwd, src = fc.import_fa4(Path(args.root))
    git_rev = subprocess.run(["git", "-C", args.root, "log", "-1", "--format=%h %s"],
                             capture_output=True, text=True).stdout.strip()
    trace_buf = None
    if args.trace:
        trace_buf = torch.zeros(MAX_CTAS * TRACE_CTA_BYTES, dtype=torch.uint8, device="cuda")
        os.environ["FA4_TRACE_ADDR"] = hex(trace_buf.data_ptr())

    c = build_case(args)
    max_k = max(c.sk, args.max_seqlen_k)
    inp = fc.make_inputs(c, max_k)
    pages = -(-c.sk // c.page)
    if args.page_order == "identity":
        inp["page_table"][:, :pages] = torch.arange(
            c.batch * pages, dtype=torch.int32, device="cuda").view(c.batch, pages)
    if args.aligned_seqlens:
        inp["seqused_k"] = (inp["seqused_k"] // 64 * 64).clamp(min=64)
    ref = fc.reference(c, inp)
    fn = fc.make_fn(fwd, c, "fp8", inp, args.num_splits, max_k)

    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(label=args.label, case=args.case, shape=asdict(c), trace=args.trace,
                variant=args.variant, stages=args.stages, page_order=args.page_order,
                aligned_seqlens=args.aligned_seqlens, flush_mb=args.flush_mb,
                num_splits=args.num_splits, max_seqlen_k=max_k, root=str(Path(args.root).resolve()),
                src=str(src), git=git_rev, gpu=torch.cuda.get_device_name(),
                seqused_k=inp["seqused_k"].tolist(), clocks_start=gpu_clocks())
    print(f"[sol] {args.label}: {git_rev} | {meta['gpu']} | trace={args.trace} "
          f"variant='{args.variant}' stages={args.stages or 'default'}", flush=True)

    out = fn()
    torch.cuda.synchronize()
    err = out.float() - ref
    rel_l1 = (err.abs().mean() / ref.abs().mean()).item()
    bias = (err.mean() / ref.mean()).item()
    tol_l1, tol_bias = fc.TOLERANCES["fp8"]
    meta.update(rel_l1=rel_l1, rel_bias=bias, max_abs=err.abs().max().item(),
                passed=bool(rel_l1 <= tol_l1 and abs(bias) <= tol_bias))
    print(f"[sol] {args.label}: rel_l1 {rel_l1:.2e} bias {bias:+.1e} "
          f"{'PASS' if meta['passed'] else 'FAIL'}", flush=True)

    flush = fc.L2Flush(args.flush_mb)
    meta["graph_us"] = fc.time_graph(fn, flush, args.iters)
    meta["kernel_us"], meta["kernels"] = fc.time_kernels(fn, flush, args.prof_iters)
    print(f"[sol] {args.label}: graph {meta['graph_us']:.2f} us, kernels {meta['kernel_us']:.2f} us",
          flush=True)
    for k in meta["kernels"]:
        print(f"        {k['us_per_call']:7.2f} us  x{k['per_call']:.0f}  {k['name'][:120]}")

    if args.trace:
        graph = capture_graph(fn)
        words = trace_buf.view(torch.int32).view(MAX_CTAS, TRACE_CTA_BYTES // 8, 2)
        n_cta = None
        frames, replay_us = [], []
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for it in range(args.trace_iters + 3):
            trace_buf.zero_()
            flush()
            e0.record()
            graph.replay()
            e1.record()
            torch.cuda.synchronize()
            if it < 3:
                continue
            if n_cta is None:
                magic = words[:, 0, 0].cpu().numpy().astype(np.uint32)
                valid = np.nonzero(magic == TRACE_MAGIC)[0]
                assert len(valid), "no CTA wrote a trace header: is the trace worktree in use?"
                n_cta = int(valid[-1]) + 1
                print(f"[sol] {args.label}: {len(valid)} traced CTAs (of {n_cta} slots)", flush=True)
            frames.append(words[:n_cta].cpu().numpy().view(np.uint32))
            replay_us.append(e0.elapsed_time(e1) * 1e3)
        np.save(out_dir / "trace.npy", np.stack(frames))
        meta.update(n_cta=n_cta, trace_iters=len(frames), trace_replay_us=replay_us)
    meta["clocks_end"] = gpu_clocks()
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[sol] {args.label}: wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
