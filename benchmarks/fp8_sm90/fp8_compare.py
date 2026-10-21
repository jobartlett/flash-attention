#!/usr/bin/env python
"""FA4 SM90 microbenchmark + correctness: FP16/BF16 vs original FP8 vs accelerated FP8.

Each kernel version lives in its own git worktree. A worker process imports
flash_attn.cute from exactly one worktree, so the two FP8 versions never share a process.

All variants compute the same math: inputs are generated in fp32, quantized to E4M3 with a
per-tensor scale, and the FP16/BF16 baselines get the dequantized values. One fp32 PyTorch
reference therefore scores every variant.

  # from the repo root on the original-FP8 branch, with the accelerated branch in a worktree:
  #   git worktree add ../fa4-accel jhaotingc/fa4-sm90-native-fp8-b
  # full comparison (spawns one worker per tree, prints tables)
  python benchmarks/fp8_sm90/fp8_compare.py compare --orig . --accel ../fa4-accel

  # one tree / case / dtype, e.g. under nsys or ncu
  python benchmarks/fp8_sm90/fp8_compare.py worker --root ../fa4-accel \
      --cases gemma-swa-p16 --dtypes fp8 --profile-iters 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    d: int
    hq: int
    hk: int
    batch: int
    sq: int  # query tokens per request
    sk: int  # max KV length; request i has sk - 37 * i keys
    window: int  # sliding window size in keys; 0 = full causal
    page: int  # KV cache page size


CASES = {
    c.name: c
    for c in [
        # Gemma-4-31B TP4 SWA verify step (MTP=3): the customer shape.
        Case("gemma-swa-p64", 256, 8, 4, 4, 4, 28216, 1024, 64),
        Case("gemma-swa-p16", 256, 8, 4, 4, 4, 28216, 1024, 16),
        # Same shape without the window.
        Case("d256-full-b4-q4-k28k", 256, 8, 4, 4, 4, 28216, 0, 16),
        # Generic decode shapes, so tuning is not Gemma-only.
        Case("d128-b64-q1-k2k", 128, 32, 8, 64, 1, 2048, 0, 16),
        Case("d128-b16-q4-k16k", 128, 32, 8, 16, 4, 16384, 0, 16),
        # Long KV, nonzero-mean V: exposes FP8 PV accumulation bias.
        Case("d128-b1-q1-k128k", 128, 32, 8, 1, 1, 131072, 0, 16),
        Case("d128-prefill-4k", 128, 32, 8, 1, 4096, 4096, 0, 64),
    ]
}
DEFAULT_CASES = ",".join(CASES)

# Pass thresholds on (mean |err| / mean |ref|, |mean err| / |mean ref|).
TOLERANCES = {"fp8": (0.02, 0.005), "fp16": (0.005, 0.002), "bf16": (0.01, 0.002)}


# ----------------------------------------------------------------------------- worker


def import_fa4(root: Path):
    """Import flash_attn.cute from `root`, bypassing any installed/editable copy and the
    FA2 flash_attn/__init__.py (which needs the FA2 C extension)."""
    root = root.resolve()
    sys.meta_path[:] = [
        f for f in sys.meta_path if "__editable__" not in getattr(f, "__module__", "")
    ]
    pkg = types.ModuleType("flash_attn")
    pkg.__path__ = [str(root / "flash_attn")]
    sys.modules["flash_attn"] = pkg
    from flash_attn.cute import interface

    src = Path(interface.__file__).resolve()
    assert src.is_relative_to(root), f"imported {src}, expected a file under {root}"
    return interface._flash_attn_fwd, src


def make_inputs(c: Case, seed: int = 0):
    import torch

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    pages = -(-c.sk // c.page)
    nblk = c.batch * pages
    q = torch.randn(c.batch * c.sq, c.hq, c.d, device=dev, generator=g) * 0.5
    k = torch.randn(nblk, c.page, c.hk, c.d, device=dev, generator=g) * 0.5
    v = torch.randn(nblk, c.page, c.hk, c.d, device=dev, generator=g) * 0.5 + 0.5
    page_table = torch.randperm(nblk, device=dev, generator=g, dtype=torch.int32)
    page_table = page_table.view(c.batch, pages)
    seqused_k = torch.tensor(
        [max(c.sq, c.sk - 37 * i) for i in range(c.batch)], dtype=torch.int32, device=dev
    )
    cu_seqlens_q = torch.arange(
        0, (c.batch + 1) * c.sq, c.sq, dtype=torch.int32, device=dev
    )

    fp8 = {}
    deq = {}
    for name, x in (("q", q), ("k", k), ("v", v)):
        scale = x.abs().amax().float() / 448.0
        x8 = (x / scale).to(torch.float8_e4m3fn)
        fp8[name] = x8
        fp8[name + "_descale"] = scale.view(1, 1).expand(c.batch, c.hk)
        deq[name] = x8.float() * scale
    return dict(
        fp8=fp8, deq=deq, page_table=page_table, seqused_k=seqused_k, cu_seqlens_q=cu_seqlens_q
    )


def reference(c: Case, inp) -> "torch.Tensor":
    """fp32 attention on the dequantized values, (total_q, hq, d)."""
    import torch

    q, k, v = inp["deq"]["q"], inp["deq"]["k"], inp["deq"]["v"]
    rep = c.hq // c.hk
    scale = 1.0 / math.sqrt(c.d)
    outs = []
    for b in range(c.batch):
        L = int(inp["seqused_k"][b])
        k0 = max(0, L - c.sq - (c.window - 1)) if c.window else 0
        blocks = inp["page_table"][b].long()
        kb = k[blocks].flatten(0, 1)[k0:L].repeat_interleave(rep, dim=1)
        vb = v[blocks].flatten(0, 1)[k0:L].repeat_interleave(rep, dim=1)
        qb = q[b * c.sq : (b + 1) * c.sq]
        s = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        i = torch.arange(c.sq, device=q.device)[:, None] + (L - c.sq)
        j = torch.arange(k0, L, device=q.device)[None, :]
        mask = j <= i
        if c.window:
            mask &= j >= i - (c.window - 1)
        s.masked_fill_(~mask, float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", s.softmax(-1), vb))
        del s, kb, vb
    return torch.cat(outs)


def make_fn(fwd, c: Case, dtype: str, inp, num_splits: int):
    import torch

    common = dict(
        cu_seqlens_q=inp["cu_seqlens_q"],
        seqused_k=inp["seqused_k"],
        max_seqlen_q=c.sq,
        max_seqlen_k=c.sk,
        page_table=inp["page_table"],
        causal=True,
        window_size_left=c.window - 1 if c.window else None,
        window_size_right=0 if c.window else None,
        num_splits=num_splits,
    )
    if dtype == "fp8":
        f = inp["fp8"]
        return lambda: fwd(
            f["q"], f["k"], f["v"],
            q_descale=f["q_descale"], k_descale=f["k_descale"], v_descale=f["v_descale"],
            out_dtype=torch.bfloat16, **common,
        )[0]
    t = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    q, k, v = (inp["deq"][n].to(t) for n in "qkv")
    return lambda: fwd(q, k, v, **common)[0]


class L2Flush:
    """D2D memcpy over a buffer larger than L2. Shows up as Memcpy, not a kernel."""

    def __init__(self, mb: int):
        import torch

        n = mb * 1024 * 1024
        self.src = torch.empty(n, dtype=torch.uint8, device="cuda")
        self.dst = torch.empty_like(self.src)

    def __call__(self):
        if self.src.numel():
            self.dst.copy_(self.src)


def time_graph(fn, flush: L2Flush, iters: int) -> float:
    """Median latency of one CUDA-graph replay (cold L2), in us."""
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
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush()
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    return t[len(t) // 2]


def time_kernels(fn, flush: L2Flush, iters: int):
    """Per-call sum of GPU kernel durations (cold L2) via the torch profiler, in us."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            flush()
            fn()
        torch.cuda.synchronize()
    kernels = []
    for e in prof.key_averages():
        if e.key.startswith(("Memcpy", "Memset")):
            continue
        total = getattr(e, "self_device_time_total", None)
        if total is None:
            total = getattr(e, "self_cuda_time_total", 0.0)
        if total <= 0:
            continue
        kernels.append(
            dict(name=e.key[:160], per_call=e.count / iters, us_per_call=total / iters)
        )
    kernels.sort(key=lambda k: -k["us_per_call"])
    return sum(k["us_per_call"] for k in kernels), kernels


def worker(args):
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    import torch

    fwd, src = import_fa4(Path(args.root))
    torch.backends.cuda.matmul.allow_tf32 = False
    git_rev = subprocess.run(
        ["git", "-C", args.root, "log", "-1", "--format=%h %s"], capture_output=True, text=True
    ).stdout.strip()
    print(f"[worker] {src}\n[worker] {git_rev}\n[worker] {torch.cuda.get_device_name()}")

    flush = L2Flush(args.flush_mb)
    dtypes = args.dtypes.split(",")
    cases = [CASES[n] for n in args.cases.split(",")]
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.profile_iters:
        fns = []
        for c in cases:
            inp = make_inputs(c)
            for dt in dtypes:
                fn = make_fn(fwd, c, dt, inp, args.num_splits)
                fn()  # JIT compile outside the capture range
                fns.append((f"{c.name}/{dt}", fn))
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for label, fn in fns:
            torch.cuda.nvtx.range_push(label)
            for _ in range(args.profile_iters):
                flush()
                fn()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(f"[worker] profiled {len(fns)} variant(s) x {args.profile_iters} iters")
        return

    results = dict(root=str(Path(args.root).resolve()), git=git_rev,
                   gpu=torch.cuda.get_device_name(), runs=[])
    for c in cases:
        inp = make_inputs(c)
        ref = reference(c, inp)
        for dt in dtypes:
            row = dict(case=c.name, dtype=dt, **{"shape": asdict(c)})
            try:
                fn = make_fn(fwd, c, dt, inp, args.num_splits)
                out = fn()
                torch.cuda.synchronize()
                err = out.float() - ref
                rel_l1 = (err.abs().mean() / ref.abs().mean()).item()
                bias = (err.mean() / ref.mean()).item()
                tol_l1, tol_bias = TOLERANCES[dt]
                row.update(
                    out_dtype=str(out.dtype),
                    max_abs=err.abs().max().item(),
                    rel_l1=rel_l1,
                    rel_bias=bias,
                    passed=bool(rel_l1 <= tol_l1 and abs(bias) <= tol_bias),
                )
                try:
                    row["graph_us"] = time_graph(fn, flush, args.iters)
                except Exception as e:
                    row["graph_us"] = float("nan")
                    row["graph_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                    print(f"[worker] CUDA graph capture failed: {row['graph_error']}")
                row["kernel_us"], row["kernels"] = time_kernels(fn, flush, args.prof_iters)
                if out_dir:
                    torch.save(out.cpu(), out_dir / f"{c.name}__{dt}.pt")
            except Exception as e:  # keep going: a version may not support a shape
                row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            results["runs"].append(row)
            msg = row.get("error") or (
                f"graph {row['graph_us']:7.1f} us  kernels {row['kernel_us']:7.1f} us  "
                f"rel_l1 {row['rel_l1']:.2e}  bias {row['rel_bias']:+.1e}  "
                f"{'PASS' if row['passed'] else 'FAIL'}"
            )
            print(f"[worker] {c.name:22s} {dt:5s} {msg}", flush=True)
            if args.verbose and "kernels" in row:
                for k in row["kernels"]:
                    print(f"           {k['us_per_call']:7.2f} us  x{k['per_call']:.0f}  {k['name']}")
        del inp, ref
        torch.cuda.empty_cache()
    if out_dir:
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))


# ----------------------------------------------------------------------------- compare


def compare(args):
    out_dir = Path(args.out_dir)
    baselines = [b for b in args.baselines.split(",") if b]
    plan = [
        ("orig", args.orig, baselines + ["fp8"]),
        ("accel", args.accel, ["fp8"] + (baselines if args.baselines_in_accel else [])),
    ]
    for label, root, dtypes in plan:
        cmd = [
            sys.executable, __file__, "worker", "--root", root, "--dtypes", ",".join(dtypes),
            "--cases", args.cases, "--out-dir", str(out_dir / label),
            "--iters", str(args.iters), "--prof-iters", str(args.prof_iters),
            "--flush-mb", str(args.flush_mb), "--num-splits", str(args.num_splits),
        ] + (["--verbose"] if args.verbose else [])
        print(f"\n=== {label}: {root}\n$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    res = {lbl: json.loads((out_dir / lbl / "results.json").read_text()) for lbl, *_ in plan}
    table = {}
    for lbl in res:
        for r in res[lbl]["runs"]:
            col = r["dtype"] if r["dtype"] != "fp8" else f"fp8-{lbl}"
            if lbl == "accel" and r["dtype"] != "fp8":
                col = f"{r['dtype']}-accel"
            table.setdefault(r["case"], {})[col] = r
    cols = baselines + ["fp8-orig", "fp8-accel"]
    if args.baselines_in_accel:
        cols += [f"{b}-accel" for b in baselines]

    for lbl in res:
        print(f"{lbl:6s}: {res[lbl]['git']}  ({res[lbl]['root']})")
    print(f"GPU: {res['orig']['gpu']}")

    def cell(r, key, fmt):
        if r is None:
            return "-"
        return "ERR" if "error" in r else format(r[key], fmt)

    for key, title in (("graph_us", "CUDA-graph latency, us (median, cold L2)"),
                       ("kernel_us", "sum of kernel durations per call, us (cold L2)")):
        print(f"\n{title}")
        hdr = f"{'case':22s}" + "".join(f"{c:>12s}" for c in cols)
        hdr += f"{'accel/orig':>12s}" + "".join(f"{'vs ' + b:>10s}" for b in baselines)
        print(hdr)
        for case, row in table.items():
            line = f"{case:22s}" + "".join(f"{cell(row.get(c), key, '.1f'):>12s}" for c in cols)
            a, o = row.get("fp8-accel"), row.get("fp8-orig")

            def ratio(num, width):
                good = all(r is not None and "error" not in r and r[key] == r[key] for r in (num, a))
                return f"{num[key] / a[key]:>{width - 1}.2f}x" if good else f"{'-':>{width}s}"

            line += ratio(o, 12) + "".join(ratio(row.get(b), 10) for b in baselines)
            print(line)

    print("\nCorrectness vs fp32 reference: rel_l1 / rel_bias / PASS|FAIL")
    for case, row in table.items():
        for c in cols:
            r = row.get(c)
            if r is None:
                continue
            if "error" in r:
                print(f"  {case:22s} {c:12s} ERROR {r['error']}")
            else:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"  {case:22s} {c:12s} {r['rel_l1']:.2e}  {r['rel_bias']:+.1e}  "
                      f"max_abs {r['max_abs']:.3f}  {status}")

    import torch

    print("\nFP8 accel vs FP8 orig output (same inputs):")
    for case in table:
        po = out_dir / "orig" / f"{case}__fp8.pt"
        pa = out_dir / "accel" / f"{case}__fp8.pt"
        if po.exists() and pa.exists():
            o, a = torch.load(po).float(), torch.load(pa).float()
            d = (a - o).abs()
            print(f"  {case:22s} max |diff| {d.max().item():.3e}  "
                  f"mean |diff| / mean |orig| {(d.mean() / o.abs().mean()).item():.2e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("worker", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--cases", default=DEFAULT_CASES, help=f"comma list of: {DEFAULT_CASES}")
        s.add_argument("--iters", type=int, default=200, help="CUDA-graph timing iterations")
        s.add_argument("--prof-iters", type=int, default=50, help="profiler timing iterations")
        s.add_argument("--flush-mb", type=int, default=128, help="L2 flush size; 0 = warm L2")
        s.add_argument("--num-splits", type=int, default=0, help="0 = split heuristic (vLLM)")
        s.add_argument("--verbose", action="store_true", help="print per-kernel breakdown")
    w = sub.choices["worker"]
    w.add_argument("--root", required=True, help="worktree to import flash_attn.cute from")
    w.add_argument("--dtypes", default="fp16,bf16,fp8")
    w.add_argument("--out-dir", default=None)
    w.add_argument("--profile-iters", type=int, default=0,
                   help="skip checks/timing; run N iters between cudaProfilerStart/Stop")
    c = sub.choices["compare"]
    c.add_argument("--orig", required=True, help="worktree with the original FP8 kernel")
    c.add_argument("--accel", required=True, help="worktree with the accelerated FP8 kernel")
    c.add_argument("--baselines", default="fp16,bf16", help="16-bit baselines, run in --orig")
    c.add_argument("--baselines-in-accel", action="store_true",
                   help="also run the 16-bit baselines in --accel (should match orig)")
    c.add_argument("--out-dir", default="fp8_results")
    args = p.parse_args()
    worker(args) if args.mode == "worker" else compare(args)


if __name__ == "__main__":
    main()
