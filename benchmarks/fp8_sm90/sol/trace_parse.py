#!/usr/bin/env python
"""Parse a speed-of-light results directory (run_sol.sh): traced runs, variants, microbenchmarks.

  python benchmarks/fp8_sm90/sol/trace_parse.py sol-results [--iter median|N]

Writes sol-results/report/:
  summary.md                  variant table, per-phase breakdowns per warp role, load latencies,
                              microbenchmark tables
  <label>.perfetto.json       one replay as a per-CTA / per-warp timeline (open in ui.perfetto.dev)
  <label>.timeline.png        Gantt of the same replay (needs matplotlib)
  <label>.segments.csv        every segment of every CTA of every replay
Needs numpy only (matplotlib optional). The slot map mirrors flash_attn/cute/sol_trace.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- slot map (sol_trace.py, v1)
TRACE_VERSION = 1
TRACE_MAGIC = 0x7ACEFA4D
HDR_MAGIC, HDR_SM, HDR_CTA_XY, HDR_CTA_Z, HDR_RANGE, HDR_ACTIVE = 0, 1, 2, 3, 4, 5
HDR_CAL, HDR_CAL_ITERS, HDR_OBS, HDR_TILE = 6, 9, 10, 11
MAX_TILES, TILE_STRIDE = 8, 8

W0 = dict(W0_ENTRY=16, W0_PREFETCH=17, W0_PT_ISSUED=18, W0_SEQ_ISSUED=19, W0_CMB_INIT=20,
          W0_PIPE_INIT=21, W0_BAR_DONE=22, P_START=23, P_RANGE=24, P_PT_READY=25, P_Q_ACQ=26,
          P_Q_ISSUED=27, P_HOLD_DONE=28, P_ALL_ISSUED=29, P_EXIT=30)
P_TILE = 32
PT_FIELDS = ["K_ACQ_START", "K_ACQ", "K_ISSUED", "V_ACQ", "V_ISSUED"]
OBS = dict(O_START=96, O_PT=97, O_Q=98, O_CMB=99, O_END=100)
O_K, O_V = 104, 112
CONS = dict(W4_ENTRY=120, W4_BAR_DONE=121, C_START=122, C_SETUP=123, C_Q=124, C_PREWAIT=125,
            C_LOOP_DONE=126, C_ROWSUM=127, C_FOLD=128, CMB_START=129, CMB_ARMED=130,
            CMB_CLUSTER=131, CMB_PUSHED=132, CMB_LANDED=133, CMB_DONE=134, C_EXIT=135)
C_TILE = 136
CT_FIELDS = ["K", "MMA_ISSUED", "V", "VT", "MMA_DONE", "FOLD", "P"]
CAL = dict(W2_CAL_START=200, W2_CAL_END=201)

# Segment names: a segment runs from the previous event of its warp to this event.
SEG_NAME = {
    "W0_PREFETCH": "kernel entry + TMA descriptor prefetch",
    "W0_PT_ISSUED": "page-table bulk copy issue",
    "W0_SEQ_ISSUED": "seqlen / descale preload issue",
    "W0_CMB_INIT": "combine mbarrier init + cluster arrive",
    "W0_PIPE_INIT": "pipeline mbarrier init",
    "W0_BAR_DONE": "CTA barrier (pipeline_init_wait)",
    "P_START": "setmaxreg (producer)",
    "P_RANGE": "split range",
    "P_PT_READY": "wait page-table row",
    "P_Q_ACQ": "acquire Q slot",
    "P_Q_ISSUED": "issue Q",
    "P_HOLD_DONE": "hold: wait K0 + Q landed",
    "P_ALL_ISSUED": "producer loop tail",
    "P_EXIT": "producer exit",
    "K_ACQ_START": "page lookup / next tile",
    "K_ACQ": "acquire K slot (empty wait)",
    "K_ISSUED": "issue K boxes",
    "V_ACQ": "acquire V slot (empty wait)",
    "V_ISSUED": "issue V boxes",
    "W4_BAR_DONE": "prologue + CTA barrier",
    "C_START": "setmaxreg (consumer)",
    "C_SETUP": "consumer setup (seqlen, masks)",
    "C_Q": "wait Q",
    "C_PREWAIT": "prewait: all stages landed",
    "K": "wait K",
    "MMA_ISSUED": "issue QK (+PV prev)",
    "V": "wait V",
    "VT": "load V^T to registers",
    "MMA_DONE": "wait WGMMAs",
    "FOLD": "fold O (prev tile)",
    "P": "softmax + P store",
    "C_LOOP_DONE": "issue last PV",
    "C_ROWSUM": "row-sum reduce",
    "C_FOLD": "wait last PV + fold",
    "CMB_START": "to merge",
    "CMB_ARMED": "merge: arm mbarrier",
    "CMB_CLUSTER": "merge: cluster barrier",
    "CMB_PUSHED": "merge: push partials (st.async)",
    "CMB_LANDED": "merge: wait remote partials",
    "CMB_DONE": "merge: combine + store O",
    "C_EXIT": "consumer exit",
    "W2_CAL_END": "calibration spin",
}


def category(name: str) -> str:
    if name.startswith("merge"):
        return "merge-wait" if ("wait" in name or "barrier" in name) else "merge"
    if name.startswith(("wait", "hold", "acquire", "prewait")):
        return "wait"
    if any(s in name for s in ("prologue", "entry", "setmaxreg", "init", "preload", "setup",
                               "page-table", "split range", "exit", "calibration", "to merge",
                               "barrier", "observer")):
        return "overhead"
    return "work"


COLORS = {"wait": "#d62728", "merge-wait": "#ff7f0e", "merge": "#2ca02c", "overhead": "#7f7f7f",
          "work": "#1f77b4"}


def slot_tables():
    """name -> slot for every timestamp slot, grouped by warp role."""
    prod = dict(W0)
    for t in range(MAX_TILES):
        for i, f in enumerate(PT_FIELDS):
            prod[f"{f}#{t}"] = P_TILE + TILE_STRIDE * t + i
    obs = dict(OBS)
    for t in range(MAX_TILES):
        obs[f"O_K#{t}"] = O_K + t
        obs[f"O_V#{t}"] = O_V + t
    cons = dict(CONS)
    for t in range(MAX_TILES):
        for i, f in enumerate(CT_FIELDS):
            cons[f"{f}#{t}"] = C_TILE + TILE_STRIDE * t + i
    return {"producer (warp 0)": prod, "observer (warp 1)": obs, "calibration (warp 2)": dict(CAL),
            "consumer (warp 4)": cons}


ROLES = slot_tables()
ROLE_TID = {"producer (warp 0)": 0, "observer (warp 1)": 1, "calibration (warp 2)": 2,
            "consumer (warp 4)": 4}


def seg_label(ev: str) -> str:
    base, _, t = ev.partition("#")
    name = SEG_NAME.get(base, base)
    return f"{name} [{t}]" if t else name


# ---------------------------------------------------------------- time reconstruction


def s32(x):
    return (x + 2**31) % 2**32 - 2**31


def reconstruct(tr: np.ndarray, sm_mhz: float | None):
    """tr: uint32 [iters, n_cta, 256, 2]. Returns (times ns [iters, n_cta, 256] with NaN for
    missing slots, info dict). Each CTA's events are placed with its SM clock (exact within the
    CTA) and anchored on the global timer: at the calibration edge when present (exact), else
    at the tightest of its events (globaltimer reads lag the true time by < one resolution)."""
    tr = tr.astype(np.int64)
    g, c = tr[..., 0], tr[..., 1]
    iters, n = g.shape[:2]
    valid = tr[:, :, HDR_MAGIC, 0] == TRACE_MAGIC
    present = (g != 0) | (c != 0)
    present[:, :, :16] = False
    entry = W0["W0_ENTRY"]
    # Unwrap globaltimer_lo per replay around the median entry.
    ref_g = np.zeros(iters, dtype=np.int64)
    for i in range(iters):
        ok = present[i, :, entry] & valid[i]
        ref_g[i] = int(np.median(g[i, ok, entry])) if ok.any() else 0
    dg = s32(g - ref_g[:, None, None])
    # Clock relative to the CTA's entry.
    ref_c = c[:, :, entry]
    dc = s32(c - ref_c[:, :, None])
    cal_g = s32(g[:, :, HDR_CAL:HDR_CAL + 3] - ref_g[:, None, None])
    cal_c = s32(c[:, :, HDR_CAL:HDR_CAL + 3] - ref_c[:, :, None])
    has_cal = (g[:, :, HDR_CAL:HDR_CAL + 3] != 0).all(-1) & valid
    res = np.diff(cal_g, axis=-1)[has_cal]
    resolution = float(np.median(res)) if res.size else float("nan")
    # Clock rate: every event read g with g <= true < g + res, and the first calibration edge
    # is exact, so each event bounds f = dc / dt from both sides. Intersect the bounds over all
    # events (robust quantiles), falling back to the median midpoint if they conflict.
    f = float("nan")
    if has_cal.any() and math.isfinite(resolution) and resolution > 0:
        eg, ec = cal_g[..., 0][..., None], cal_c[..., 0][..., None]
        dtc = (ec - dc).astype(np.float64)
        hi_t = (eg - dg).astype(np.float64)
        lo_t = hi_t - resolution
        ok = present & has_cal[..., None] & (dtc > 0) & (lo_t > 0)
        if ok.sum() >= 8:
            f_lo, f_hi = dtc[ok] / hi_t[ok], dtc[ok] / lo_t[ok]
            lo, hi = np.quantile(f_lo, 0.99), np.quantile(f_hi, 0.01)
            f = float((lo + hi) / 2) if lo <= hi else float(np.median((f_lo + f_hi) / 2))
    f_src = "calibration"
    if not math.isfinite(f) or not (0.5 < f < 3.0):
        f = (sm_mhz or 1980.0) / 1000.0
        f_src = "nvidia-smi" if sm_mhz else "assumed 1980 MHz"
    off = np.where(present, dg - dc / f, -np.inf).max(-1)  # tightest event per CTA
    off = np.where(has_cal, cal_g[..., 0] - cal_c[..., 0] / f, off)
    t = np.where(present, off[..., None] + dc / f, np.nan)
    t[~valid] = np.nan
    # Zero each replay at its first CTA entry.
    zero = np.nanmin(np.where(present[..., entry] & valid, t[..., entry], np.nan), axis=1)
    t = t - zero[:, None, None]
    info = dict(clock_ghz=f, clock_source=f_src, globaltimer_resolution_ns=resolution,
                cal_fraction=float(has_cal.mean()), iters=iters, n_cta=n,
                valid_fraction=float(valid.mean()))
    return t, info


def headers(tr, it=0):
    h = tr[it].astype(np.int64)
    return dict(
        smid=h[:, HDR_SM, 0], rank=h[:, HDR_SM, 1], bx=h[:, HDR_CTA_XY, 0], by=h[:, HDR_CTA_XY, 1],
        bz=h[:, HDR_CTA_Z, 0], nblocks=s32(h[:, HDR_CTA_Z, 1]), nmin=s32(h[:, HDR_RANGE, 0]),
        nmax=s32(h[:, HDR_RANGE, 1]), split=h[:, HDR_ACTIVE, 0], nsplits=h[:, HDR_ACTIVE, 1],
        head=h[:, HDR_TILE, 0], batch=h[:, HDR_TILE, 1], obs_iters=h[:, HDR_OBS, 0],
        obs_mask=h[:, HDR_OBS, 1],
    )


def role_events(t_cta: np.ndarray, role: str):
    """[(name, time)] of a role's present events in one CTA, in time order."""
    evs = [(nm, t_cta[s]) for nm, s in ROLES[role].items() if np.isfinite(t_cta[s])]
    order = {nm: i for i, nm in enumerate(ROLES[role])}
    return sorted(evs, key=lambda e: (e[1], order[e[0]]))


def cta_start(t_cta):
    return np.nanmin([t_cta[W0["W0_ENTRY"]], t_cta[CONS["W4_ENTRY"]]])


def segments(t_cta: np.ndarray, role: str):
    """[(label, start, end)] for a warp role; the first segment starts at the CTA's entry.
    The observer has one segment (its polling window); its landings are drawn by loads()."""
    if role == "observer (warp 1)":
        a, b = t_cta[OBS["O_START"]], t_cta[OBS["O_END"]]
        return [("observer polling", a, b)] if np.isfinite(a) and np.isfinite(b) else []
    evs = role_events(t_cta, role)
    out = []
    prev = cta_start(t_cta) if role != "calibration (warp 2)" else None
    for nm, tt in evs:
        if prev is not None and nm not in ("W0_ENTRY", "W4_ENTRY", "W2_CAL_START", "O_START"):
            out.append((seg_label(nm), prev, tt))
        prev = tt
    return out


def loads(t_cta: np.ndarray):
    """[(name, issue, landed)] of every async load the observer saw."""
    out = []
    def pair(name, issue_slot, land_slot):
        a, b = t_cta[issue_slot], t_cta[land_slot]
        if np.isfinite(b):
            out.append((name, a if np.isfinite(a) else np.nan, b))
    pair("page table", W0["W0_PT_ISSUED"], OBS["O_PT"])
    pair("Q", W0["P_Q_ISSUED"], OBS["O_Q"])
    for t in range(MAX_TILES):
        pair(f"K{t}", P_TILE + TILE_STRIDE * t + 2, O_K + t)
        pair(f"V{t}", P_TILE + TILE_STRIDE * t + 4, O_V + t)
    pair("combine partials", CONS["CMB_PUSHED"], OBS["O_CMB"])
    return out


# ---------------------------------------------------------------- stats helpers


def q(v, p):
    v = [x for x in v if x == x]
    return float(np.percentile(v, p)) if v else float("nan")


def fmt(x, nd=0):
    return "-" if x != x else f"{x:.{nd}f}"


def stat_row(name, v):
    return f"| {name} | {fmt(q(v, 50))} | {fmt(q(v, 10))} | {fmt(q(v, 90))} | {fmt(q(v, 100))} | {len([x for x in v if x == x])} |"


STAT_HDR = "| quantity | median ns | p10 | p90 | max | n |\n|---|---:|---:|---:|---:|---:|"


# ---------------------------------------------------------------- per-run analysis


def analyze_run(run_dir: Path, rep_dir: Path, which_iter: str):
    meta = json.loads((run_dir / "meta.json").read_text())
    label = meta["label"]
    trp = run_dir / "trace.npy"
    if not trp.exists():
        return meta, None, []
    tr = np.load(trp)
    sm_mhz = None
    try:
        sm_mhz = float(meta.get("clocks_start", "").split(",")[1].split()[0])
    except Exception:
        pass
    t, info = reconstruct(tr, sm_mhz)
    hdr = headers(tr)
    iters, n = t.shape[:2]
    md = [f"### {label}", ""]
    md.append(f"{meta['shape']}, variant `{meta['variant'] or '-'}`, stages "
              f"{meta['stages'] or 'default'}, page order {meta['page_order']}, flush "
              f"{meta['flush_mb']} MB; {n} CTAs x {iters} replays; clock {info['clock_ghz']:.3f} "
              f"GHz ({info['clock_source']}), globaltimer resolution "
              f"{fmt(info['globaltimer_resolution_ns'])} ns, calibrated CTAs "
              f"{100 * info['cal_fraction']:.0f}%.")
    def hist(v):
        u, cnt = np.unique(v, return_counts=True)
        return {int(a): int(b) for a, b in zip(u, cnt)}

    sms_all = tr[:, :, HDR_SM, 0].astype(np.int64)
    used, per = [], []
    for i in range(iters):
        bc = np.bincount(sms_all[i])
        used.append(int((bc > 0).sum()))
        per.append(int(bc.max()))
    sms = hdr["smid"]
    md.append(f"SMs used per replay: {hist(used)}; most CTAs on one SM per replay: {hist(per)}; "
              f"KV tiles per CTA: {hist(hdr['nblocks'])}; observer iterations (median): "
              f"{int(np.median(hdr['obs_iters']))}.")
    md.append("")

    # Kernel-level spans per replay.
    span, start_skew, exit_skew, life = [], [], [], []
    for i in range(iters):
        st = np.array([cta_start(t[i, k]) for k in range(n)])
        ex = t[i, :, CONS["C_EXIT"]]
        span.append(np.nanmax(ex) - np.nanmin(st))
        start_skew.append(np.nanmax(st) - np.nanmin(st))
        exit_skew.append(np.nanmax(ex) - np.nanmin(ex))
        life.extend(list(ex - st))
    md += ["**Kernel (traced build)**", "", STAT_HDR, stat_row("first CTA entry to last consumer exit", span),
           stat_row("CTA start skew (last entry - first entry)", start_skew),
           stat_row("CTA exit skew", exit_skew), stat_row("CTA lifetime (entry to consumer exit)", life), ""]

    # Segment durations per role.
    seg_d = defaultdict(lambda: defaultdict(list))
    rows = []
    for i in range(iters):
        for k in range(n):
            for role in ROLES:
                for lab, a, b in segments(t[i, k], role):
                    seg_d[role][lab].append(b - a)
                    rows.append((i, k, int(sms_all[i, k]), role, lab, a, b))
    for role in ("consumer (warp 4)", "producer (warp 0)"):
        md += [f"**{role}: segments in time order (each ends at the named event)**", "", STAT_HDR]
        # order by median start
        starts = defaultdict(list)
        for r in rows:
            if r[3] == role:
                starts[r[4]].append(r[5])
        for lab in sorted(seg_d[role], key=lambda x: q(starts[x], 50)):
            md.append(stat_row(lab, seg_d[role][lab]))
        md.append("")

    # Consumer phase totals.
    def phase_tot(i, k):
        tc = t[i, k]
        st = cta_start(tc)
        cs = segments(tc, "consumer (warp 4)")
        tot = defaultdict(float)
        for lab, a, b in cs:
            tot[category(lab)] += b - a
        first_k = tc[C_TILE + 0]
        loop_end = tc[CONS["C_FOLD"]]
        return dict(
            prologue=tc[CONS["W4_BAR_DONE"]] - st,
            to_first_compute=first_k - st,
            mainloop=loop_end - first_k,
            merge=tc[CONS["CMB_DONE"]] - tc[CONS["CMB_START"]],
            wait_total=tot["wait"], merge_wait=tot["merge-wait"], work=tot["work"],
            merge_work=tot["merge"], overhead=tot["overhead"])
    pt = defaultdict(list)
    for i in range(iters):
        for k in range(n):
            for kk, v in phase_tot(i, k).items():
                pt[kk].append(v)
    md += ["**Consumer phase totals per CTA**", "", STAT_HDR]
    for kk, lab in (("prologue", "entry to consumer start (prologue + barrier)"),
                    ("to_first_compute", "entry to K0 usable (first compute)"),
                    ("mainloop", "K0 usable to last O fold (main loop)"),
                    ("merge", "cluster merge"), ("wait_total", "sum of consumer data waits"),
                    ("work", "sum of consumer compute segments"),
                    ("merge_wait", "merge: waiting on cluster / remote partials"),
                    ("merge_work", "merge: own work"), ("overhead", "consumer overhead segments")):
        md.append(stat_row(lab, pt[kk]))
    md.append("")

    # Load latencies (issue -> landed) and consumer lag (landed -> consumer saw it).
    lat = defaultdict(list)
    lag = defaultdict(list)
    for i in range(iters):
        for k in range(n):
            tc = t[i, k]
            for nm, a, b in loads(tc):
                lat[nm].append(b - a)
            for tt in range(MAX_TILES):
                for nm, land, seen in ((f"K{tt}", O_K + tt, C_TILE + TILE_STRIDE * tt + 0),
                                       (f"V{tt}", O_V + tt, C_TILE + TILE_STRIDE * tt + 2)):
                    if np.isfinite(tc[land]) and np.isfinite(tc[seen]):
                        lag[nm].append(tc[seen] - tc[land])
    md += ["**Async loads: issue to landed (observer warp)**", "", STAT_HDR]
    for nm in lat:
        md.append(stat_row(nm, lat[nm]))
    md += ["", "**Consumer lag: load landed to consumer past its wait (>0: consumer was busy)**",
           "", STAT_HDR]
    for nm in lag:
        md.append(stat_row(nm, lag[nm]))
    md.append("")

    # Representative replay.
    if which_iter == "median":
        it = int(np.argsort(span)[len(span) // 2])
    else:
        it = min(int(which_iter), iters - 1)
    hdr_it = headers(tr, it)
    write_perfetto(rep_dir / f"{label}.perfetto.json", t[it], hdr_it, label)
    png = write_png(rep_dir / f"{label}.timeline.png", t[it], hdr_it, label, span[it])
    with open(rep_dir / f"{label}.segments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["replay", "cta", "smid", "role", "segment", "start_ns", "end_ns"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.1f}", f"{r[6]:.1f}"])
    md.append(f"Timeline of replay {it} (span {span[it]:.0f} ns): `{label}.perfetto.json`"
              + (f", `{label}.timeline.png`." if png else "."))
    md.append("")
    summary = dict(span=q(span, 50), life=q(life, 50), start_skew=q(start_skew, 50),
                   prologue=q(pt["prologue"], 50), first=q(pt["to_first_compute"], 50),
                   mainloop=q(pt["mainloop"], 50), merge=q(pt["merge"], 50),
                   merge_wait=q(pt["merge_wait"], 50), wait=q(pt["wait_total"], 50),
                   k0=q(lat.get("K0", []), 50), info=info)
    return meta, summary, md


def cta_name(hdr, k):
    return (f"CTA {k}: b{hdr['bz'][k]} h{hdr['bx'][k]} split{hdr['by'][k]} rank{hdr['rank'][k]} "
            f"SM{hdr['smid'][k]} ({hdr['nblocks'][k]} tiles)")


def write_perfetto(path: Path, t_it: np.ndarray, hdr, label):
    ev = []
    n = t_it.shape[0]
    order = sorted(range(n), key=lambda k: (hdr["bz"][k], hdr["bx"][k], hdr["by"][k]))
    for sort_idx, k in enumerate(order):
        pid = k + 1
        ev.append(dict(ph="M", pid=pid, name="process_name", args=dict(name=cta_name(hdr, k))))
        ev.append(dict(ph="M", pid=pid, name="process_sort_index", args=dict(sort_index=sort_idx)))
        for role, tid in ROLE_TID.items():
            ev.append(dict(ph="M", pid=pid, tid=tid, name="thread_name", args=dict(name=role)))
            for lab, a, b in segments(t_it[k], role):
                ev.append(dict(ph="X", pid=pid, tid=tid, name=lab, cat=category(lab),
                               ts=a / 1000.0, dur=max(b - a, 0) / 1000.0))
        for j, (nm, a, b) in enumerate(loads(t_it[k])):
            if np.isfinite(a):
                ev.append(dict(ph="b", pid=pid, tid=1, cat="load", id=f"{k}-{j}",
                               name=f"{nm} in flight", ts=a / 1000.0))
                ev.append(dict(ph="e", pid=pid, tid=1, cat="load", id=f"{k}-{j}",
                               name=f"{nm} in flight", ts=b / 1000.0))
            ev.append(dict(ph="i", pid=pid, tid=1, s="t", name=f"{nm} landed", ts=b / 1000.0))
    path.write_text(json.dumps(dict(traceEvents=ev, displayTimeUnit="ns",
                                    otherData=dict(label=label))))


def write_png(path: Path, t_it: np.ndarray, hdr, label, span):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return False
    n = t_it.shape[0]
    order = sorted(range(n), key=lambda k: (hdr["bz"][k], hdr["bx"][k], hdr["by"][k]))
    fig, ax = plt.subplots(figsize=(16, max(4, 0.16 * n * 2 + 1)))
    for y, k in enumerate(order):
        base = 2 * y
        for lab, a, b in segments(t_it[k], "consumer (warp 4)"):
            ax.barh(base, (b - a) / 1000, left=a / 1000, height=0.9, color=COLORS[category(lab)],
                    linewidth=0)
        for lab, a, b in segments(t_it[k], "producer (warp 0)"):
            ax.barh(base + 0.95, (b - a) / 1000, left=a / 1000, height=0.5,
                    color=COLORS[category(lab)], alpha=0.5, linewidth=0)
        for nm, a, b in loads(t_it[k]):
            mk = {"K": "|", "V": "x", "Q": "o", "p": "s", "c": "*"}.get(nm[0], ".")
            ax.plot([b / 1000], [base + 0.95], marker=mk, color="k", markersize=4)
    ax.set_yticks([2 * y for y in range(n)])
    ax.set_yticklabels([f"b{hdr['bz'][k]}h{hdr['bx'][k]}s{hdr['by'][k]}" for k in order],
                       fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("us since first CTA entry")
    ax.set_title(f"{label}: consumer warp (thick) and producer warp (thin) per CTA; markers = "
                 f"loads landed (| K, x V, o Q, s page table, * merge partials); span {span:.0f} ns")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items()], loc="lower right",
              fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- microbenchmarks


def microbench_md(path: Path):
    if not path.exists():
        return []
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip().startswith("{")]
    md = ["## Microbenchmarks", ""]
    by = defaultdict(list)
    for r in recs:
        by[r["test"]].append(r)
    if by["device"]:
        md += [f"Device: {by['device'][0]}", ""]
    if by["chase"]:
        md += ["**Dependent-load latency (pointer chase, 2048 hops)**", "",
               "| region | ns/hop (median of reps) | cycles/hop |", "|---|---:|---:|"]
        g = defaultdict(list)
        for r in by["chase"]:
            g[r["cfg"]].append(r)
        for cfg, rs in g.items():
            md.append(f"| {cfg} | {q([r['ns_per_hop'] for r in rs], 50):.0f} | "
                      f"{q([r['cycles_per_hop'] for r in rs], 50):.0f} |")
        md.append("")
    if by["bulk"]:
        md += ["**One CTA, one 16 KB tile: issue to landed**", "",
               "| copy | L2 | median ns | min ns | issue cycles |", "|---|---|---:|---:|---:|"]
        for r in by["bulk"]:
            md.append(f"| {r['mode']} | {r['l2']} | {r['median_ns']:.0f} | {r['min_ns']:.0f} | "
                      f"{r['median_issue_cycles']:.0f} |")
        md.append("")
    if by["burst"]:
        md += ["**N CTAs each issuing the kernel's first loads (cold L2): landing time after the "
               "CTA's issue, ns (median over CTAs and reps; max in parentheses)**", "",
               "| mode | cluster | CTAs | entry skew | K0 | Q | V0 | K1 | V1 | all landed (from first entry) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        g = defaultdict(list)
        for r in by["burst"]:
            g[(r["mode"], r["cluster"], r["n_cta"])].append(r)
        for (mode, cl, nc), rs in g.items():
            cells = []
            skew = [max(r["entry"]) - min(r["entry"]) for r in rs]
            for key in ("k0", "q", "v0", "k1", "v1"):
                v, m = [], []
                for r in rs:
                    d = [a - b for a, b in zip(r[key], r["issued"]) if a >= 0]
                    v += d
                    if d:
                        m.append(max(d))
                cells.append(f"{fmt(q(v, 50))} ({fmt(q(m, 50))})" if v else "-")
            last = []
            for r in rs:
                ends = [max(x for x in vals if x >= 0) for vals in
                        zip(r["k0"], r["q"], r["v0"], r["k1"], r["v1"]) if any(x >= 0 for x in vals)]
                if ends:
                    last.append(max(ends))
            md.append(f"| {mode} | {cl} | {nc} | {fmt(q(skew, 50))} | " + " | ".join(cells) +
                      f" | {fmt(q(last, 50))} |")
        md.append("")
    if by["const"]:
        md += ["**Kernel-parameter (constant bank) chase**", "",
               "| pattern | cold cycles/hop | warm cycles/hop | cold ns/hop |", "|---|---:|---:|---:|"]
        for r in by["const"]:
            md.append(f"| {r['cfg']} | {r['cold_cycles_per_hop']:.0f} | "
                      f"{r['warm_cycles_per_hop']:.0f} | {r['cold_ns_per_hop']:.0f} |")
        md.append("")
    if by["dsmem"]:
        md += ["**DSMEM (6-CTA cluster)**", "", "| peer rank | st.async round trip ns | "
               "cluster barrier ns |", "|---:|---:|---:|"]
        for r in by["dsmem"]:
            md.append(f"| {r['peer']} | {r['rtt_ns']:.0f} | {r['cluster_barrier_ns']:.0f} |")
        md.append("")
    if by["launch"]:
        md += ["**Empty kernel, grid 4x6x4 x 256 threads, 88 KB smem (CUDA graph, event-timed)**", "",
               "| kernel | cluster (1,6,1) | launches / graph | median us / launch | min |",
               "|---|---:|---:|---:|---:|"]
        for r in by["launch"]:
            md.append(f"| {r['kernel']} | {r['cluster']} | {r['launches_per_graph']} | "
                      f"{r['median_us_per_launch']:.2f} | {r['min_us_per_launch']:.2f} |")
        md.append("")
    if by["dispatch"]:
        md += ["**CTA dispatch skew of a 96-CTA launch (stamp at entry)**", "",
               "| cluster | median skew ns | p90 entry ns | distinct SMs |", "|---:|---:|---:|---:|"]
        for cl in (0, 1):
            rs = [r for r in by["dispatch"] if r["cluster"] == cl]
            if rs:
                md.append(f"| {cl} | {q([max(r['start_ns']) for r in rs], 50):.0f} | "
                          f"{q([q(r['start_ns'], 90) for r in rs], 50):.0f} | "
                          f"{q([len(set(r['smid'])) for r in rs], 50):.0f} |")
        md.append("")
    return md


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("results", help="run_sol.sh output directory")
    p.add_argument("--iter", default="median", help="replay to draw: median (by span) or index")
    args = p.parse_args()
    root = Path(args.results)
    rep = root / "report"
    rep.mkdir(exist_ok=True)
    runs = sorted((root / "runs").glob("*/meta.json"))
    table, details = [], []
    for m in runs:
        meta, summ, md = analyze_run(m.parent, rep, args.iter)
        table.append((meta, summ))
        details += md
        print(f"[parse] {meta['label']}")
    md = ["# Speed-of-light study: FP8 Gemma decode kernel", ""]
    md += ["## Runs", "",
           "Untraced timings come from the untraced builds; traced columns (medians, ns) from the "
           "timestamped build of the same configuration.", "",
           "| label | variant | stages | pass | graph us | kernel us | traced span | CTA life | "
           "start skew | prologue | entry to K0 usable | main loop | merge | merge wait | K0 issue to land |",
           "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for meta, s in table:
        s = s or {}
        md.append(
            f"| {meta['label']} | {meta['variant'] or '-'} | {meta['stages'] or '-'} | "
            f"{'PASS' if meta.get('passed') else 'FAIL'} | {meta.get('graph_us', float('nan')):.2f} | "
            f"{meta.get('kernel_us', float('nan')):.2f} | " + " | ".join(
                fmt(s.get(k, float("nan"))) for k in ("span", "life", "start_skew", "prologue",
                                                       "first", "mainloop", "merge", "merge_wait",
                                                       "k0")) + " |")
    md.append("")
    md += microbench_md(root / "microbench.jsonl")
    md += ["## Traced runs in detail", ""] + details
    (rep / "summary.md").write_text("\n".join(md) + "\n")
    print(f"[parse] wrote {rep / 'summary.md'}")


if __name__ == "__main__":
    main()
