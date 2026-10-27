# Speed-of-light instrumentation for the SM90 native-FP8 swap-AB decode kernel.
#
# Off unless FA4_TRACE_ADDR is set. Every hook is a no-op Python call when tracing is off, so the
# untraced kernel is unchanged. With tracing on, lane 0 of chosen warps stores
# (globaltimer_lo, clock) pairs into a per-CTA record of TRACE_CTA_SLOTS 8-byte slots at
# FA4_TRACE_ADDR + linear_cta_id * TRACE_CTA_BYTES. The slot map below is mirrored in
# benchmarks/fp8_sm90/sol/trace_parse.py; keep the two in sync (TRACE_VERSION).
#
# FA4_SOL_VARIANT (comma list) selects experimental kernel changes; see SolConfig.

import os
from dataclasses import dataclass

import cutlass
from cutlass import Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

TRACE_VERSION = 1
TRACE_MAGIC = 0x7ACEFA4D
TRACE_CTA_SLOTS = 256
TRACE_CTA_BYTES = TRACE_CTA_SLOTS * 8
TRACE_MAX_TILES = 8
TRACE_TILE_STRIDE = 8

# Header, written with plain stores (values, not timestamps).
HDR_MAGIC = 0  # (magic, version)
HDR_SM = 1  # (smid, cluster_ctarank)
HDR_CTA_XY = 2  # (blockIdx.x, blockIdx.y)
HDR_CTA_Z = 3  # (blockIdx.z, num_blocks)
HDR_RANGE = 4  # (n_block_min, n_block_max)
HDR_ACTIVE = 5  # (split_idx, num_splits)
HDR_CAL = 6  # 6, 7, 8: (globaltimer_lo, clock) right after three globaltimer edges
HDR_CAL_ITERS = 9  # (spin iterations to edge 1, to edge 3)
HDR_OBS = 10  # (observer poll iterations, K landed mask | V landed mask << 16)
HDR_TILE = 11  # (head_idx, batch_idx)

# Warp 0 (prologue, then the TMA producer).
W0_ENTRY = 16
W0_PREFETCH = 17  # TMA descriptor prefetch issued
W0_PT_ISSUED = 18  # page-table row bulk copy issued
W0_SEQ_ISSUED = 19  # seqlen / descale preloads issued
W0_CMB_INIT = 20  # combine mbarrier init + cluster arrive
W0_PIPE_INIT = 21  # pipeline mbarrier inits done (before the CTA barrier)
W0_BAR_DONE = 22  # after pipeline_init_wait (CTA barrier)
P_START = 23  # producer branch entry (after setmaxreg)
P_RANGE = 24  # split range computed
P_PT_READY = 25  # page-table row in smem
P_Q_ACQ = 26
P_Q_ISSUED = 27
P_HOLD_DONE = 28  # variant hold: K0 and Q landed
P_ALL_ISSUED = 29
P_EXIT = 30
P_TILE = 32  # + 8 * t + field
PT_K_ACQ_START, PT_K_ACQ, PT_K_ISSUED, PT_V_ACQ, PT_V_ISSUED = range(5)

# Warp 1 (observer: true landing times of every async load).
O_START = 96
O_PT = 97
O_Q = 98
O_CMB = 99
O_END = 100
O_K = 104  # + t
O_V = 112  # + t

# Warp 4 (first consumer warp; the four consumer warps sync every tile).
W4_ENTRY = 120
W4_BAR_DONE = 121
C_START = 122
C_SETUP = 123  # masks / row bounds set up, about to wait for Q
C_Q = 124
C_PREWAIT = 125  # variant prewait: every stage's K and V landed
C_LOOP_DONE = 126  # last PV GEMM issued
C_ROWSUM = 127  # row sums reduced across warps
C_FOLD = 128  # last O tile folded
CMB_START = 129
CMB_ARMED = 130
CMB_CLUSTER = 131  # cluster barrier wait done
CMB_PUSHED = 132  # all st.async pushes issued
CMB_LANDED = 133  # every split's partial landed in this CTA
CMB_DONE = 134  # final O / LSE stored
C_EXIT = 135
C_TILE = 136  # + 8 * t + field
CT_K, CT_MMA_ISSUED, CT_V, CT_VT, CT_MMA_DONE, CT_FOLD, CT_P = range(7)

W2_CAL_START = 200
W2_CAL_END = 201


@dataclass(frozen=True)
class SolConfig:
    trace_addr: int = 0
    observer: bool = True  # noobs: no observer warp
    calibrate: bool = True  # nocal: no globaltimer edge calibration
    hold: bool = False  # hold: issue K0 + Q, wait until both land, then the rest
    prewait: bool = False  # prewait: consumers wait for every stage's K and V before computing
    stream: bool = False  # stream: consumers only wait for and release K/V (no math)
    nomerge: bool = False  # nomerge: skip the cluster combine (output is wrong)

    @property
    def trace(self) -> bool:
        return self.trace_addr != 0

    @property
    def key(self):
        return (
            self.trace_addr,
            self.observer,
            self.calibrate,
            self.hold,
            self.prewait,
            self.stream,
            self.nomerge,
        )


_KNOWN_VARIANTS = {"noobs", "nocal", "hold", "prewait", "stream", "nomerge"}


def sol_config_from_env() -> SolConfig:
    addr = int(os.environ.get("FA4_TRACE_ADDR", "0") or "0", 0)
    variants = {v for v in os.environ.get("FA4_SOL_VARIANT", "").split(",") if v}
    unknown = variants - _KNOWN_VARIANTS
    assert not unknown, f"unknown FA4_SOL_VARIANT entries {sorted(unknown)}"
    return SolConfig(
        trace_addr=addr,
        observer="noobs" not in variants,
        calibrate="nocal" not in variants,
        hold="hold" in variants,
        prewait="prewait" in variants,
        stream="stream" in variants,
        nomerge="nomerge" in variants,
    )


def sol_env_key():
    return tuple(
        os.environ.get(k, "") for k in ("FA4_TRACE_ADDR", "FA4_SOL_VARIANT", "FA4_SOL_STAGES")
    )


@dsl_user_op
def trace_ev(addr: cutlass.Int64, who: Int32, want: Int32, *, loc=None, ip=None) -> None:
    """Lane 0 of the warp with who == want stores (globaltimer_lo, clock) at addr."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(who).ir_value(loc=loc, ip=ip),
            Int32(want).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\t.reg .u32 g, c, l;\n\t"
        "mov.u32 l, %laneid;\n\t"
        "setp.eq.u32 p, l, 0;\n\t"
        "setp.eq.and.u32 p, $1, $2, p;\n\t"
        "mov.u32 g, %globaltimer_lo;\n\t"
        "mov.u32 c, %clock;\n\t"
        "@p st.global.v2.b32 [$0], {g, c};\n\t}",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _sreg_u32(name: str, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            f"mov.u32 $0, %{name};",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def read_globaltimer_lo(*, loc=None, ip=None) -> Int32:
    return _sreg_u32("globaltimer_lo", loc=loc, ip=ip)


@dsl_user_op
def read_clock(*, loc=None, ip=None) -> Int32:
    return _sreg_u32("clock", loc=loc, ip=ip)


@dsl_user_op
def read_timer_pair(*, loc=None, ip=None):
    """(globaltimer_lo, clock), read back to back."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [],
        "mov.u32 $0, %globaltimer_lo;\n\tmov.u32 $1, %clock;",
        "=r,=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def read_smid(*, loc=None, ip=None) -> Int32:
    return _sreg_u32("smid", loc=loc, ip=ip)


@dsl_user_op
def read_cluster_ctarank(*, loc=None, ip=None) -> Int32:
    return _sreg_u32("cluster_ctarank", loc=loc, ip=ip)


@dsl_user_op
def mbar_test_wait(mbar_addr: Int32, parity: Int32, *, loc=None, ip=None) -> Int32:
    """1 if the mbarrier phase with the given parity has completed (non-blocking)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(mbar_addr).ir_value(loc=loc, ip=ip), Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.test_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def nanosleep(ns: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
