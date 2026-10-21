#!/usr/bin/env bash
# nsys + ncu reports for one FA4 variant on one shape.
#
#   benchmarks/fp8_sm90/profile_kernels.sh <tree> <label> [case] [dtype]
#   benchmarks/fp8_sm90/profile_kernels.sh .           orig  gemma-swa-p16 fp8
#   benchmarks/fp8_sm90/profile_kernels.sh ../fa4-accel accel gemma-swa-p16 fp8
#   benchmarks/fp8_sm90/profile_kernels.sh .           fp16  gemma-swa-p16 fp16
#
# Env knobs: NCU_SET (default full), NCU_COUNT (kernels to capture, default 4),
#            NCU_CLOCKS (base|none, default base), REPORT_DIR (default ./reports),
#            NCU_KERNEL (ncu --kernel-name filter; check names in the nsys summary first).
set -euo pipefail

ROOT=${1:?worktree}
LABEL=${2:?label}
CASE=${3:-gemma-swa-p16}
DTYPE=${4:-fp8}
HERE=$(cd "$(dirname "$0")" && pwd)
REPORT_DIR=${REPORT_DIR:-$HERE/reports}
NCU_SET=${NCU_SET:-full}
NCU_COUNT=${NCU_COUNT:-4}
NCU_CLOCKS=${NCU_CLOCKS:-base}
NCU_KERNEL=${NCU_KERNEL:-regex:[Ff]lash|cutlass|[Cc]ombine}
mkdir -p "$REPORT_DIR"
TAG="${LABEL}_${CASE}_${DTYPE}"
RUN=(python "$HERE/fp8_compare.py" worker --root "$ROOT" --cases "$CASE" --dtypes "$DTYPE")

# Timeline: the capture range is the cudaProfilerStart/Stop window around the timed loop,
# so JIT compilation and warmup are excluded.
nsys profile -f true -o "$REPORT_DIR/nsys_$TAG" \
    --trace=cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    "${RUN[@]}" --profile-iters 20
nsys stats --force-overwrite true -f csv -o "$REPORT_DIR/nsys_$TAG" --report cuda_gpu_kern_sum \
    "$REPORT_DIR/nsys_$TAG.nsys-rep" >/dev/null
nsys stats --report cuda_gpu_kern_sum "$REPORT_DIR/nsys_$TAG.nsys-rep"

# Kernel metrics: line info for source-level SASS view; disk cache off so the lineinfo
# build is not satisfied by a cached cubin compiled without it.
CUTE_DSL_LINEINFO=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0 \
ncu -f -o "$REPORT_DIR/ncu_$TAG" \
    --profile-from-start off \
    --set "$NCU_SET" --import-source yes \
    --cache-control all --clock-control "$NCU_CLOCKS" \
    --kernel-name "$NCU_KERNEL" \
    --launch-count "$NCU_COUNT" \
    "${RUN[@]}" --profile-iters 2

echo "Reports in $REPORT_DIR:"
ls -1 "$REPORT_DIR" | grep "$TAG"
