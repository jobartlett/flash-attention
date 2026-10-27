#!/usr/bin/env bash
# Speed-of-light study driver for the SM90 FP8 decode kernel (run on an H200 node, inside the
# container, from the root of the main checkout).
#
#   bash benchmarks/fp8_sm90/sol/run_sol.sh TRACE_TREE OUT_DIR
#
# TRACE_TREE  worktree of branch fa4-sm90-native-fp8-d-trace (the kernel under test)
# OUT_DIR     results directory; copy the whole directory back for parsing
#
# STEPS selects what runs (default: "smoke micro trace variants parse"):
#   smoke     base untraced + traced, few iterations; aborts on errors or a correctness FAIL
#   micro     builds and runs sol_microbench.cu -> OUT_DIR/microbench.jsonl
#   trace     base traced (p16, p64), observer-off traced, calibration-off traced
#   variants  each variant untraced and traced (p16)
#   parse     trace_parse.py OUT_DIR -> OUT_DIR/report/
# ONLY="hold stream" restricts the variants step to those labels. Logs go to OUT_DIR/logs/.
set -uo pipefail

TREE=${1:?usage: run_sol.sh TRACE_TREE OUT_DIR}
OUT=${2:?usage: run_sol.sh TRACE_TREE OUT_DIR}
STEPS=${STEPS:-smoke micro trace variants parse}
ONLY=${ONLY:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=$(cd "$TREE" && pwd)
mkdir -p "$OUT/runs" "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
PY=${PY:-python3}
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0
export PYTHONUNBUFFERED=1

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

{
  date
  hostname
  nvidia-smi || true
  nvidia-smi -q -d CLOCK || true
  echo "main: $(git -C "$HERE" log -1 --format='%H %s')"
  echo "trace tree: $TREE $(git -C "$TREE" log -1 --format='%H %s')"
  git -C "$TREE" status --short
  $PY -c "import torch, cutlass; print('torch', torch.__version__, 'cutlass', cutlass.__version__)" || true
  nvcc --version || true
} > "$OUT/logs/env.txt" 2>&1

FAILED=()

# run LABEL [trace_run.py args...]: one process, one kernel build.
run() {
  local label=$1
  shift
  echo "=== $label: $*"
  if $PY "$HERE/trace_run.py" --root "$TREE" --label "$label" --out-dir "$OUT/runs" "$@" \
      > "$OUT/logs/$label.log" 2>&1; then
    grep -E "PASS|FAIL|graph|traced CTAs" "$OUT/logs/$label.log" | sed 's/^/    /'
  else
    echo "    ERROR (see logs/$label.log):"
    tail -n 15 "$OUT/logs/$label.log" | sed 's/^/    /'
    FAILED+=("$label")
    return 1
  fi
}

# pair LABEL [args...]: the untraced build (timing) and the traced build (timeline).
pair() {
  local label=$1
  shift
  if [[ -n "$ONLY" && " $ONLY " != *" $label "* ]]; then return 0; fi
  run "$label" "$@"
  run "$label-trace" --trace "$@"
}

if has_step smoke; then
  echo "##### smoke"
  for args in "base-smoke" "base-smoke-trace --trace"; do
    # shellcheck disable=SC2086
    run $args --iters 20 --prof-iters 5 --trace-iters 3 || { echo "smoke failed"; exit 1; }
    label=${args%% *}
    if grep -q FAIL "$OUT/logs/$label.log"; then echo "smoke: $label FAIL"; exit 1; fi
  done
fi

if has_step micro; then
  echo "##### micro"
  ( cd "$OUT" && nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo -o sol_microbench \
      "$HERE/sol_microbench.cu" -L"${CUDA_HOME:-/usr/local/cuda}/lib64/stubs" -lcuda \
      > logs/microbench_build.log 2>&1 ) || { echo "microbench build failed"; tail -n 30 "$OUT/logs/microbench_build.log"; FAILED+=(micro-build); }
  if [[ -x "$OUT/sol_microbench" ]]; then
    for sec in chase bulk burst const dsmem launch; do
      echo "=== microbench $sec"
      "$OUT/sol_microbench" "$sec" >> "$OUT/microbench.jsonl" 2>> "$OUT/logs/microbench.log" \
        || { echo "    ERROR (see logs/microbench.log)"; FAILED+=("micro-$sec"); }
    done
    if command -v nsys > /dev/null; then
      nsys profile -o "$OUT/microbench_launch" --force-overwrite true -t cuda \
        "$OUT/sol_microbench" launch > "$OUT/logs/microbench_nsys.log" 2>&1 || true
    fi
  fi
fi

if has_step trace; then
  echo "##### trace"
  run base-p16-trace --trace --case gemma-swa-p16
  run base-p64-trace --trace --case gemma-swa-p64
  run base-p64 --case gemma-swa-p64
  run noobs-trace --trace --variant noobs
  run nocal-trace --trace --variant nocal
fi

if has_step variants; then
  echo "##### variants"
  pair base
  pair hold --variant hold
  pair prewait --variant prewait
  pair stages3 --stages 3
  pair stages3-prewait --stages 3 --variant prewait
  pair stream --variant stream
  pair nomerge --variant nomerge
  pair stream-nomerge --variant stream,nomerge
  pair ptid --page-order identity
  pair noflush --flush-mb 0
  pair balanced --window 1149 --aligned-seqlens
  pair 1cluster --batch 1 --hk 1 --hq 2
  pair 4cluster --batch 1 --hk 4 --hq 8
  pair b16 --case gemma-swa-b16-p16
fi

if has_step parse; then
  echo "##### parse"
  $PY "$HERE/trace_parse.py" "$OUT" > "$OUT/logs/parse.log" 2>&1 \
    || { echo "parse failed (see logs/parse.log)"; FAILED+=(parse); }
fi

echo "##### done: $OUT"
if ((${#FAILED[@]})); then echo "errors in: ${FAILED[*]}"; exit 1; fi
