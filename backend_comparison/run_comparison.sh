#!/usr/bin/env bash
# Run benchmark_backend.py for each requested backend and emit a comparison report.
#
# Usage:
#   ./run_comparison.sh                    # usd, usdrt, tensor
#   ./run_comparison.sh usd usdrt          # explicit list
#   ./run_comparison.sh --num-prims 4096 usd usdrt tensor
#   NUM_PRIMS=2048 ITERS=1000 ./run_comparison.sh
#   ISAAC_PATH=/opt/isaac-sim ./run_comparison.sh    # explicit install dir
#
# Outputs:
#   /tmp/benchmark_<backend>.json   (one per backend)
#   /tmp/backend_comparison.md      (Markdown comparison)
#
# Notes:
# - The path to Isaac Sim's python.sh is resolved in this order:
#     1. The ISAAC_PATH environment variable (must contain python.sh).
#     2. ./python.sh relative to the current working directory.
#     3. ../python.sh relative to the current working directory.
#   Set ISAAC_PATH to the Isaac Sim install directory (the one that contains
#   python.sh) to skip the auto-detect.
# - Each backend is run in a fresh Python process so FSD can be toggled at app
#   launch time (it can't be changed mid-process).
# - usdrt / tensor require Fabric Scene Delegate, so the wrapper
#   passes --enable-fsd for them automatically.

set -euo pipefail

# ---- Defaults ----------------------------------------------------------
BACKENDS_DEFAULT=(usd usdrt tensor)
NUM_PRIMS=${NUM_PRIMS:-100}
ITERS=${ITERS:-500}
WARMUP=${WARMUP:-50}

# Resolve the script's directory first — we use it both to find the
# benchmark/compare scripts and to derive a portable default output dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default OUTPUT_DIR is `<repo>/output`. Inside the Isaac Sim container the
# repo is mounted at `/ws`, so this resolves to `/ws/output` — which the
# docker-compose bind-mounts to `<repo>/output` on the host.
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/../output"}
mkdir -p "${OUTPUT_DIR}"
echo "==> Outputs will be written to: ${OUTPUT_DIR}"

# ---- Locate the build root --------------------------------------------
# 1. Honor ISAAC_PATH if it points at a directory containing python.sh.
# 2. Otherwise fall back to the legacy auto-detect (./python.sh, ../python.sh).
if [[ -n "${ISAAC_PATH:-}" && -x "${ISAAC_PATH}/python.sh" ]]; then
    BUILD_DIR="$ISAAC_PATH"
elif [[ -x "./python.sh" ]]; then
    BUILD_DIR="$(pwd)"
elif [[ -x "../python.sh" ]]; then
    cd ..
    BUILD_DIR="$(pwd)"
else
    echo "error: could not find python.sh — set ISAAC_PATH to the Isaac Sim install" \
        "directory or run from a release build dir" >&2
    exit 1
fi
echo "==> Using Isaac Sim at: $BUILD_DIR"

BENCH="${SCRIPT_DIR}/benchmark_backend.py"
COMPARE="${SCRIPT_DIR}/compare_results.py"

# ``compare_results.py`` is pure stdlib, so any Python works. Prefer the host's
# ``python3`` when available; fall back to Isaac Sim's bundled launcher (which
# is the only Python guaranteed to be on PATH inside the Isaac Sim container).
PYTHON_CMD=$(command -v python3 2>/dev/null || echo "$BUILD_DIR/python.sh")
echo "==> Using Python interpreter for the compare step: $PYTHON_CMD"

if [[ ! -f "$BENCH" ]]; then
    echo "error: $BENCH not found" >&2
    exit 1
fi

# ---- Parse args --------------------------------------------------------
BACKENDS=()
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        usd|usdrt|tensor)
            BACKENDS+=("$arg")
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done
if [[ ${#BACKENDS[@]} -eq 0 ]]; then
    BACKENDS=("${BACKENDS_DEFAULT[@]}")
fi

# ---- Run each backend --------------------------------------------------
echo "==> Running ${#BACKENDS[@]} backend(s): ${BACKENDS[*]}"
for backend in "${BACKENDS[@]}"; do
    out_json="${OUTPUT_DIR}/benchmark_${backend}.json"
    echo
    echo "==> [${backend}] writing ${out_json}"
    fsd_flag=()
    case "$backend" in
        usdrt|tensor) fsd_flag=(--enable-fsd) ;;
    esac
    # shellcheck disable=SC2086
    "$BUILD_DIR/python.sh" "$BENCH" \
        --backend "$backend" \
        --num-prims "$NUM_PRIMS" \
        --iters "$ITERS" \
        --warmup "$WARMUP" \
        --output "$out_json" \
        "${fsd_flag[@]}" \
        "${EXTRA_ARGS[@]}"
done

# ---- Compare -----------------------------------------------------------
echo
echo "==> Producing comparison report"
REPORTS=()
for backend in "${BACKENDS[@]}"; do
    REPORTS+=("${OUTPUT_DIR}/benchmark_${backend}.json")
done

# Markdown -> stdout and to a file
"$PYTHON_CMD" "$COMPARE" "${REPORTS[@]}" --format markdown | tee "${OUTPUT_DIR}/backend_comparison.md"
# CSV as a machine-readable sidecar
"$PYTHON_CMD" "$COMPARE" "${REPORTS[@]}" --format csv > "${OUTPUT_DIR}/backend_comparison.csv"
echo
echo "==> Wrote ${OUTPUT_DIR}/backend_comparison.md and ${OUTPUT_DIR}/backend_comparison.csv"
