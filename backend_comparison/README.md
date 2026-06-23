# USD vs USDRT (and tensor) — backend microbenchmark

Microbenchmark for the multi-backend hot path of the experimental prim
wrappers in `isaacsim.core.experimental.prims`.

It does **not** measure end-to-end physics throughput. It measures the
per-call cost of the wrapper's read/write path: how long it takes to
`set_world_poses` / `get_world_poses` / `set_velocities` / ... on a
batch of N prims, when the wrapper is steered at a specific backend
via [`backend_utils.use_backend`](../../../../exts/isaacsim.core.experimental.utils/isaacsim/core/experimental/utils/backend.py).

## What the benchmark exercises

The matrix below is the **actual** backend support as documented on the
respective methods in `isaacsim.core.experimental.prims` v6.0. It is
**not** uniform across methods — see §"Backend support matrix" below for
the consequences.

| Wrapper | Method | Backends (per docstring) |
|---|---|---|
| `RigidPrim` | `set_world_poses` | tensor, usd, usdrt, fabric |
| `RigidPrim` | `get_world_poses` | tensor, usd, usdrt, fabric |
| `RigidPrim` | `set_velocities` / `get_velocities` | **tensor, usd** |
| `RigidPrim` | `set_velocities(linear only)` (via `set_velocities(linear, None)`) | **tensor, usd** |
| `RigidPrim` | `get_velocities()[0]` (linear) | **tensor, usd** |
| `RigidPrim` | `set_velocities(angular only)` (via `set_velocities(None, angular)`) | **tensor, usd** |
| `RigidPrim` | `get_velocities()[1]` (angular) | **tensor, usd** |
| `XformPrim` | `set_world_poses` / `get_world_poses` | usd, usdrt, fabric |
| `GeomPrim` | `set_collision_approximations` / `get_collision_approximations` | usd only (baseline) |

> **Why no separate `set_linear_velocities` / `set_angular_velocities`?**
> Those methods do not exist on `RigidPrim` in isaacsim 6.0. The single
> `set_velocities(linear, angular)` method covers all three cases by
> passing one of the args as `None`.

## Prerequisites

Run from a release build:

```bash
cd _build/linux-x86_64/release
```

`usdrt` and `tensor` backends require Fabric Scene Delegate.
The driver script enables it automatically for those backends; if you
run the benchmark directly, add `--enable-fsd`.

## Run a single backend

```bash
./python.sh benchmark_backend.py --backend usd   --num-prims 1024 --iters 500
./python.sh benchmark_backend.py --backend usdrt --num-prims 1024 --iters 500 --enable-fsd
./python.sh benchmark_backend.py --backend tensor --num-prims 1024 --iters 500 --enable-fsd
```

Each invocation writes `/tmp/benchmark_<backend>.json` and prints a
quick table to stdout:

```
operation                                      mean(ms)     median        p95        p99       ops/s
----------------------------------------------------------------------------------------------------
RigidPrim.set_world_poses                       0.8721     0.8622     0.9510     1.0210       1147
RigidPrim.get_world_poses                       0.6104     0.6044     0.7011     0.7982       1638
...
```

## Run a full comparison

```bash
./run_comparison.sh
# equivalent to:
NUM_PRIMS=1024 ITERS=500 ./run_comparison.sh usd usdrt tensor

# Tweak scale and re-run:
NUM_PRIMS=4096 ITERS=2000 ./run_comparison.sh usd usdrt
```

This produces three artefacts under `/tmp/`:

| File | Format | Purpose |
|---|---|---|
| `benchmark_<backend>.json` | JSON | Raw per-op timings (mean, median, p95, p99, ops/s) |
| `backend_comparison.md` | Markdown | Side-by-side table, with relative speedup vs. the reference backend |
| `backend_comparison.csv` | CSV | Same data, machine-readable |

## How the backend is selected

```python
import isaacsim.core.experimental.utils.backend as backend_utils
from isaacsim.core.experimental.prims import RigidPrim

with backend_utils.use_backend("usdrt", raise_on_unsupported=True):
    rigid = RigidPrim(paths=paths)             # construction is backend-agnostic
    rigid.set_world_poses(positions, quats)     # this call goes through USDRT
# outside the block: back to the default (usd)
```

The first positional arg of `use_backend` is one of `"usd"`, `"usdrt"`,
`"fabric"`, `"tensor"`, `"simstate"`. The wrapper reads the current
backend via `backend_utils.get_current_backend(...)` and dispatches
the call. Methods that are USD-only (like every `GeomPrim` method) are
unaffected by the context manager.

`raise_on_unsupported=True` makes the wrapper *error* (rather than
silently fall back to USD) if the requested backend is not available
on the active prim. Useful for catching misconfigurations in CI.

## Backend support matrix

Not every backend supports every method. The benchmark **does not**
crash when a method isn't supported on the active backend — it records
the operation as `"skipped"` in the JSON report and renders
`"skipped (<reason>)"` in stdout. The cross-backend comparison table
then shows `n/a` for that cell.

The matrix below is hard-coded in
[`benchmark_backend.py::_OPERATION_BACKENDS`](benchmark_backend.py)
and matches the docstring of every method it covers:

| Operation | `usd` | `tensor` | `usdrt` |
|---|:---:|:---:|:---:|
| `RigidPrim.set_world_poses`        | ✓ | ✓ | ✓ |
| `RigidPrim.get_world_poses`        | ✓ | ✓ | ✓ |
| `RigidPrim.set_velocities(linear, angular)` | ✓ | ✓ | — |
| `RigidPrim.get_velocities`         | ✓ | ✓ | — |
| `RigidPrim.set_velocities(linear only)`     | ✓ | ✓ | — |
| `RigidPrim.get_velocities()[0]` (linear)    | ✓ | ✓ | — |
| `RigidPrim.set_velocities(angular only)`    | ✓ | ✓ | — |
| `RigidPrim.get_velocities()[1]` (angular)   | ✓ | ✓ | — |
| `XformPrim.set_world_poses`        | ✓ | — | ✓ |
| `XformPrim.get_world_poses`        | ✓ | — | ✓ |
| `GeomPrim.set_collision_approximations`  | ✓ (USD-only baseline, run on every backend) |
| `GeomPrim.get_collision_approximations`  | ✓ (USD-only baseline, run on every backend) |

The `RigidPrim` velocity methods have no `usdrt` / `fabric` dispatch in
the wrapper — they always go through the `tensor` view (when available)
or `UsdPhysics.RigidBodyAPI.SetVelocityAttr` (otherwise). Trying to
benchmark them on `usdrt` or `fabric` with `raise_on_unsupported=True`
in `use_backend(...)` raises a `RuntimeError`; the benchmark instead
records them as skipped so the comparison table stays readable.

## How to read the output

Speedup is reported as `reference / current` expressed as a multiplier.
A value of `3.86×` means "this backend is 3.86× faster than the reference"
(the reference takes 3.86× as long per call). `>1×` is faster; `<1×` is
slower; `1×` is a tie. The first column is the reference backend's mean
per-call time in ms.

What to look for:

- **`RigidPrim.set_world_poses` / `get_world_poses`** are the most
  representative. USDRT and Fabric avoid the per-attribute USD attr
  read/write; they read from a Fabric scene delegate that mirrors
  the stage. Expect a large drop in mean time per call once N grows
  past a few hundred prims.
- **`set_velocities` (linear + angular together)** is the path that
  goes through the PhysX tensor view; for very large N, the tensor
  backend may beat the USDRT path because it writes one contiguous
  block. It is only meaningful to compare `tensor` vs `usd` here —
  the other backends skip it.
- **`XformPrim.set_world_poses` / `get_world_poses`** are the
  inherited path that doesn't touch the physics tensor view at all;
  they show the "pure" USDRT/Fabric speedup over USD with no tensor
  shortcut in the mix.
- **`GeomPrim.set_collision_approximations`** will be roughly the
  same across all backends because it is USD-only. Use it as a
  sanity check that the harness isn't accidentally measuring
  something else.

## Caveats

- The benchmark times Python-side wall time. To see GPU/carb
  profiler-level hotspots (which the user-space timer won't reveal),
  wrap with the Tracy capture from `tools/profiling/tracy_capture.py`
  instead — see the `profile-isaac-sim` skill.
- `RigidPrim.set_world_poses` in the `tensor` backend *teleports*
  the rigid body (bypasses PhysX integration). If you care about
  accurate contact behaviour, do not benchmark that path on
  contacting stacks — use `apply_forces` / `set_velocities` for
  physically meaningful comparisons.
- The benchmark runs in headless mode by default. Pass `--no-headless`
  to open a viewport (only useful when debugging the harness).
