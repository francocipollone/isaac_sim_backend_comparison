# Copyright 2026 Franco Cipollone
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Compare isaacsim.core.experimental.prims performance across USD / USDRT / tensor backends.

Spawns a grid of dynamic cubes and benchmarks the multi-backend hot-path methods
on the experimental prim wrappers:

  - RigidPrim: set_world_poses, get_world_poses,
              set_velocities, get_velocities
              (linear-only and angular-only variants via set_velocities(...,None)/set_velocities(None,...))
  - XformPrim (inherited): set_world_poses, get_world_poses
  - GeomPrim (USD-only baseline): set_collision_approximations, get_collision_approximations

The active backend is selected at runtime via ``backend_utils.use_backend(...)``.
This is a microbenchmark, not a representative physics workload: it measures the
per-call cost of the wrapper's read/write path, not PhysX stepping.

.. note::

    Backend support is **not** uniform across methods. The matrix below is
    derived from the docstrings of ``isaacsim.core.experimental.prims`` v6.0.
    Operations that aren't supported on a given backend are reported as
    ``"skipped"`` rather than crashing the run, so cross-backend tables stay
    readable.

    =====  ==========================  ==============================
    Method ``RigidPrim``              ``XformPrim``
    =====  ==========================  ==============================
    set/get_world_poses  tensor, usd, usdrt            usd, usdrt
    set/get_velocities   tensor, usd                  (n/a)
    =====  ==========================  ==============================

Usage
-----
    # Single backend run, writes /tmp/benchmark_usd.json
    ./python.sh benchmark_backend.py --backend usd --num-prims 1024 --iters 500

    # Use --enable-fsd to opt into Fabric Scene Delegate (required for usdrt):
    ./python.sh benchmark_backend.py --backend usdrt --num-prims 1024 --enable-fsd

The output JSON is consumed by ``compare_results.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# 1. ---- Args -----------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--backend",
    choices=["usd", "usdrt", "tensor"],
    default="usd",
    help="Backend to drive the multi-backend prim methods with. "
    "usdrt requires Fabric Scene Delegate (use --enable-fsd).",
)
parser.add_argument(
    "--num-prims",
    type=int,
    default=1024,
    help="Number of dynamic cubes to spawn in a 1D grid (default: 1024).",
)
parser.add_argument(
    "--iters",
    type=int,
    default=500,
    help="Iterations per benchmarked method (default: 500).",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=50,
    help="Warmup iterations excluded from the report (default: 50).",
)
parser.add_argument(
    "--enable-fsd",
    action="store_true",
    help="Enable Fabric Scene Delegate at app launch (required for usdrt).",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output JSON path (default: /tmp/benchmark_<backend>.json).",
)
parser.add_argument(
    "--headless",
    action="store_true",
    default=True,
    help="Run without a viewport (default: True).",
)
parser.add_argument(
    "--no-headless",
    dest="headless",
    action="store_false",
    help="Run with a viewport (useful when debugging).",
)
args = parser.parse_args()

# 2. ---- SimulationApp --------------------------------------------------
# Fabric Scene Delegate must be set BEFORE SimulationApp is constructed,
# because it controls which scene delegate is built. The usdrt backend
# requires it. Tensor works on top of FSD as well; pure usd is fine without.
settings: dict = {}
if args.enable_fsd:
    settings["/app/useFabricSceneDelegate"] = True

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "settings": settings})

# 3. ---- Imports (must come after SimulationApp) -----------------------
import numpy as np  # noqa: E402

import isaacsim.core.experimental.utils.backend as backend_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import Cube, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim  # noqa: E402

# 4. ---- Stage + prims --------------------------------------------------

stage_utils.create_new_stage()
GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])

paths = [f"/World/cube_{i:05d}" for i in range(args.num_prims)]
positions = np.zeros((args.num_prims, 3), dtype=np.float32)
positions[:, 0] = np.arange(args.num_prims) * 0.5  # spread out so they don't all collide
positions[:, 2] = 1.0
orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (args.num_prims, 1))

# Spawn the visual + collision mesh...
for i, path in enumerate(paths):
    Cube(paths=path, positions=positions[i], sizes=0.1)
# ...then wrap with the prim wrappers. Doing this in the requested backend.
with backend_utils.use_backend(args.backend, raise_on_unsupported=True):
    rigid = RigidPrim(paths=paths, positions=positions, orientations=orientations)
    xform = XformPrim(paths=paths)
    # GeomPrim always uses the USD path internally; the backend switch is irrelevant here.
    geom = GeomPrim(paths=paths, apply_collision_apis=True)

# Initialize physics so velocity methods have a real PhysX view to read.
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402

app_utils.play()
for _ in range(10):
    simulation_app.update()


# 5. ---- Benchmark harness ---------------------------------------------

def bench(label: str, fn, iters: int, warmup: int) -> dict:
    """Run ``fn`` iters times, drop the first ``warmup`` samples, return stats."""
    # warmup
    for _ in range(warmup):
        fn()
    simulation_app.update()  # let the runtime catch up between sections
    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1e3)
    samples_ms.sort()
    return {
        "label": label,
        "iters": iters,
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "stdev_ms": statistics.pstdev(samples_ms),
        "min_ms": samples_ms[0],
        "p95_ms": samples_ms[int(0.95 * (len(samples_ms) - 1))],
        "p99_ms": samples_ms[int(0.99 * (len(samples_ms) - 1))],
        "max_ms": samples_ms[-1],
        "ops_per_sec": 1000.0 / statistics.fmean(samples_ms),
    }


# Per-operation backend support, derived from the docstrings of
# ``isaacsim.core.experimental.prims`` v6.0. If ``args.backend`` is not in
# the supported set, the operation is reported as ``"skipped"`` instead of
# crashing the run. GeomPrim always goes through USD regardless of the active
# backend context, so it is supported on all three.
_OPERATION_BACKENDS: dict[str, set[str]] = {
    "RigidPrim.set_world_poses": {"tensor", "usd", "usdrt"},
    "RigidPrim.get_world_poses": {"tensor", "usd", "usdrt"},
    # RigidPrim.set_velocities / get_velocities: tensor, usd ONLY.
    "RigidPrim.set_velocities(linear,angular)": {"tensor", "usd"},
    "RigidPrim.get_velocities": {"tensor", "usd"},
    "RigidPrim.set_velocities(linear only)": {"tensor", "usd"},
    "RigidPrim.get_velocities[0]": {"tensor", "usd"},
    "RigidPrim.set_velocities(angular only)": {"tensor", "usd"},
    "RigidPrim.get_velocities[1]": {"tensor", "usd"},
    # XformPrim.set_world_poses / get_world_poses: usd, usdrt (NOT tensor).
    "XformPrim.set_world_poses": {"usd", "usdrt"},
    "XformPrim.get_world_poses": {"usd", "usdrt"},
    # GeomPrim is USD-only but is timed on every backend as a constant-cost
    # baseline.
    "GeomPrim.set_collision_approximations": {"tensor", "usd", "usdrt"},
    "GeomPrim.get_collision_approximations": {"tensor", "usd", "usdrt"},
}


def maybe_bench(label: str, fn, iters: int, warmup: int) -> dict:
    """Run ``bench`` if ``args.backend`` supports ``label``, else return a
    skipped marker. Keeps the cross-backend comparison table dense without
    crashing on unsupported backends.
    """
    supported = _OPERATION_BACKENDS.get(label)
    if supported is None or args.backend in supported:
        return bench(label, fn, iters, warmup)
    return {
        "label": label,
        "skipped": True,
        "reason": f"backend '{args.backend}' not in supported set {sorted(supported)}",
        "supported_backends": sorted(supported),
    }


def make_input_poses():
    """Generate per-iteration inputs (avoid paying the gen cost inside the timed region)."""
    pos = np.random.uniform(low=-0.05, high=0.05, size=(args.num_prims, 3)).astype(np.float32)
    pos[:, 2] = 1.0
    quat = np.random.randn(args.num_prims, 4).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    # quat = (w, x, y, z) per the prim API
    quat = np.roll(quat, 1, axis=-1)
    return pos, quat


def make_input_velocities():
    lin = np.random.uniform(-1.0, 1.0, size=(args.num_prims, 3)).astype(np.float32)
    ang = np.random.uniform(-1.0, 1.0, size=(args.num_prims, 3)).astype(np.float32)
    return lin, ang


input_poses = make_input_poses()
input_vels = make_input_velocities()

# Pre-bind so the timed lambda is the smallest possible.
with backend_utils.use_backend(args.backend, raise_on_unsupported=True):
    results: list[dict] = []

    # ---- RigidPrim (multi-backend) ----
    results.append(
        maybe_bench(
            "RigidPrim.set_world_poses",
            lambda: rigid.set_world_poses(positions=input_poses[0], orientations=input_poses[1]),
            args.iters,
            args.warmup,
        )
    )
    results.append(
        maybe_bench(
            "RigidPrim.get_world_poses",
            lambda: rigid.get_world_poses(),
            args.iters,
            args.warmup,
        )
    )
    # Note: ``set_velocities`` / ``get_velocities`` are the only public velocity
    # methods on ``RigidPrim`` in isaacsim 6.0 (linear-only and angular-only
    # variants are expressed by passing one of the args as ``None``).
    results.append(
        maybe_bench(
            "RigidPrim.set_velocities(linear,angular)",
            lambda: rigid.set_velocities(
                linear_velocities=input_vels[0], angular_velocities=input_vels[1]
            ),
            args.iters,
            args.warmup,
        )
    )
    results.append(
        maybe_bench(
            "RigidPrim.get_velocities",
            lambda: rigid.get_velocities(),
            args.iters,
            args.warmup,
        )
    )
    results.append(
        maybe_bench(
            "RigidPrim.set_velocities(linear only)",
            lambda: rigid.set_velocities(linear_velocities=input_vels[0]),
            args.iters,
            args.warmup,
        )
    )
    results.append(
        maybe_bench(
            "RigidPrim.get_velocities[0]",
            lambda: rigid.get_velocities()[0],
            args.iters,
            args.warmup,
        )
    )
    results.append(
        maybe_bench(
            "RigidPrim.set_velocities(angular only)",
            lambda: rigid.set_velocities(angular_velocities=input_vels[1]),
            args.iters,
            args.warmup,
        )
    )
    results.append(
        maybe_bench(
            "RigidPrim.get_velocities[1]",
            lambda: rigid.get_velocities()[1],
            args.iters,
            args.warmup,
        )
    )

    # ---- XformPrim (inherited, multi-backend) ----
    results.append(
        maybe_bench(
            "XformPrim.set_world_poses",
            lambda: xform.set_world_poses(positions=input_poses[0], orientations=input_poses[1]),
            args.iters,
            args.warmup,
        )
    )
    results.append(
        maybe_bench(
            "XformPrim.get_world_poses",
            lambda: xform.get_world_poses(),
            args.iters,
            args.warmup,
        )
    )

# ---- GeomPrim (USD-only) ----
# GeomPrim does not respect the backend context manager for its own methods; it
# always goes through the USD stage. We time it anyway to make the difference
# visible: if the bench is much faster than RigidPrim.set_world_poses, that
# reflects "the wrapper does nothing hot" rather than "USDRT is fast."
def set_collision_approx():
    geom.set_collision_approximations("convexHull")


results.append(maybe_bench("GeomPrim.set_collision_approximations", set_collision_approx, args.iters, args.warmup))
results.append(maybe_bench("GeomPrim.get_collision_approximations", lambda: geom.get_collision_approximations(), args.iters, args.warmup))

# 6. ---- Report --------------------------------------------------------

import platform  # noqa: E402

report = {
    "backend": args.backend,
    "fsd_enabled": bool(settings.get("/app/useFabricSceneDelegate", False)),
    "num_prims": args.num_prims,
    "iters": args.iters,
    "warmup": args.warmup,
    "host": platform.node(),
    "python": platform.python_version(),
    "results": results,
}

out_path: Path = args.output or Path(f"/tmp/benchmark_{args.backend}.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(report, indent=2))
print(f"\nWrote {out_path}\n")

# Pretty-print a quick table to stdout so the user can eyeball it before
# opening the JSON.
header = f"{'operation':<40} {'mean(ms)':>10} {'stdev':>10} {'median':>10} {'p95':>10} {'p99':>10} {'ops/s':>12}"
print(header)
print("-" * len(header))
for r in results:
    if r.get("skipped"):
        print(f"{r['label']:<40} {'skipped':>10}  ({r['reason']})")
        continue
    print(
        f"{r['label']:<40} "
        f"{r['mean_ms']:>10.4f} {r['stdev_ms']:>10.4f} {r['median_ms']:>10.4f} "
        f"{r['p95_ms']:>10.4f} {r['p99_ms']:>10.4f} {r['ops_per_sec']:>12.0f}"
    )

simulation_app.close()
