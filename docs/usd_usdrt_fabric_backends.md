# Fabric, USDRT, and USD — A Reference

> Living document for the Isaac Sim 6.0 USD backends. Captures the mental
> model, the configuration knobs (in the experience file and via carb), the
> Python APIs, and concrete recommendations for the benchmark harness in
> this repo.
>
> Sources: [`isaacsim.core.simulation_manager`](https://docs.omniverse.nvidia.com/isaacsim/latest),
> [`isaacsim.core.experimental.utils.stage`](https://docs.omniverse.nvidia.com/isaacsim/latest),
> [`example_experience.kit`](../../backend_comparison/example_experience.kit) (a **reference example**
> of a Fabric/USDRT-enabled experience — the benchmark harness does **not**
> load it; see §2.1.1 for what the harness actually does), and
> [`backend_comparison/benchmark_backend.py`](../../backend_comparison/benchmark_backend.py)
> (the harness that exercises the multi-backend prim dispatch paths in §3.7).

---

## TL;DR

|                             | pxr.Usd (slow-path)                              | usdrt.Usd (Fabric)                               |
| --------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| Module                      | `pxr.Usd`, `pxr.UsdGeom`, …                      | `usdrt.Usd`, `usdrt.UsdGeom`, …                  |
| Stage accessor              | `omni.usd.get_context().get_stage()`             | `usdrt.Usd.Stage.Attach(stage_id)`               |
| Helper (explicit)           | `stage_utils.get_current_stage(backend="usd")`   | `stage_utils.get_current_stage(backend="usdrt")` |
| Helper (implicit default)   | `with backend_utils.use_backend("usd"):`         | `with backend_utils.use_backend("usdrt"):`       |
| Storage                     | Python object graph                              | C++ struct-of-arrays, GPU-bindable               |
| Best for                    | Authoring, layer stack, schemas, one-off queries | Hot loops, live physics state, Warp kernels      |
| Always available?           | Yes                                              | Yes, when the Fabric extensions are loaded       |
| Carries live physics state? | No (by default)                                  | Yes                                              |

There is **no global "current stage"** that flips between backends. You grab
whichever handle you need per call. The two are parallel in-memory
representations of the same scene, kept in sync by Fabric's change-notification
system. The `use_backend` row above is the implicit counterpart of the
explicit `backend=` arg: it sets a thread-local default that
`stage_utils.get_current_stage(backend=None)` and the experimental prim
wrappers consult on entry. See §3.7 for the full rules.

The slow-path (`pxr.Usd`) is the _always-there_ half of the table: once a
stage is opened, the `pxr.Usd.Stage` lives in memory for the rest of the
session, regardless of FSD. Fabric is the optional mirror on top. See
§1.3 for the full picture.

---

## Backends at a glance

Four backend strings appear in the API. They are **not** interchangeable concepts, and three of them regularly confuse people:

| String     | What it actually is                                      | A stage? | Requires FSD?  | Detail       |
| ---------- | -------------------------------------------------------- | -------- | -------------- | ------------ |
| `"usd"`    | `pxr.Usd.Stage` (always available)                       | yes      | no             | §1, §4       |
| `"usdrt"`  | `usdrt.Usd.Stage`                                        | yes      | yes            | §1, §4, §2.5 |
| `"fabric"` | **Alias for `"usdrt"`** — same `usdrt.Usd.Stage`         | yes      | yes            | §3.3         |
| `"tensor"` | Warp physics tensor view (not a stage accessor)          | **no**   | different gate | §3.7         |

**Two terminology traps:**

- **`usdrt` (library) ≠ Fabric (FSD runtime).** They ship as a pair: the library exposes the Python module (`import usdrt`); FSD is the C++ SoA runtime backing the hot path. See §2.5 for the full picture.
- **`"tensor"` is a third, separate backend.** It only exists on `RigidPrim`, is its **default** if you don't open `use_backend(...)`, and uses **teleport semantics** (writes through the physics tensor view, bypassing PhysX integration). See §3.7.

If you only remember three things: (1) `"usdrt"` and `"fabric"` are the same string (§3.3). (2) `"tensor"` is the `RigidPrim` default with teleport semantics (§3.7). (3) `"usdrt"` / `"fabric"` require Fabric Scene Delegate, and the check fires inside `use_backend(...)` (§3.7).

---

## 1. Mental model

There are _three_ representations of the scene, layered. **The .usd file is
the source of truth** for authored content; both in-memory stages are views
over it.

```
                       ┌──────────────────────────────┐
                       │  .usd file on disk           │  ← source of truth
                       │  (layers, composition arcs)  │
                       └───────────────┬──────────────┘
                                       │  Usd.Stage opens layers
                                       ▼
   ┌────────────────────────────────────────────────────────────┐
   │  Slow-path stage  (pxr.Usd.Stage)                          │
   │  • full USD API: layers, schemas, composition, authoring   │
   │  • Python-friendly, but slow for hot loops                 │
   │  • gets authored prims and their authored attributes       │
   │  • does NOT receive per-step physics updates by default    │
   └────────────────────────────────────────────────────────────┘
                                       ▲
                                       │  Fabric scene delegate
                                       │  syncs authored content
                                       │  and writes physics state
                                       ▼
   ┌────────────────────────────────────────────────────────────┐
   │  Fabric stage  (usdrt.Usd.Stage)                           │
   │  • subset of USD: prims, attributes, transforms, relations │
   │  • no layer stack, no schema authoring, no composition API │
   │  • C++ SoA storage; very fast reads                        │
   │  • receives live physics state (transforms, velocities,    │
   │    joint states) when /physics/fabricUpdate* = true        │
   │  • binds to Warp wp.fabricarray for zero-copy GPU access   │
   └────────────────────────────────────────────────────────────┘
                                       ▲
                                       │  reads from Fabric
                                       │
                            ┌──────────┴──────────┐
                            │  Renderer / viewport│
                            │  (Hydra, RTX)       │
                            └─────────────────────┘
```

Key properties:

- The **two stages are parallel**, not nested. Neither is "the real one."
- A `pxr.Usd.Prim` and a `usdrt.Usd.Prim` at the same path refer to the same
  logical prim, but they are **different Python types** with no shared
  identity. You cannot pass one to the other side's API.
- Path **strings** (`"/World/Camera"`) flow through fine — both backends
  consume strings.
- The two stages are **eventually consistent**, not transactional. After a
  write to one side, the other side may lag by one Fabric notification
  (typically one `kit` tick).
- The **renderer** is the one place with a global "which side do I read
  from" knob, controlled by `useFabricSceneDelegate` in the experience file.
  Your Python code is not subject to that knob.

### Mental shortcut

> **Fabric is a faster mirror for reads. Pick the slow-path when you need
> USD's authoring surface, pick Fabric when you're in a hot loop or need
> live physics state. Both are always available.**

### 1.1 Stage initialization timeline

The chain of trust is `.usd file → slow-path stage → Fabric stage` —
**Fabric is seeded from slow-path, not directly from the .usd file**.
There are five distinct phases:

```
[App startup]
    │   omni.physx.fabric, usdrt.scenegraph, omni.physics.stageupdate
    │   are loaded. Python bindings ready. Fabric runtime ready.
    │   No stages exist yet — there's no Fabric data to query.
    ▼
[Stage open]  ← omni.usd.get_context().open_stage(path) / SimulationApp
    │   1. .usd file is parsed.
    │   2. Slow-path pxr.Usd.Stage is built: layers, composition arcs,
    │      prim hierarchy, authored attributes.
    │   3. Fabric scene delegate is wired to that stage.
    │   4. Fabric stage is seeded from the slow-path:
    │        - prim hierarchy is mirrored (Xform tree, types, names)
    │        - authored attribute values are mirrored
    │        - special "init" prims (e.g. /ExternalSimulationTime) are
    │          seeded to default values (typically 0)
    │   This all happens during the first stage-update tick after open —
    │   typically a few ms. Before that, Fabric queries return empty.
    ▼
[Pre-play]
    │   Fabric is fully queryable. Authoring edits via pxr propagate to
    │   Fabric after one kit tick. NO physics state has been written yet.
    │   Joints are at authored value (often 0). Body transforms are at
    │   authored value. The sim clock is at 0.
    ▼
[Play]
    │   PhysX steps begin. Per step:
    │     - PhysX writes transforms/velocities/joint-states/force-sensors
    │       to Fabric (via physics.fabricUpdate*).
    │     - /physics/updateToUsd = false → no mirror to slow-path.
    │   Live state is now visible on Fabric. Authored content keeps
    │   mirroring both ways.
    ▼
[Stop]
    │   PhysX stops stepping. Fabric state freezes at last-step value.
    │   Authored content continues mirroring.
```

The key answer: **the .usd content is in Fabric's memory at stage-open time,
not at play.** Play is what starts the _physics-driven_ writes; the authored
scene is already there.

### 1.2 What you get before play

`stage_utils.get_current_stage(backend="usdrt")` returns a `usdrt.Usd.Stage`
that is **fully populated with the authored scene**, _not_ an empty or
partial object. Concretely:

| What                                           | Before play                         | After play                            |
| ---------------------------------------------- | ----------------------------------- | ------------------------------------- |
| Prim hierarchy (`/World/Robot/base_link` etc.) | ✓ (authored)                        | ✓ (authored, mirrored from slow-path) |
| Prim types (`IsA(usdrt.UsdGeom.Xform)`)        | ✓                                   | ✓                                     |
| Authored attribute values                      | ✓                                   | ✓                                     |
| `xformOp:transform` of a physics-driven link   | **authored value** (often identity) | **live physics value**                |
| Joint positions on a robot articulation        | **authored value** (usually 0)      | **live value** from PhysX             |
| Velocities                                     | 0 (no simulation yet)               | live from PhysX                       |
| `/ExternalSimulationTime`                      | 0 (seeded by `onAttach`)            | live from sim                         |
| A prim you just authored on pxr this tick      | may be stale (one tick lag)         | live                                  |

So the Fabric stage is _usable_ before play — you can walk the tree, read
authored attribute values, and find frames by name (which is exactly what
`resolve_parent_prim_path` does). Physics-driven data is just at its
authored (often zero) initial state.

### 1.3 The slow-path: the always-there substrate

The §1 mental model introduces two stages. A natural follow-up: is the
slow-path (`pxr.Usd.Stage`) _always_ available, or can FSD turn it off?
The answer is _always available_ — Fabric is a read-side mirror, not a
replacement. This sub-section grounds the reader in the substrate that
everything else depends on.

#### You can always grab a slow-path stage

After `omni.usd.get_context().open_stage(path)` (or `SimulationApp` opens
its default stage), the `pxr.Usd.Stage` is in memory for the whole
session, regardless of FSD:

```python
# both of these work in any Isaac Sim app, FSD on or off
import omni.usd
slow = omni.usd.get_context().get_stage()                  # always

import isaacsim.core.experimental.utils.stage as stage_utils
slow = stage_utils.get_current_stage(backend="usd")        # always
```

You can walk the prim tree, call `DefinePrim`, edit layer stacks, apply
schemas, set attributes — all the full USD authoring surface — at any
time. None of that depends on Fabric being loaded.

#### What the configuration knobs actually control

The settings in §2.1 and §2.2 don't turn the slow-path on or off. They
change _what gets written to it_ and _what reads from it_:

| Setting                                    | What it actually does to the slow-path                                                                                                          |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `useFabricSceneDelegate`                   | Tells Hydra (the renderer) to read from Fabric instead. **Does not** affect the slow-path's existence.                                          |
| `physics.fabricEnabled`                    | Tells PhysX to write per-step state to Fabric. **Does not** turn the slow-path off.                                                             |
| `physics.fabricUpdateTransformations` etc. | Which per-step fields go to Fabric. Slow-path presence is unaffected.                                                                           |
| `/physics/updateToUsd = false` (this repo) | Suppresses the _slow-path writeback_ of per-step physics state. The slow-path is still there, it just doesn't get a copy of live physics state. |
| `/physics/updateToUsd = true`              | Mirrors live physics state to the slow-path. Even then, the slow-path was already in memory.                                                    |

FSD is a separate consumer that happens to share data with the slow-path
via a change-notification handler. The slow-path doesn't depend on FSD
to exist.

#### The "live vs. authored" subtlety

This is the one place where "always available" needs a footnote. The
slow-path is always reachable, but **what you read off it depends on
what's being written to it**:

- **Authored values** (the values in the .usd file on disk, or what
  you just wrote via `pxr.Usd.Attribute.Set`): always visible on the
  slow-path.
- **Live physics-driven values** (what PhysX wrote to a body _this
  step_): visible on the slow-path **only if** `/physics/updateToUsd
= true`. This repo turns that off (§2.2) and the experience file is
  set accordingly.

So the practical rule from §1:

> A `pxr.Usd.Prim` and a `usdrt.Usd.Prim` at the same path refer to the
> same logical prim, but the **physics-driven values** you read off them
> may differ. The slow-path is the substrate; Fabric is the live-state
> mirror.

Concretely, with this repo's defaults:

```python
slow = stage_utils.get_current_stage(backend="usd")
fast = stage_utils.get_current_stage(backend="usdrt")

# Both stages have the prim — always.
slow_prim = slow.GetPrimAtPath("/World/cube_0")
fast_prim = fast.GetPrimAtPath("/World/cube_0")

# But the transform read may differ:
slow_xform = slow_prim.GetAttribute("xformOp:transform").Get()    # authored value
fast_xform = fast_prim.GetAttribute("xformOp:transform").Get()    # live physics value
```

This isn't a "slow-path is sometimes disabled" issue. The slow-path is
always in memory and always queryable. It's just that the question "is
this value live?" is answered "no" for slow-path reads when Fabric is
on and `updateToUsd` is off.

#### The lifecycle, in three points

1. **The slow-path is created when a stage is opened.** Before any
   `open_stage` call, there's no stage in either backend. After it,
   the slow-path is in memory for the whole session.
2. **The slow-path is the only path you can author through.** The §4
   capability matrix shows every authoring row (`DefinePrim`, layer
   stack, schema registration, …) is `✓` for slow-path and `✗` for
   Fabric. That asymmetry is the reason the slow-path is always
   there — Fabric is a read-side optimization, not an authoring
   replacement.
3. **You can switch stages, but each new stage also gets a slow-path.**
   `omni.usd.get_context().open_stage(other_path)` releases the old
   `pxr.Usd.Stage` and builds a new one. The Fabric mirror is rebuilt
   to match. The slow-path is never _absent_ during a session — only
   _replaced_ by a new one.

#### What to take away

- `pxr.Usd` is **always** available, regardless of FSD. You can always
  use the full USD authoring surface.
- `usdrt.Usd` is **available when the Fabric extensions are loaded**
  (typically: when the .kit experience file has
  `useFabricSceneDelegate = true` and the FSD dependencies are enabled).
- The two stages are **parallel in-memory views** of the same scene —
  neither one _replaces_ the other. Both can be held at the same time
  via `stage_utils.get_current_stage(backend=...)`.
- The slow-path is what holds **authored content forever**. Fabric
  holds **live physics state for the duration of the session** (it's
  re-seeded from slow-path on every stage-open).

---

## 2. Configuration

### 2.1 Experience file settings

[`example_experience.kit`](../../backend_comparison/example_experience.kit) is a **reference
example** of a Fabric/USDRT-enabled experience — the complete set of
knobs, wired together, ready to copy into your own `.kit`. It is **not**
loaded by the benchmark (see §2.1.1); the harness uses the stock Isaac
Sim experience and toggles FSD at runtime. Knobs set by the example:

| Section         | Key                                   | Value   | What it does                                                                                                                              |
| --------------- | ------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `[settings.app]`| `useFabricSceneDelegate`              | `true`  | Tells the renderer (Hydra / RTX) to read scene description from Fabric. Does **not** affect your Python code.                             |
| `[dependencies]`| `omni.physics.stageupdate`            | loaded  | Provides the per-stage Fabric update callbacks.                                                                                           |
| `[dependencies]`| `omni.physx.fabric`                   | loaded  | The PhysX ↔ Fabric bridge. Required for physics-driven updates on Fabric.                                                                 |
| `[dependencies]`| `usdrt.scenegraph`                    | loaded  | The Python `usdrt` module (the Fabric scenegraph bindings).                                                                               |
| `[settings]`    | `physics.fabricEnabled`               | `true`  | Master switch. Loads the fabric bridge and tells PhysX to write to Fabric.                                                                |
| `[settings]`    | `physics.fabricUpdateTransformations` | `true`  | Per-step body / link transforms go to Fabric.                                                                                             |
| `[settings]`    | `physics.fabricUpdateVelocities`      | `true`  | Per-step body / link velocities go to Fabric.                                                                                             |
| `[settings]`    | `physics.fabricUpdateJointStates`     | `true`  | Per-step articulation joint positions / velocities go to Fabric.                                                                          |
| `[settings]`    | `physics.fabricUpdateForceSensors`    | `true`  | Per-step force-sensor outputs go to Fabric.                                                                                               |
| `[settings]`    | `omnigraph.updateToUsd`               | `false` | OmniGraph nodes do not write per-tick attribute updates back to the slow-path stage. Performance optimization. Enable only for debugging. |
| `[settings]`    | `physics.updateForceSensorsToUsd`     | `false` | Force sensor data is not mirrored to pxr.Usd. (The Fabric write is still on; this is the slow-path writeback.)                            |
| `[settings]`    | `omnihydra.useFastSceneDelegate`      | `true`  | Hydra uses the fast scene delegate (Fabric-based). Pairs with `useFabricSceneDelegate = true`.                                            |
| `[settings]`    | `omnihydra.useSceneGraphInstancing`   | `true`  | Hydra uses Fabric instancing for repeated prims (clones, references).                                                                     |

#### 2.1.1 How the benchmark harness actually drives Fabric

The example experience is a documented baseline, not the active config.
The benchmark loads the **stock** Isaac Sim 6.0 experience and toggles
FSD at runtime if `--enable-fsd` is passed:

```python
# filepath: backend_comparison/benchmark_backend.py
settings: dict = {}
if args.enable_fsd:
    settings["/app/useFabricSceneDelegate"] = True

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "settings": settings})
```

`run_comparison.sh` passes `--enable-fsd` automatically for `usdrt`
and `tensor` because those backends require FSD; the `usd`
backend runs without it. None of the other knobs in §2.1's table are
set at runtime — they would only matter if the benchmark loaded a
custom experience, which it doesn't (see the `experience=` arg on
`SimulationApp` for how to do that explicitly).

If you want the harness to consume `example_experience.kit` directly,
add `"experience": str(Path(__file__).parent / "example_experience.kit")`
to the `SimulationApp` kwargs. The doc doesn't show that wiring today
because the harness is intentionally exercising only the runtime FSD
toggle path — using a custom experience would also enable the rest of
§2.1's knobs, which would obscure the cross-backend differences.

### 2.2 Carb settings that `SimulationManager.enable_fabric` flips

When you call `SimulationManager.enable_fabric(True)` (see §3.1), it sets the
following carb settings in addition to enabling the extension:

| Carb key                           | Value when fabric is on | What it means                                                            |
| ---------------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `/physics/updateToUsd`             | `false`                 | PhysX does not write body / link transforms back to the slow-path stage. |
| `/physics/updateParticlesToUsd`    | `false`                 | Particle systems do not mirror to pxr.Usd.                               |
| `/physics/updateVelocitiesToUsd`   | `false`                 | Velocities do not mirror to pxr.Usd.                                     |
| `/physics/updateForceSensorsToUsd` | `false`                 | Force-sensor data does not mirror to pxr.Usd.                            |

These are the _slow-path_ writebacks. Fabric handles the corresponding
_fast-path_ writes via `physics.fabricUpdate*` from §2.1.

### 2.3 Net effect on behavior

When everything above is on (which is the case in `example_experience.kit`,
the reference example):

- The renderer reads from Fabric. Transforms in the viewport are the live,
  physics-driven values.
- PhysX writes per-step state (transforms, velocities, joints, force sensors)
  to Fabric only. The slow-path stage is **not** updated each step.
- Authoring a prim or attribute on pxr propagates to Fabric via the change
  notification handler (one `kit` tick of lag).
- A physics-driven transform read on pxr will return the **authored** value
  (the one in the .usd file), not the live one. This is the most common
  surprise.

### 2.5 What Fabric Scene Delegate actually is

The carb settings in §2.1 and the `use_backend` check in §3.7 all hook into
one component: **Fabric Scene Delegate** (FSD). This section explains what
it is, what it does at runtime, and why every Fabric-related setting in the
experience file exists. The rest of the doc assumes you've read this.

#### 2.5.1 Scene delegate, in USD terms

USD is a schema library — `UsdStage` knows about prims, attributes, and
composition, but it has no opinion on _how to render_ or _how to simulate_
a prim. The actual work of "given this prim, what lights/cameras/meshes/
draw calls does it contribute?" is delegated to a **scene delegate**, a
C++ interface that a consumer (Hydra, PhysX, the camera stack, …)
implements.

The classic, slow-path scene delegate is `UsdImagingGLSceneDelegate` (for
Hydra) and the matching Python/C++ walkers PhysX uses to find rigid
bodies. They work by **walking the USD stage** on the consumer's thread,
asking each prim "are you a mesh? what's your transform? what's your
material?" every frame. Correct, but expensive at scale — every read is
a virtual call into Python, a prim lookup, an attribute read.

#### 2.5.2 What Fabric Scene Delegate is

Fabric Scene Delegate is a **C++ struct-of-arrays re-implementation** of
the same scene delegate contract, but with a completely different data
layout:

|                | Standard (slow-path) scene delegate                 | Fabric Scene Delegate                                        |
| -------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| Storage        | Walks `pxr.Usd.Stage` on demand                     | Pre-mirrors the scene into a C++ SoA                         |
| Per-frame read | Virtual call into Python, attr lookup, attr `Get()` | Pointer / index into a contiguous buffer                     |
| GPU binding    | ✗ (USD data isn't Warp-friendly)                    | ✓ (`wp.fabricarray` zero-copy)                               |
| Where it lives | `omni.usd` / `omni.hydra.scenegraph.delegate.usd`   | `omni.hydra.fabric.scenegraph_delegate` + `usdrt.scenegraph` |
| Selected by    | default                                             | `[settings.app] useFabricSceneDelegate = true`               |

The Fabric runtime keeps the SoA in sync with the slow-path
`pxr.Usd.Stage` via a change-notification handler: authoring on pxr
propagates to Fabric on the next `kit` tick. PhysX then writes its
per-step state (transforms, velocities, joint states) **directly** to
the Fabric SoA, bypassing the slow-path. The renderer, the physics view,
and any consumer using FSD then read the live state from one place.

#### 2.5.3 Why the prim wrappers need it

The `usdrt` and `fabric` paths in `XformPrim.set_world_poses`,
`RigidPrim.get_world_poses`, etc. all go through one method on the
wrapper:

```python
fabric_hierarchy = self._get_fabric_hierarchy()         # fabric API
matrix = fabric_hierarchy.get_world_xform(usdrt.Sdf.Path(self.paths[index]))
```

`_get_fabric_hierarchy()` returns a handle into Fabric's scene hierarchy.
**That handle only exists if Fabric Scene Delegate has been brought up at
app launch.** If FSD isn't enabled, there is no Fabric scene graph to
attach to, and the call would return null or crash. That's exactly what
the runtime check in `use_backend` is protecting against (see §3.7):

```python
# isaacsim/core/experimental/utils/impl/backend.py
if backend in ["usdrt", "fabric"] and not _fsd_enabled:
    raise RuntimeError(
        "'usdrt' and 'fabric' backends require Fabric Scene Delegate (FSD) "
        "to be enabled. Enable FSD in .kit experience settings "
        "('app.useFabricSceneDelegate = true') to use them."
    )
```

The dependency chain is:

```
useFabricSceneDelegate = true   (in .kit, read at app launch)
        ↓
Fabric runtime is constructed; FSD is the default scene delegate
        ↓
Fabric scene graph is built when a stage is opened
        ↓
fabric_hierarchy handle is reachable
        ↓
XformPrim / RigidPrim usdrt paths work
        ↓
Python caller can `use_backend("usdrt")` without error
```

If you skip the `use_backend` call entirely, the wrappers never ask for
the Fabric handle and the missing FSD never bites you — you just stay on
`usd` (or `tensor`, for `RigidPrim`). This is why the runtime check is
"fail loudly" rather than "fail silently" — it surfaces a configuration
problem only when the user actually tries to use the FSD-dependent path.

#### 2.5.4 Why it matters even when you don't ask for it

Three things only work because FSD exists, regardless of which Python
backend you pick:

1. **The renderer's hot path.** When `useFabricSceneDelegate = true` is
   on, Hydra reads scene description from Fabric, not from `pxr.Usd`.
   That's what makes the viewport fast enough for sensor-heavy scenes.
   The viewport performance has nothing to do with your Python code;
   it's a separate consumer that benefits from the same SoA.
2. **PhysX writing live state somewhere.** PhysX writes per-step
   transforms/velocities/joint-states to Fabric (via
   `physics.fabricUpdate*`) and the viewport reads from there. Without
   Fabric, the per-step state has nowhere to live except the slow-path
   stage, and the slow-path writeback (`/physics/updateToUsd`) is by
   design off in this repo's experience file. So without FSD: no live
   state anywhere.
3. **`wp.fabricarray` zero-copy GPU access.** The C++/CUDA side of
   Warp can `wp.fabricarray.bind(...)` directly to Fabric buffers.
   This is the substrate for GPU-resident RL training, batched
   simulation, and sensor pipelines that produce tensors on the GPU.
   None of it works without FSD.

#### 2.5.5 Why it's a launch-time setting

Fabric is a process-wide C++ runtime with shared state (the SoA, the
change-notification handler, the GPU mappings). Bringing it up requires
loading the right extensions (`omni.hydra.fabric`, `usdrt.scenegraph`,
`omni.physx.fabric`) and initializing a global **before any stage is
opened**. Hence the carb settings are read from the experience file at
app launch, and the `use_backend` runtime check (§3.7) is read-only — it
can refuse to use the path, but it can't enable FSD on the fly. Trying
to flip `useFabricSceneDelegate` mid-session has no effect; the decision
is locked in at startup.

For `use_backend("usdrt")` to work, you need at minimum:

- `useFabricSceneDelegate = true` (or any non-default scene delegate
  that constructs a Fabric scene graph)
- `omni.physx.fabric` and `usdrt.scenegraph` loaded as dependencies
- (Strictly for live state, not for the prim wrapper to compile)
  `physics.fabricEnabled = true`

#### 2.5.6 How to verify FSD is on in a running app

```python
import carb
import isaacsim.core.experimental.utils.stage as stage_utils

settings = carb.settings.get_settings()
print("useFabricSceneDelegate:", settings.get("/app/useFabricSceneDelegate"))
print("physics.fabricEnabled:  ", settings.get("/physics/fabricEnabled"))
# optional: confirm the Fabric stage actually has data
fast = stage_utils.get_current_stage(backend="usdrt")
print("fabric stage populated:  ", fast.GetPrimAtPath("/World").IsValid())
```

If `useFabricSceneDelegate` is `None` or `False`, the hot paths in the
experimental prim wrappers will fail loudly the first time you open
`use_backend("usdrt")`, with the exact `RuntimeError` documented in §3.7.

#### 2.5.7 TL;DR

- **Scene delegate** = USD's way of letting consumers (renderer,
  physics, …) walk a prim and ask "what do you do?"
- **Fabric Scene Delegate** = a C++ SoA-backed re-implementation of
  that contract, used because it's faster (no Python), GPU-bindable
  (`wp.fabricarray`), and the only place PhysX writes live state in
  this repo's setup.
- **Why the prim wrappers need it**: their `usdrt`/`fabric` paths call
  `fabric_hierarchy.get_world_xform(...)`, which requires a Fabric
  handle that only exists when FSD is up.
- **Why it's a launch-time setting**: Fabric is a process-wide C++
  runtime; it has to be initialized before any stage is opened, hence
  the experience-file carb setting rather than a runtime toggle.

---

## 3. Python API surface

### 3.1 `SimulationManager` (the public entry point)

```python
# isaacsim.core.simulation_manager
@classmethod
def enable_fabric(cls, enable: bool) -> None:
    """Enable or disable physics fabric integration and associated settings.

    .. note::
        This only applies to PhysX. For other physics engines (like Newton),
        this is a no-op since they handle fabric/USD updates differently.
    """
    if cls._engine != "physx":
        return
    app_utils.enable_extension("omni.physx.fabric", enabled=enable)
    cls._physx_fabric_interface = omni.physxfabric.get_physx_fabric_interface() if enable else None
    cls._carb_settings.set_bool("/physics/updateToUsd", not enable)
    cls._carb_settings.set_bool("/physics/updateParticlesToUsd", not enable)
    cls._carb_settings.set_bool("/physics/updateVelocitiesToUsd", not enable)
    cls._carb_settings.set_bool("/physics/updateForceSensorsToUsd", not enable)


@classmethod
def is_fabric_enabled(cls) -> bool:
    """True if `omni.physx.fabric` is enabled."""
    return app_utils.is_extension_enabled("omni.physx.fabric")


@classmethod
def enable_fabric_usd_notice_handler(cls, stage_id: int, enable: bool) -> None:
    """Toggle Fabric's USD notice handler for the given stage.

    Disable only if you want to suppress prim-creation / prim-deletion
    notifications on the Fabric side.
    """
    cls._simulation_manager_interface.enable_fabric_usd_notice_handler(stage_id, enable)
```

What `enable_fabric` does that the `.kit` file alone **doesn't**:

1. Captures the `omni.physxfabric` C++ interface so you can drive Fabric
   updates from Python.
2. Flips the `/physics/updateToUsd*` carb settings to `false` — the explicit
   decision to stop mirroring physics state back to the slow-path stage.

The extension itself is loaded at app startup by the experience file, but
the C++ interface handle is only acquired when `enable_fabric(True)` runs.

### 3.3 The `stage_utils.get_current_stage` dispatcher

This is the official helper for grabbing either backend's stage:

```python
# isaacsim.core.experimental.utils.stage
def get_current_stage(*, backend: str | None = None) -> Usd.Stage | usdrt.Usd.Stage:
    """Get the stage set in the context manager or the default stage.

    Backends: ``"usd"``, ``"usdrt"``, ``"fabric"``.
    """
    if backend is None:
        backend = backend_utils.get_current_backend(["usd", "usdrt", "fabric"])
    stage = getattr(_context, "stage", omni.usd.get_context().get_stage())
    if backend in ["usdrt", "fabric"]:
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(stage).ToLongInt()
        if stage_id < 0:
            stage_id = stage_cache.Insert(stage).ToLongInt()
        return usdrt.Usd.Stage.Attach(stage_id)
    return stage
```

Usages:

```python
import isaacsim.core.experimental.utils.stage as stage_utils

slow = stage_utils.get_current_stage(backend="usd")    # pxr.Usd.Stage
fast = stage_utils.get_current_stage(backend="usdrt")  # usdrt.Usd.Stage
fast2 = stage_utils.get_current_stage(backend="fabric")  # same as "usdrt"
```

There's also a `backend_utils.use_backend(...)` context manager if you want
to flip the _default_ backend for a block of code. It is the **implicit**
counterpart of the explicit `backend=` argument — see §3.7 for the
propagation rules, the per-method defaults, the FSD check, and the
`RigidPrim` ↔ `tensor` surprise.

### 3.4 Low-level: `usdrt.Usd.Stage.Attach`

If you don't want to pull in `isaacsim.core.experimental.utils`, you can
attach directly:

```python
import usdrt
import omni.usd
from pxr import UsdUtils

slow_stage = omni.usd.get_context().get_stage()
stage_id = UsdUtils.StageCache.Get().GetId(slow_stage).ToLongInt()
fabric_stage = usdrt.Usd.Stage.Attach(stage_id)
```

This is the pattern shown in the official test at
`tests/test_simulation_manager.py:1042` for reading the simulation time
attribute on `/ExternalSimulationTime`.

### 3.5 Verifying Fabric is populated

To confirm Fabric is populated in your scene (and catch the brief pre-sync
window right after `open_stage`), drop this into the runner right after
`build_scene(...)` and before `sim_ctx.reset()`:

```python
import isaacsim.core.experimental.utils.stage as stage_utils

fabric = stage_utils.get_current_stage(backend="usdrt")
slow = stage_utils.get_current_stage(backend="usd")

# Both should have the same authored prim hierarchy before play.
for path in ["/World", "/World/Camera", scene_usd_path]:
    fp = fabric.GetPrimAtPath(path) if path.startswith("/") else None
    sp = slow.GetPrimAtPath(path) if path.startswith("/") else None
    print(f"{path}: fabric={fp and fp.IsValid()}, slow={sp and sp.IsValid()}")
```

If both come back `True` for `/World` and your scene prims, Fabric is
populated. If Fabric returns `False` while slow is `True`, you caught the
brief pre-sync window (a few ms after stage-open) — wait one tick with
`await omni.kit.app.get_app().next_update_async()` and re-check.

### 3.6 Common gotcha: authoring on pxr before the first Fabric tick

There's a sharp edge right after stage-open (and any time you author + read
in the same tight sequence):

```python
from pxr import UsdGeom
import omni.kit.app

slow_stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(slow_stage, "/World/MyNewPrim")
# ...immediately:
fabric_prim = fabric.GetPrimAtPath("/World/MyNewPrim")
fabric_prim.IsValid()  # might be False on the first attempt
```

The Fabric scene delegate hasn't yet processed the authoring notification.
One tick fixes it:

```python
await omni.kit.app.get_app().next_update_async()
fabric_prim = fabric.GetPrimAtPath("/World/MyNewPrim")
fabric_prim.IsValid()  # True
```

For sensor setup this is irrelevant — sensors are created _after_ the
initial sync is well past, and you're reading authored values that were
on disk long before the app started. But for code that authors and reads
in the same tight sequence, plan to insert a sync between write and
read. Two equivalent forms:

```python
# Async (Kit scripts, Jupyter, async run-loops):
import omni.kit.app
await omni.kit.app.get_app().next_update_async()

# Sync (this repo's benchmark, headless harness):
import isaacsim.core.experimental.utils.app as app_utils  # noqa
app_utils.play()                                          # or
simulation_app.update()                                   # pump once
```

The async form returns once a kit tick has elapsed; the sync form blocks
the calling thread until the run-loop has advanced at least one tick.
Either is sufficient to let the Fabric scene delegate consume the
authoring notification.

### 3.7 `backend_utils.use_backend` — the implicit default propagator

Most of the experimental API has an explicit `backend="..."` argument, but
threading it through every call is awkward. The
`isaacsim.core.experimental.utils.backend.use_backend` context manager is
the supported escape hatch: a thread-local default that _any_ subsequent
`backend=None` API consults on entry.

```python
import isaacsim.core.experimental.utils.backend as backend_utils
from isaacsim.core.experimental.prims import RigidPrim, XformPrim

with backend_utils.use_backend("usdrt", raise_on_unsupported=True):
    rigid = RigidPrim(paths=["/World/cube_0"])
    rigid.set_world_poses(positions=...)   # routes through USDRT
    rigid.get_world_poses()                # routes through USDRT
# back to default outside the block
```

#### What `use_backend` is and isn't

`use_backend` is **not** a "switch the active stage" — §1 already covers
that there is no global current stage. What it actually does is plant a
thread-local default that _any_ `backend=None` API consults on entry.
APIs that take an explicit `backend=` argument also honor it: pass
`backend="usd"` explicitly and that wins; pass nothing and the context
is consulted.

APIs that consult the context:

| API                                                                                   | What `use_backend("usdrt")` changes about it                                                              |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `stage_utils.get_current_stage(backend=None)`                                         | Returns a `usdrt.Usd.Stage` (otherwise returns `pxr.Usd.Stage`)                                           |
| `XformPrim.get_world_poses` / `set_world_poses`                                       | Routes through `_get_fabric_hierarchy().get_world_xform(...)` instead of `pxr.Usd.Attribute.Get` / `.Set` |
| `RigidPrim.set_world_poses` / `get_world_poses` / `set_velocities` / `get_velocities` | Routes through Fabric's rigid-body view                                                                   |
| `RigidPrim.set_linear_velocities` / `get_linear_velocities`                           | Routes through Fabric's rigid-body view                                                                   |
| `RigidPrim.set_angular_velocities` / `get_angular_velocities`                         | Routes through Fabric's rigid-body view                                                                   |
| `Articulation` (multi-backend methods)                                                | Same — routes through the Fabric hierarchy / FSD view                                                     |
| `GeomPrim.*` (every method)                                                           | **Unaffected** — every method is tagged `Backends: usd` in its docstring                                  |

#### The per-method default backend

If you don't open a `use_backend(...)` context at all, each method falls
back to a _per-method_ default. The default is the first item of the
method's supported-backends list, and **it is not always `usd`**:

| Wrapper     | Method's supported list                | Default if no `use_backend` is open                 |
| ----------- | -------------------------------------- | --------------------------------------------------- |
| `XformPrim` | `["usd", "usdrt", "fabric"]`           | `usd`                                               |
| `RigidPrim` | `["tensor", "usd", "usdrt", "fabric"]` | **`tensor`** (if physics view is ready) — see below |
| `GeomPrim`  | every method is `Backends: usd`        | `usd`                                               |

> **Surprise: the `RigidPrim` default is `tensor`, not `usd`.**
> `tensor` is the first item of the supported list, and the dispatch reads
> it positionally. If the physics view is ready (it almost always is by
> the time you start calling prim methods), your call goes through the
> tensor path without you ever opening a context. See the next section
> for the implications.

To verify, look for the `Backends:` line in the method's docstring and
treat the first item as the default.

#### The `tensor` backend and its silent fallback

`RigidPrim` has a guard in `_check_for_tensor_backend`:

```python
if backend == "tensor" and not self.is_physics_tensor_entity_valid():
    if backend_utils.is_backend_set():
        if backend_utils.should_raise_on_fallback():
            raise RuntimeError(
                f"Physics tensor entity is not valid. Fallback set to 'usd' backend."
            )
        carb.log_warn(
            f"Physics tensor entity is not valid for use with 'tensor' backend. "
            f"Falling back to 'usd' backend."
        )
    return "usd"   # silent when no context manager is active
```

So the effective behavior is:

| Situation                                                                  | Effective backend for `RigidPrim`                  |
| -------------------------------------------------------------------------- | -------------------------------------------------- |
| No `use_backend` set, physics view ready                                   | `tensor` (teleport semantics on `set_world_poses`) |
| No `use_backend` set, physics view NOT ready yet                           | `usd` (silent, no warning)                         |
| `use_backend("tensor")`, physics view NOT ready, default flags             | `usd` + **carb warning**                           |
| `use_backend("tensor")` + `raise_on_fallback=True`, physics view NOT ready | raises `RuntimeError`                              |

The **teleport semantic** matters: with `tensor`, `RigidPrim.set_world_poses`
writes directly into the PhysX tensor view, bypassing PhysX integration.
On a stack of contacting bodies this can produce unrealistic motion. If
you're benchmarking or driving a real physics workload, prefer the
explicit `use_backend("usdrt")` (or `"fabric"`) so the integrator still
runs.

#### The FSD check fires only inside `use_backend`

The Fabric Scene Delegate requirement is enforced _only_ inside the
context manager:

```python
# isaacsim/core/experimental/utils/impl/backend.py
if backend in ["usdrt", "fabric"] and not _fsd_enabled:
    raise RuntimeError(
        "'usdrt' and 'fabric' backends require Fabric Scene Delegate (FSD) "
        "to be enabled. Enable FSD in .kit experience settings "
        "('app.useFabricSceneDelegate = true') to use them."
    )
```

If you never call `use_backend("usdrt")`, no FSD check fires. You can run
with FSD disabled and your code will still work — just slowly, on USD.
The catch is that the _default_ path for `RigidPrim` is `tensor`, which
also relies on the physics view being valid; that's a different check
(see above).

#### Thread-local scope

`_context` is a `threading.local()`, so each thread has its own backend.
If you spawn a worker thread and call `rigid.set_world_poses(...)` from
it without opening `use_backend` in that thread, the call uses the
default (`tensor` for `RigidPrim`, `usd` for `XformPrim`). For
multi-threaded code, open `use_backend` in each thread that does prim
work.

#### SimState is a separate channel

`/isaacsim/articulation/simStateMode` (read by `get_simstate_mode()`) is
orthogonal. It only affects whether the SimState view gets a copy of the
data in parallel; it does not change which of `usd` / `usdrt` / `tensor`
the prim dispatches to. Setting `simStateMode = "mirror"` does **not**
change the dispatch — it controls whether SimState is also written.

#### When to use which form

- **Explicit `backend="usdrt"`** when a single call needs to cross to a
  different representation than the surrounding code uses, or when
  writing a library/helper that you don't want to be affected by the
  caller's context.
- **Implicit `use_backend("usdrt")`** when you're writing a block of code
  that touches many APIs in the same representation — the prim wrappers,
  `get_current_stage`, materials, etc. — and you want them all to agree.

For most call sites in this codebase the explicit form on
`stage_utils.get_current_stage` is enough; the implicit form is only
worth the cost when you start calling experimental prim methods inside
the block.

---

## 4. Capability matrix

| Task                                                    | pxr.Usd (slow)             | usdrt.Usd (Fabric)                  |
| ------------------------------------------------------- | -------------------------- | ----------------------------------- |
| Open / edit layer stack                                 | ✓                          | ✗                                   |
| `UsdGeom.Xform.Define(stage, path)` (author a prim)     | ✓                          | ✗                                   |
| `Stage.DefinePrim`, attribute authoring                 | ✓                          | ✗                                   |
| `Stage.OverridePrim`, `Stage.DeclarePrim`               | ✓                          | ✗                                   |
| `Sdf` layer manipulation (sublayer, reference, payload) | ✓                          | ✗                                   |
| Schema application / type registration                  | ✓                          | ✗                                   |
| `Usd.PrimRange(root)` iteration                         | ✓                          | ✗ (use `GetAllDescendants` instead) |
| `prim.GetAllDescendants()`                              | ✓                          | ✓                                   |
| `prim.GetName()`                                        | ✓                          | ✓                                   |
| `prim.GetPath()` → string                               | ✓ (`.pathString`)          | ✓ (`str(prim.GetPrimPath())`)       |
| `prim.IsA(Xform)`                                       | ✓ (pxr.UsdGeom.Xform)      | ✓ (usdrt.UsdGeom.Xform)             |
| Read prims by path                                      | ✓                          | ✓ (faster)                          |
| Read attribute values                                   | ✓                          | ✓ (much faster, SoA)                |
| **Read live physics-driven transforms**                 | **stale (authored value)** | **✓ fresh**                         |
| **Read live joint states, velocities, force sensors**   | **stale**                  | **✓ fresh**                         |
| Bind to `wp.fabricarray` (zero-copy GPU access)         | ✗                          | ✓                                   |
| Receive change notifications                            | ✓ (Usd.Notice)             | ✓ (Fabric notice handler, separate) |

### 4.1 The rule of thumb

- **pxr.Usd** for **authoring and one-off queries** that need the full USD
  surface (layer stack, schemas, composition, animation curves, asset
  references).
- **usdrt** for **per-frame reads** of attributes and transforms, especially
  when those values are physics-driven. This is the hot path.
- **usdrt** is the only way to bind to Warp `wp.fabricarray` (zero-copy GPU
  access from kernels).

### 4.2 Why you can't use `usdrt.UsdGeom.Xform` in pxr code (or vice versa)

The types are **100% non-interchangeable**. They live in different modules
(`pxr.UsdGeom` vs `usdrt.UsdGeom`) and are different Python classes. Mixing
them produces hard `TypeError`s, never silent corruption. See §5.

---

## 5. Mixing `usd` and `usdrt` APIs

### 5.1 What fails

| Mixing attempt                                     | Result                                                                                                                                          |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Pass `pxr.Usd.Prim` to a `usdrt.Usd.Stage` method  | `TypeError: incompatible prim` (or similar)                                                                                                     |
| `usdrt_prim.GetAllDescendants()` returns           | Items are `usdrt.Usd.Prim`; downstream `IsA(pxr.UsdGeom.Xform)` returns `False` because the prim isn't actually a pxr prim                      |
| `pxr.Usd.PrimRange(usdrt_prim)`                    | `TypeError: Usd.PrimRange expected pxr.Usd.Prim, got usdrt.Usd.Prim`                                                                            |
| `prim.GetPath()` from usdrt                        | Returns a usdrt path object; `str(...)` gives `/foo/bar`, but you can't pass it to `UsdGeom.Xform.Define(stage, path)` (which wants `Sdf.Path`) |
| `pxr_prim.GetName() == usdrt_prim.GetName()`       | Works (string compare)                                                                                                                          |
| `pxr_prim.GetPath() == usdrt_prim.GetPath()`       | Fails — different `Sdf.Path` types, even if the strings match                                                                                   |
| Read same attribute from both stages in same frame | Returns same value (eventually consistent)                                                                                                      |

### 5.2 What works

- **Path strings** (`"/World/Camera"`) are interchangeable. Both backends
  consume them. A function that takes a string path and returns a string
  path is backend-agnostic.
- **Authoring on pxr → reading on usdrt** (or vice versa) is eventually
  consistent. The Fabric change-notification handler propagates writes
  between stages, typically within one `kit` tick.

### 5.3 The lag hazard

After **authoring a prim on pxr**, reading it on usdrt in the same tick can
return stale results. Fix with a kit tick between the two:

```python
from pxr import UsdGeom
import omni.kit.app

UsdGeom.Xform.Define(slow_stage, "/World/foo")
await omni.kit.app.get_app().next_update_async()  # let Fabric sync
fabric_prim = fabric_stage.GetPrimAtPath("/World/foo")
```

For physics-driven values, there is no equivalent sync — the slow-path
simply does not receive the per-step updates when `physics.updateToUsd = false`.
Read from Fabric for live values; read from pxr only for the authored
fallback.

---

## 6. C++-level notes (for context)

`usdrt.Usd.Stage.Attach(stage_id)` does not construct a new scene. It
attaches a typed view over the same data Fabric already holds in memory and
gives you a Python wrapper around it. Both stages are backed by the same
`SdfLayer` tree on disk; Fabric is just an additional in-memory mirror of
the resolved prim hierarchy, optimized for read access.

- `pxr.Usd.Prim` and `usdrt.Usd.Prim` for the same path: different Python
  types, different memory layouts, different schema classes — but the same
  logical prim.
- Path representation differs: pxr uses `Sdf.Path` (string-like, supports
  composition operations); usdrt uses an index-based handle that stringifies
  to `/foo/bar`. You cannot pass one to the other side's API.

---

## 8. Recommendations for this codebase

1. **Verify the experience file is correct.** Run
   `grep -E 'fabric|usdrt' example_experience.kit` (or your own custom
   experience) to confirm the relevant knobs are set as documented in
   §2.1. The reference example in this repo
   ([`example_experience.kit`](../../backend_comparison/example_experience.kit))
   shows the recommended configuration — copy it into your own `.kit`
   when authoring a custom experience.

2. **Use usdrt where it actually pays off:**
   - Per-frame physics state readback (transforms, joint states, velocities, force sensors, contact reporting).
   - Warp kernels reading `wp.fabricarray`.

3. **If you ever need both worlds in one module**, alias the imports
   explicitly:

   ```python
   from pxr import UsdGeom as UsdGeomPx
   import usdrt.UsdGeom as UsdGeomRt
   ```

   The `pxr.UsdGeom.Xform` and `usdrt.UsdGeom.Xform` symbols look identical
   but are different classes.

4. **Don't read physics-driven transforms on pxr.** If you need live joint
   state per frame, read it from the Fabric stage:

   ```python
   fabric_stage = stage_utils.get_current_stage(backend="usdrt")
   prim = fabric_stage.GetPrimAtPath("/World/Robot/base_link")
   xform = prim.GetAttribute("xformOp:transform").Get()
   ```

   Reading the same path on pxr returns the authored value, not the
   physics-driven one.

5. **Use the dispatcher.** Prefer
   `stage_utils.get_current_stage(backend=...)` over hand-rolling
   `omni.usd.get_context().get_stage()` + `usdrt.Usd.Stage.Attach(...)`.
   It's the supported entry point and handles stage-id caching.

6. **For authoring, stay on pxr.** `UsdGeom.Xform.Define`,
   `Stage.DefinePrim`, attribute creation, layer editing — all
   pxr-only. Author on pxr, let Fabric sync.

7. **Mind the post-open sync window.** Right after `open_stage` (and any
   time you author + read in the same tight sequence), Fabric may be a
   few ms behind the slow-path. If a query for a prim you just authored
   on pxr returns `False`, insert one
   `await omni.kit.app.get_app().next_update_async()` and retry. This
   window is invisible for sensor setup (prims were on disk long before
   the app started) but matters for code that authors and reads in the
   same tight sequence.

8. **For init-time one-shot lookups, use whichever backend is convenient
   — both work.** Functions that walk the authored prim hierarchy once at
   scene-load and return a path string are backend-agnostic; the
   slow-path and Fabric both carry that hierarchy at stage-open time
   (§1.1). The only reason to prefer usdrt here is consistency with
   code that does the same lookup per frame.

9. **Open `use_backend(...)` explicitly around prim hot paths.** The
   `RigidPrim` default backend is `tensor` (because that's the first
   item in its supported list), which on `set_world_poses` writes
   through the physics tensor view and **bypasses PhysX integration**.
   If you want a non-teleport, integrator-respecting write — which is
   almost always — pass `use_backend("usdrt", raise_on_unsupported=True)`
   around the block. The `raise_on_unsupported=True` flag is cheap
   insurance: it surfaces FSD-misconfiguration as a `RuntimeError` at
   the first call instead of letting the call silently land on a
   slower path. See §3.7 for the propagation rules.

---

## 9. Quick reference card

```python
# Imports
import omni.usd                                   # omni.usd.get_context()
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.simulation_manager            # SimulationManager
from isaacsim.core.simulation_manager import SimulationManager
# from pxr import Usd, UsdGeom                     # slow-path types
# import usdrt                                     # Fabric types

# Acquire stages
slow = stage_utils.get_current_stage(backend="usd")
fast = stage_utils.get_current_stage(backend="usdrt")

# Set the implicit default for a block (experimental prim methods + get_current_stage)
import isaacsim.core.experimental.utils.backend as backend_utils
with backend_utils.use_backend("usdrt", raise_on_unsupported=True):
    # Every experimental prim method in this block routes through USDRT.
    # get_current_stage(backend=None) also returns the Fabric stage.
    prim = fast.GetPrimAtPath("/World/MyPrim")

# Authoring (pxr only)
from pxr import UsdGeom
xform = UsdGeom.Xform.Define(slow, "/World/MyPrim")

# Per-frame reads (Fabric)
prim = fast.GetPrimAtPath("/World/MyPrim")
attr = prim.GetAttribute("xformOp:transform")
value = attr.Get()

# Toggle Fabric at runtime (after switch_physics_engine("physx"))
SimulationManager.enable_fabric(True)
assert SimulationManager.is_fabric_enabled()

# State query
stage_id = stage_utils.get_stage_id(slow)
```

---

## 10. Glossary

- **Slow-path stage** — the `pxr.Usd.Stage`. Python-friendly, full USD API.
  Reads the .usd file directly. **Always in memory** once a stage is
  opened, regardless of FSD — the only path that supports the full
  authoring surface. Does not receive per-step physics updates by
  default when Fabric is on (`/physics/updateToUsd = false`); the live
  values live on the Fabric side. See §1.3.
- **Fabric stage** — the `usdrt.Usd.Stage`. C++-backed struct-of-arrays
  view over the same scene, optimized for hot loops and GPU binding.
- **Fabric scene delegate (FSD)** — the C++ struct-of-arrays
  re-implementation of USD's scene delegate contract that backs the
  Fabric hot path. See §2.5 for the full picture. The shortest
  description: it lets the renderer, the physics view, and the
  experimental prim wrappers all read scene state from a single C++
  SoA mirror instead of walking `pxr.Usd.Stage` on the consumer's
  thread every frame. Selected by `[settings.app]
useFabricSceneDelegate = true`; must be on at app launch, cannot be
  flipped mid-session.
- **Fabric change-notification handler** — the bridge that propagates
  authoring edits from one stage to the other. Off by default per stage;
  enable with `SimulationManager.enable_fabric_usd_notice_handler(stage_id, True)`.
- **Authored value** — the value of an attribute as written in the .usd
  file. Visible on both stages.
- **Live (physics-driven) value** — the value updated by PhysX each
  physics step. Visible on Fabric only when `physics.fabricUpdate*` is
  on; mirrored to pxr only when `/physics/updateToUsd` is on (rarely).
- **USDRT** — "USD Runtime", the C++ Fabric scenegraph library; exposed in
  Python as the `usdrt` module.
- **`use_backend` context manager** — the implicit counterpart of
  `stage_utils.get_current_stage(backend=...)`. Sets a thread-local
  default that `backend=None` calls (and the experimental prim wrappers
  like `XformPrim` and `RigidPrim`) consult on entry. Valid values are
  `"usd"`, `"usdrt"`, `"fabric"`, `"tensor"`, `"simstate"`. `usdrt` and
  `fabric` require Fabric Scene Delegate; the check fires inside the
  context manager, so an app with FSD off never trips it (it just stays
  on the slow path). See §3.7.
- **Per-method default backend** — the first item of a method's
  supported-backends list (e.g. `["usd", "usdrt", "fabric"]` for
  `XformPrim`, `["tensor", "usd", "usdrt", "fabric"]` for `RigidPrim`).
  This is the dispatch that runs when no `use_backend` is active. For
  `RigidPrim` it is **`tensor`**, not `usd` — a frequent source of
  surprise. See §3.7.
