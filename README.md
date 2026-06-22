# isaac_sim_backend_comparison

A guide + microbenchmark for the Isaac Sim 6.0 USD backends (`usd`, `usdrt`,
`tensor`). See [`docs/usd_usdrt_fabric_backends.md`](docs/usd_usdrt_fabric_backends.md)
for the background on what each backend does, when to use it, and how to
configure it. See [`backend_comparison/README.md`](backend_comparison/README.md)
for the microbenchmark that exercises the multi-backend hot path of
`isaacsim.core.experimental.prims`.

## Run the benchmark inside Docker

```bash
# 1. Build the image (one-off)
docker compose build

# 2. Run the benchmark (the script is mounted at /ws inside the container)
docker compose run --rm isaac-sim /ws/backend_comparison/run_comparison.sh usd usdrt

# 3. Read the artifacts straight off the host (they're bind-mounted):
ls output/                         # benchmark_usd.json, benchmark_usdrt.json, …
cat output/backend_comparison.md
```

By default the script writes its outputs to `/ws/output` inside the container,
which the `docker-compose.yml` bind-mounts to `<repo>/output/` on the host. So
the JSON reports and the comparison table land on the host filesystem the
moment the script finishes — no `docker cp` required.

To override the output dir, pass it as `OUTPUT_DIR`:

```bash
docker compose run --rm -e OUTPUT_DIR=/tmp isaac-sim \
    /ws/backend_comparison/run_comparison.sh usd
```

Inside the container, `ISAAC_PATH=/isaac-sim` is set automatically (the Isaac
Sim install ships at `/isaac-sim/` in the base image). The
`run_comparison.sh` script resolves `python.sh` from there, so the user's
command above works without any extra flags.

## Run a single backend

```bash
docker compose run --rm isaac-sim /ws/backend_comparison/run_comparison.sh usd
docker compose run --rm isaac-sim /ws/backend_comparison/run_comparison.sh usdrt
```

## Open an interactive shell

```bash
docker compose run --rm isaac-sim /isaac-sim/python.sh   # Isaac Sim's Python REPL
docker compose run --rm isaac-sim bash                  # plain bash; then run anything
```

## Run the benchmark outside Docker

If you have an Isaac Sim install on the host (release build or pre-built
package), point `run_comparison.sh` at it via `ISAAC_PATH`:

```bash
ISAAC_PATH=/path/to/isaac-sim ./backend_comparison/run_comparison.sh usd usdrt
```

See [`backend_comparison/README.md`](backend_comparison/README.md) for
benchmark-only docs (CLI flags, output format, caveats).
