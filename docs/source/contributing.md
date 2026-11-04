# Contributing

## Running the Tests

```bash
pytest
```

Most of the test suite runs anywhere. A handful of tests in
`tests/test_git_efficiency.py` (GLADE-mounted `tx2_3v3` global grid) and
`tests/test_conformal.py` (one full-resolution California solve) are
marked `benchmark` and skip themselves automatically off NCAR HPC systems
(via `tests/utils.py`'s `on_cisl_machine()`), so a plain `pytest` run is safe
anywhere — no special flags are needed unless you want to specifically run
and time them on an NCAR system:

```bash
pytest tests/test_git_efficiency.py tests/test_conformal.py -m benchmark -v -s
```

`tests/test_grid_sketcher_ipympl.py` runs the real `ipympl` backend in a
subprocess and turns a missing `ipympl` install into a skip rather than a
failure, so it is safe to run unconditionally.

Most `GridSketcher`/map-drawing tests (anything that builds a `GridSketcher`
or calls `corner_diagnostics.ocean_mask`) draw land and coastline via cartopy's
Natural Earth 50 m shapefiles. On a machine where these are already cached
(e.g. an NCAR HPC system with prior cartopy use) this needs no network
access; on a fresh environment — including a CI runner — cartopy fetches
them once, on demand, the first time they're needed, which adds a small
amount of one-time download latency to that job.

## Building the Documentation

### Regenerate the API Reference

Run from the repo root whenever modules are added or removed:

```bash
sphinx-apidoc -o docs/source/api mom6_forge --force
```

Note that `sphinx-apidoc` always emits `.rst`, even though the rest of the
documentation is written in MyST Markdown — this is expected. Sphinx builds
`.rst` and `.md` sources side by side.

### Build HTML

```bash
sphinx-build -b html docs/source docs/_build
```

Output is written to `docs/_build/`. Open `docs/_build/index.html` to preview
locally.
