# Contributing

## Running the Tests

```bash
pytest
```

Most of the test suite runs anywhere. A handful of tests are marked
`benchmark`: those in `tests/test_git_efficiency.py` need a GLADE-mounted
global grid (`tx2_3v3`), and one in `tests/test_conformal.py` times a
full-resolution California solve. They skip themselves off NCAR HPC systems
(e.g. Casper, Derecho), so a plain `pytest` run is safe anywhere. To run and
time them on an NCAR system:

```bash
pytest tests/test_git_efficiency.py tests/test_conformal.py -m benchmark -v -s
```

Tests that build a `GridSketcher` or call `corner_diagnostics.ocean_mask` draw
land from cartopy's Natural Earth shapefiles. Where cartopy has used them
before (as on NCAR systems), no network is needed; on a fresh machine,
including a CI runner, cartopy downloads them once, the first time they are
needed.

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

The build runs one notebook, `notebooks/7_demo_editors.ipynb`, so that its
widgets show on the website. It writes a `TopoLibrary/` folder and two `.nc`
files next to the notebook (all ignored by git), and its `Topo(git=True)` needs
a git user name and email to commit with.

The live widget demos on the {doc}`widgets` page need `jupyterlite-sphinx`,
`jupyterlite-core` and `jupyterlite-pyodide-kernel` (see
`.github/workflows/docs.yml` for the versions) and a `mom6_forge` wheel in
`dist/`:

```bash
pip wheel --no-deps -w dist .
```

Without either, the build still works and leaves the live demos out.
