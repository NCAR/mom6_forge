# Contributing

## Running the Tests

```bash
pytest
```

Most of the test suite runs anywhere. A handful of tests in
`tests/test_git_efficiency.py` are marked `benchmark` and require a
GLADE-mounted global grid (`tx2_3v3`) that is only available on NCAR HPC
systems (e.g. Casper, Derecho). Those tests skip themselves automatically
when that data isn't present, so a plain `pytest` run is safe anywhere — no
special flags are needed unless you want to specifically run and time them on
an NCAR system:

```bash
pytest tests/test_git_efficiency.py -m benchmark -v -s
```

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
