# Build Docs

## Regenerate API reference

Run from the repo root whenever modules are added or removed:

```bash
sphinx-apidoc -o docs/source/api mom6_forge --force
```

## Build HTML

```bash
sphinx-build -b html docs/source docs/_build
```

Output is written to `docs/_build/`. Open `docs/_build/index.html` to preview locally.