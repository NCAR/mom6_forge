# Build Docs

## Regenerate API reference

Run from the repo root whenever modules are added or removed:

```bash
sphinx-apidoc -o docs/source/api mom6_forge --force
```

## Build HTML

```bash
sphinx-build -b html -c docs/source docs _build
```

Output is written to `_build/`. Open `_build/source/index.html` to preview locally.