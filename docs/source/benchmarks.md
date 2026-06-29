# Performance Benchmarks

`mom6_forge` tracks performance regressions using [pytest-benchmark](https://pytest-benchmark.readthedocs.io). Benchmarks live in the existing test files so they share fixtures and stay in sync with the code they measure.

## Test discovery

pytest-benchmark identifies benchmarks via the `benchmark` fixture — any test that accepts it as an argument is a benchmark. `mom6_forge` also follows the `test_bench_*` naming convention to make them easy to grep for.

The fixture behaves differently depending on the flag passed to pytest:

| Flag | Behaviour |
|------|-----------|
| *(none)* | Benchmarks run and timing is collected |
| `--benchmark-only` | Only benchmark tests are collected; all others are skipped |
| `--benchmark-disable` | Benchmark tests run but timing is skipped (fixture is a pass-through) |
| `--benchmark-skip` | Benchmark tests are skipped entirely |

CI uses `--benchmark-disable` in the regular test workflow so benchmark test bodies still execute (catching functional regressions) without the timing overhead.

## Two-track design

**CI regression gates** — lightweight benchmarks on synthetic data, run in seconds on every PR targeting `main` via `.github/workflows/benchmark.yml`.

**SeaSloth** — real GEBCO data, production-sized grids, run manually on Derecho. Dashboard: <https://crocodile-cesm.github.io/SeaSloth/>.

## Running locally

```bash
conda activate mom6_forge
# Run benchmarks and print table
pytest tests/ --benchmark-only -v
# Save results to JSON
pytest tests/ --benchmark-only --benchmark-json=my_results.json
# Compare two result files
pytest-benchmark compare baseline.json my_results.json
```

## Updating the baseline

`tests/benchmark_baseline.json` is the committed reference. CI fails if any benchmark mean exceeds 130% of its baseline value.

```bash
pytest tests/ --benchmark-only \
    --benchmark-json=tests/benchmark_baseline.json \
    --benchmark-min-rounds=5
git add tests/benchmark_baseline.json
git commit -m "bench: update baseline"
```

## Adding a benchmark

Add a `test_bench_*` function to the relevant test file. Use `benchmark.pedantic` with `setup=` when the code under test mutates state:

```python
def test_bench_my_func(benchmark, tmp_path):
    def setup():
        obj = MyClass(...)
        return (obj,), {}

    def run(obj):
        obj.expensive_method()

    benchmark.pedantic(run, setup=setup, rounds=5, warmup_rounds=1)
```

Then regenerate and commit `tests/benchmark_baseline.json`.
