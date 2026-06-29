Performance Benchmarks
======================

``mom6_forge`` tracks performance regressions using `pytest-benchmark
<https://pytest-benchmark.readthedocs.io>`_. Benchmarks live directly in the
existing test files so they share fixtures and stay in sync with the code they
measure.

Two-track design
----------------

There are two complementary systems:

**CI regression gates** (this page)
  Lightweight benchmarks in the test suite — synthetic data, tiny grids, run
  in seconds. Fire on every pull request targeting ``main`` via
  ``.github/workflows/benchmark.yml``. Catch order-of-magnitude slowdowns
  before they land.

**SeaSloth** (separate repo)
  Full characterisation suite — real GEBCO data, production-sized grids, run
  manually on Derecho. Produces the interactive dashboard. See the
  ``dev/SeaSloth`` directory of the CROC monorepo.

Where the benchmarks live
-------------------------

Benchmark tests follow the naming convention ``test_bench_*`` and sit at the
bottom of whichever test file exercises the same code:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Test
     - File
     - What it measures
   * - ``test_bench_grid_construction``
     - ``tests/test_grid.py``
     - ``Grid.__init__`` supergrid computation, 40×40 cell domain
   * - ``test_bench_set_from_dataset``
     - ``tests/test_topo_bathymetry_workflows.py``
     - Full ``ocean_frac`` + ``stats`` pipeline on a 3×3 synthetic grid

Running benchmarks locally
--------------------------

Benchmarks are **skipped** in normal test runs (``--benchmark-disable`` is
passed in CI). To run them explicitly::

    conda activate mom6_forge
    cd regional_mom_workflows/mom6_forge

    # Run only benchmark tests, print table
    pytest tests/ --benchmark-only -v

    # Run and save results to a JSON file
    pytest tests/ --benchmark-only --benchmark-json=my_results.json

    # Compare two saved result files
    pytest-benchmark compare baseline.json my_results.json

Updating the baseline
---------------------

``tests/benchmark_baseline.json`` is the committed reference. The CI
regression check fails if any benchmark mean exceeds 130% of its baseline
value.

To regenerate after an intentional performance improvement (or when adding a
new benchmark)::

    pytest tests/ --benchmark-only \
        --benchmark-json=tests/benchmark_baseline.json \
        --benchmark-min-rounds=5

Then commit the updated file::

    git add tests/benchmark_baseline.json
    git commit -m "bench: update baseline"

.. note::

   The baseline was generated on Derecho login nodes (AMD EPYC 9555). GitHub
   Actions runners use different hardware, so absolute times differ. The
   regression check compares each PR's results against the baseline from that
   same runner, so cross-machine drift does not cause false positives — but
   you should regenerate the baseline once the ``benchmark.yml`` CI job has
   run at least once on a GH Actions runner and upload the resulting artifact
   as the new ``tests/benchmark_baseline.json``.

CI workflow
-----------

``.github/workflows/benchmark.yml`` fires on pull requests targeting ``main``:

1. Installs the ``mom6_forge`` conda environment (cached).
2. Runs ``pytest tests/ --benchmark-only`` and saves ``benchmark_results.json``.
3. Compares against ``tests/benchmark_baseline.json`` — fails the job if any
   benchmark is >30% slower than baseline. If the baseline is empty, the check
   is skipped with a notice.
4. Uploads ``benchmark_results.json`` as a GitHub Actions artifact for manual
   inspection.

Adding a new benchmark
----------------------

1. Add a ``test_bench_*`` function to the relevant test file. Use
   ``benchmark.pedantic`` with a ``setup=`` function when the code under test
   mutates state (so each timing round starts clean)::

       def test_bench_my_func(benchmark, tmp_path):
           def setup():
               obj = MyClass(...)
               return (obj,), {}

           def run(obj):
               obj.expensive_method()

           benchmark.pedantic(run, setup=setup, rounds=5, warmup_rounds=1)

2. Run the benchmarks and regenerate the baseline (see above).
3. Commit both the new test and the updated ``tests/benchmark_baseline.json``.
