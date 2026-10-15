# 2dvsbpp-optima

Proven optima and checked packings for the 500 instances of Pisinger & Sigurd (2005), two-dimensional
variable-sized bin packing (*Discrete Optimization* 2(2):154–167). Items are oriented (no rotation) and
packings are not restricted to guillotine cuts.

Write-up: [Proving 237/500 Optima on the Pisinger–Sigurd (MB) Benchmark](https://chhc-l.github.io/posts/2dvsbpp-optima/)

## Problem

Pack n rectangles (n = 20, 40, 60, 80, 100) into bins chosen from at most 5 bin types. Each type has a
width, a height, a cost and unlimited supply. Minimise the total cost of the bins used. The literature
uses the same instance files with two cost definitions:

* `weighted` (W): the cost given in the instance file. This is the problem of Pisinger & Sigurd.
* `area` (U): cost = bin area. This is used by the heuristic literature and by the packingsolver
  best-known-solution file (`data/bks_packingsolver_2025-03.csv`).

## Results (`results/final/`)

| Proven optimal | W | U |
|---|---|---|
| n = 20 | 100 / 100 | 100 / 100 |
| n = 40 | 83 / 100 | 65 / 100 |
| n = 60 | 38 / 100 | 11 / 100 |
| n = 80 | 8 / 100 | 7 / 100 |
| n = 100 | 8 / 100 | 7 / 100 |
| **total** | **237** | **190** |
| confirmed by two methods sharing no modelling code | 234 | 189 |

At n = 80 and 100, lower bounds were only computed for class 9. More counts, including the
comparison with the packingsolver file, are in `results/final/SUMMARY.md`.

`results/final/results.csv` has one row per (instance, variant):

| Column | Meaning |
|---|---|
| `lb` | best proven lower bound (empty if none was computed) |
| `ub` | cost of the packing in `solution` |
| `closed` | 1 if `lb == ub` |
| `confirmed` | 1 if two proofs of `lb` share no modelling code (code families: `FAMILY` in `src/build_results.py`) |
| `lb_methods` / `lb_families` | methods and code families that reach `lb` |
| `solution` | checked packing (`results/final/solutions/<name>_<variant>.json`) |
| `csv_bks`, `csv_bound`, `csv_closed` | area variant only: values from the packingsolver file |
| `ub_method`, `ub_source` | method and original file name of the run that found the packing |

How far to trust the numbers:

* **Packings.** Every packing is accepted by `src/check_solution.py`. The checker uses only the
  standard library and shares no code with the solvers.
* **Lower bounds.** These come from CP-SAT, HiGHS and floating-point column generation. Every
  bound that depends on column generation was re-derived in integer arithmetic
  (`src/verify/cg_safe.py`). Solver infeasibility answers are trusted, so these are not formal
  proof certificates. The exception is the DRAT proofs written by `src/opp_sat.py`.

## Usage

Requires Python ≥ 3.10.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/check_final.py              # re-check all 1000 packings against the table
.venv/bin/python -m unittest discover tests      # checker tests + short end-to-end solver runs
.venv/bin/python src/check_solution.py data/instances/MB_C3_11.txt results/final/solutions/MB_C3_11_area.json area
```

Each solver's usage line is at the top of its file. Every run appends one row to
`results/ledger.csv` (`src/ledger.py`) and stores its packing under `results/solutions/`.
`src/build_results.py` folds the ledger into `results/final/`. The run ledger behind the published
table is not included.

## Code (`src/`)

| Files | Method |
|---|---|
| `sweep_rep.py`, `sweep_model.py`, `sweep_run.py` | CP-SAT model in which bin k is opened by item k (symmetry breaking) |
| `cpsat_direct.py`, `cpsat_slots_v0.py` | Two other CP-SAT models, used as cross-checks |
| `cg_v2.py`, `cg_iplattice.py`, `cg_bendex.py`, `cg_bound.py` | Dantzig–Wolfe column generation with exact CP-SAT pricing. The bound is rounded by an integer program over bin-type counts and strengthened by Benders cuts and exact fixed-count checks |
| `lattice_exact.py`, `lattice_batch.py` | Enumerate bin-count vectors in order of cost and refute each one |
| `opp_*.py` | 2D orthogonal packing for one bin: DFF certificates, CP-SAT, SAT with DRAT proofs, logic-based Benders |
| `primal_run.py`, `primal_heur.py`, `primal_lns.py`, `packingsolver_ub.py` | Packings: packingsolver, a MaxRects heuristic, CP-SAT large-neighbourhood search |
| `check_solution.py`, `check_final.py` | Standalone packing checker; re-check of the published table |
| `ledger.py`, `vsbpp_io.py`, `build_results.py` | Run ledger, instance reader, results table |
| `verify/` | Independent re-derivations and cross-checks of lower bounds |

`scripts/get_sat_tools.sh` builds CaDiCaL and drat-trim into `third_party/`. They are only needed
for the DRAT proofs.

## Data

* `data/instances/`: the 500 instances, from OR-Datasets (MIT licence).
* `data/bks_packingsolver_2025-03.csv`: from packingsolver (MIT licence).

`data/SOURCES.md` gives the details.

## License

MIT (`LICENSE`). The instance files and the packingsolver CSV keep their own MIT licences
(`data/SOURCES.md`).
