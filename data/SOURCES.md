# Data provenance

## `instances/` — 500 Pisinger–Sigurd (2005) 2DVSBPP instances

* Source: https://github.com/Oscar-Oliveira/OR-Datasets
  * path `Cutting-and-Packing/2D/Datasets/MB/original.zip`
  * commit e2e151cf241ab97d860f657cbfe515232e7096d7 (2025-09-09)
  * MIT licence, © 2018 Óscar Oliveira
  * The OR-Datasets README attributes the MB set to D. Pisinger and M. Sigurd, "The
    two-dimensional bin packing problem with variable bin sizes and costs", *Discrete
    Optimization* 2(2):154–167, 2005.
* Changes: files renamed from `Original\MB_C<c>_<j>` to `MB_C<c>_<j>.txt`. The content is
  unchanged.
* Consistency checks (2026-10-13):
  * Identical, bin and item multisets, on all 500 instances to the CSV copies in
    `fontanf/packingsolver/data/rectangle/pisinger2005/`.
  * Identical to the OR-Datasets `json/` copy except in 21 instances, where the original file
    lists a bin type twice and the JSON merges the duplicate. This is equivalent because bin
    supply is unlimited.

## `bks_packingsolver_2025-03.csv` — best known values, U (area-cost) variant

* Source: https://github.com/fontanf/packingsolver
  * file `data/rectangle/data_variable_sized_bin_packing_oriented_pisinger2005.csv`
  * added 2025-03-20; snapshot taken at commit 79b67f8 (2026-10-13)
  * MIT licence, © 2020 Florian Fontan
* Columns: `Path`, `Best known solution value`, `Best known bound`, `Wei2013 / Solution value`.
* These values are for **cost = bin area**, not for the file costs. See
  the README.
