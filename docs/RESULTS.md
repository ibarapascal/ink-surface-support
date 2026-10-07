# Results

All numbers come from [`results/`](../results/). `python reproduction/rebuild.py --check`
rebuilds every CSV there from [`results/evidence.json`](../results/evidence.json).

## Setup

- **Scan:** PHercParis4 (20260411134726), level-2 coordinates, 9.6 µm per voxel.
  All spacings below are in these voxels, not scan resolution.
- **Arms:** working grid 5 voxels in every arm, same inputs and gates.
  - spacing 10, no patch (upstream uniform erosion)
  - spacing 10 + patch (`--coarse-mask-mode support`)
  - spacing 5, no patch
- **Regions:** three 1536³-voxel boxes, V1, V2 and V3. V1 and V2 were used while
  developing the method. V3 does not overlap them and was fixed before any
  patched output was produced there; it is the held-out test region (same scan
  and references, so not independent ground truth). Origins, seeds and input
  hashes are in [`reproduction/study-cases.json`](../reproduction/study-cases.json).
- **Attempts:** 32 fixed seeds per region and arm, 96 per arm. Each arm
  accepted the same 65 (V1 23, V2 24, V3 18). The 31 failures per arm
  (26 rejected by upstream gates, 5 late empty outputs) stay in the
  denominator and are listed in [`native-noinput.csv`](../results/native-noinput.csv);
  none were replaced.
- **Flatten:** each of the 195 accepted outputs went through the official
  Lasagna flatten (`flatten_fast.json`, CPU, float32, three 1000-step stages).
- **Coverage:** area of a registered reference surface (mm²) that lies within a
  tolerance of the flattened output's triangles, inside the central 1408³-voxel
  box. Tolerances: 0.5, 1, 2, 4, 8 voxels. References are scored separately and
  never summed: 20231210121321, 20230702185753, 20230929220926.

## Coverage, held-out region V3

| Reference | Tolerance (voxels) | Spacing 10, no patch | Spacing 10 + patch | Spacing 5, no patch | Gain vs spacing 10 | Gap closed | Share of spacing 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20231210121321 | 0.5 | 0.671 | 0.857 | 0.957 | +27.8% | 65.0% | 89.5% |
| 20231210121321 | 1 | 1.648 | 1.959 | 2.072 | +18.9% | 73.3% | 94.5% |
| 20231210121321 | 2 | 3.720 | 4.283 | 4.558 | +15.1% | 67.3% | 94.0% |
| 20231210121321 | 4 | 9.997 | 11.104 | 11.390 | +11.1% | 79.5% | 97.5% |
| 20231210121321 | 8 | 14.614 | 16.097 | 16.421 | +10.1% | 82.1% | 98.0% |
| 20230702185753 | 0.5 | 1.986 | 2.339 | 2.412 | +17.8% | 82.9% | 97.0% |
| 20230702185753 | 1 | 3.970 | 4.601 | 4.858 | +15.9% | 71.0% | 94.7% |
| 20230702185753 | 2 | 8.291 | 9.299 | 9.499 | +12.2% | 83.4% | 97.9% |
| 20230702185753 | 4 | 13.926 | 15.428 | 15.585 | +10.8% | 90.5% | 99.0% |
| 20230702185753 | 8 | 16.004 | 17.856 | 18.070 | +11.6% | 89.6% | 98.8% |

Coverage in mm². "Gap closed" is (patch − spacing 10) / (spacing 5 − spacing 10).
Reference 20230929220926 has 719.46 mm² of area in V3, but no output from any
arm comes near it, so it gives no answer either way.

## Coverage, all regions, 4-voxel tolerance

| Region | Reference | Spacing 10, no patch | Spacing 10 + patch | Spacing 5, no patch | Gap closed |
|---|---|---:|---:|---:|---:|
| V1 | 20231210121321 | 36.638 | 40.563 | 41.164 | 86.7% |
| V1 | 20230929220926 | 19.452 | 22.688 | 23.654 | 77.0% |
| V2 | 20231210121321 | 18.180 | 20.213 | 20.721 | 80.0% |
| V2 | 20230929220926 | 3.582 | 4.759 | 4.984 | 83.9% |
| V3 | 20231210121321 | 9.997 | 11.104 | 11.390 | 79.5% |
| V3 | 20230702185753 | 13.926 | 15.428 | 15.585 | 90.5% |

Reference 20230702185753 has no intersection in V1 or V2, and 20230929220926
none in V3. In no region, reference, box or tolerance did the patch cover less
than unpatched spacing 10. Secondary 384³-voxel boxes and every row are in
[`reference-coverage.csv`](../results/reference-coverage.csv).

Plots: [V1/V2 primary box](../figures/main10-seen-B1408.png),
[V1/V2 secondary box](../figures/main10-seen-A384.png),
[V3](../figures/V3-reference-curves.png).

## Flatten cost

Sums over accepted patches. CPU is user + system seconds of the flatten CLI;
file sizes include checkpoints and snapshots.

| Region (patches) | Spacing 10, no patch | Spacing 10 + patch | Spacing 5, no patch |
|---|---:|---:|---:|
| V1 (23), CPU s | 398.37 | 397.35 | 856.58 |
| V2 (24), CPU s | 432.42 | 431.62 | 966.15 |
| V3 (18), CPU s | 274.07 | 276.64 | 555.14 |
| V3 (18), files MB | 26.28 | 26.30 | 94.59 |

Each timing is a single run. Per-case values, wall time and peak RSS are in
[`cost.csv`](../results/cost.csv). Plots:
[V1/V2](../figures/main10-seen-cost.png), [V3](../figures/V3-consumer-cost.png).

## Distortion

The patch keeps more surface, and the flattened map also stretches more. To
compare like with like, the symmetric-Dirichlet loss (SDir, Lasagna's native
float32 term) was measured on exactly the same original source cells in both
arms, with the same area weights. Newly kept cells are left out.

| Shared cells, spacing 10 | No patch | + patch |
|---|---:|---:|
| Cells / source area | 72,066 / 664.24 mm² | same |
| Area-weighted mean SDir | 0.000178489 | 0.000199355 (+11.69%) |
| Weighted median | 0.000053883 | 0.000066757 |
| Weighted p95 | 0.000732422 | 0.000804424 |

The case mean rose in all 65 patches (V1 +11.4%, V2 +11.5%, V3 +13.6%).
SDir against unpatched spacing 5 was not measured. Per-case values are in
[`paired-distortion.csv`](../results/paired-distortion.csv); plot:
[paired-distortion.png](../figures/paired-distortion.png).

## Unknown and outside area

Some of the extra surface cannot be checked. In V3, output area inside the box
with no reference within 8 voxels ("unknown") rises from 92.13 to 109.93 mm²
for reference 20231210121321 and from 90.51 to 108.07 mm² for 20230702185753.
Output area outside the box rises from 0.92 to 2.01 mm². Unpatched spacing 5
is higher still (113.52, 111.76 and 2.21 mm²). Per output:
[`consumer-diagnostics.csv`](../results/consumer-diagnostics.csv) and the
`output_gross` records in `evidence.json`.

## Follow-up: spacing 5 vs spacing 2.5

The same comparison one step finer: spacing 5 without and with the patch,
against unpatched spacing 2.5 (working grid still 5). Two development cases
from earlier work, scored in their original boxes, so not a held-out test.

| Case (reference, box) | Tolerance | Spacing 5, no patch | Spacing 5 + patch | Spacing 2.5, no patch | Gap closed |
|---|---:|---:|---:|---:|---:|
| C0-P05 (20230702185753, 896³) | 0.5 | 0.102 | 0.122 | 0.123 | 94.6% |
| | 1 | 0.205 | 0.267 | 0.281 | 81.7% |
| | 2 | 0.458 | 0.582 | 0.596 | 90.1% |
| | 4 | 1.090 | 1.289 | 1.307 | 91.8% |
| | 8 | 2.177 | 2.436 | 2.454 | 93.5% |
| C0-P29 (20231210121321, 384³) | 0.5 | 0.230 | 0.253 | 0.261 | 75.0% |
| | 1 | 0.443 | 0.497 | 0.502 | 91.6% |
| | 2 | 0.796 | 0.878 | 0.906 | 74.7% |
| | 4 | 1.243 | 1.425 | 1.432 | 96.2% |
| | 8 | 1.589 | 2.031 | 1.813 | patch higher |

- SDir on shared cells: C0-P05 +7.56%, C0-P29 +13.57%.
- Flatten CPU (s), same arm order: C0-P05 24.22 / 23.55 / 68.84;
  C0-P29 38.45 / 37.14 / 115.48. The C0-P05 spacing-5 time is reused from an
  earlier run.
- For C0-P29, the share of the exported surface within 4 voxels of the
  reference fell from 88.87% to 81.62%: more of the reference is covered, but
  more of the output is also far from it.
- Three more cases (P10, P06, P12) were run to exercise the code only; none of
  them intersects a reference.

Plot: [spacing5-development-coverage.png](../figures/spacing5-development-coverage.png).

## Earlier and side experiments

- **Spacing 20 (upstream default).** The first trial used output spacing 20.
  The patch kept more area there too, but unpatched spacing 10 covered more of
  the reference than patched spacing 20. One case also switched to a different
  largest component, so kept area is not guaranteed to grow.
- **Lasagna `flatten_output_step`.** On C0-P05, setting the flatten output
  step to 5 on a spacing-10 surface raised coverage in both arms. This is a
  separate existing option that changes the optimization itself. Rows are in
  `reference-coverage.csv` under study `existing-output-step-option`.
- **CT spot check.** The available CT crop overlaps only C0-P29. Of the
  0.464 mm² of newly kept area inside it, three sections chosen by geometry
  showed one visible offset from a bright band. Inconclusive; no images are
  included.

## Where the gain comes from

In every measured case the support check never had to delete a corner. The
whole gain comes from skipping the final erosion, not from repairing holes.
