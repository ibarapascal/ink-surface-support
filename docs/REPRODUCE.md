# Reproduce

Three levels, from cheap to expensive: run the tests, rebuild the tables and
figures from the recorded numbers, or rerun the study on real data.

## Environment

- **Python 3.11 or newer.** The pinned test dependencies (NumPy 2.4.2,
  SciPy 1.17.1) need 3.11+. `reproduction/rebuild.py` and `tools/install_patch.py`
  use only the standard library.
- Tests: `pip install -r requirements-test.txt`.
- Figures: `pip install -r requirements-figures.txt` (Matplotlib 3.10.8).
- Running the grower also needs Villa's own dependencies (`rustworkx`, `igl`
  from libigl, `tifffile`, ...). The flatten needs a Lasagna environment; the
  study used Python 3.12 on macOS arm64 with PyTorch 2.8.0 on CPU.

## Tests

```sh
python -m unittest discover -s tests -v
python vendor/output-spacing-10/test_support_safe_finalize.py -v
```

The first command runs 37 tests: the installer (check, apply, restore, refusal
of modified or conflicting files, rollback after a failed write), the
spacing-5 interface, and the flatten wrapper (driven by a fake Lasagna CLI in a
real subprocess). The second runs 11 numerical tests of the support rule on
small synthetic grids. No data, network or optimizer is involved.

## Rebuild tables and figures

```sh
python reproduction/rebuild.py --check            # verify results/*.csv and summary.json
python reproduction/rebuild.py --out build/rebuilt
python reproduction/render_figures.py --check     # re-render and compare PNG hashes
python reproduction/render_figures.py --out build # write figures/ and assets/ under build/
```

`rebuild.py --check` recomputes every table in `results/` from
`results/evidence.json` and fails on any byte difference. It also checks the
evidence hash, attempt accounting (32 per region and arm), monotone
tolerance curves, per-component sums, the paired-distortion arithmetic, and the
hashes of the evaluation scripts in `reproduction/evaluation/`. It works on
Python 3.9+. This checks the report, not the science: it does not rerun any
measurement.

## Install the patch

```sh
python tools/install_patch.py --checkout /path/to/villa --resample-spacing 5 --output-spacing 10 --check
python tools/install_patch.py --checkout /path/to/villa --resample-spacing 5 --output-spacing 10 --apply
python tools/install_patch.py --checkout /path/to/villa --restore
```

- The installer requires `spiral-fitting/grow_track_graph.py` to be the exact
  file from Villa `fb8c2c4c2705746eefb42e9bb359560906f391e2`
  (SHA-256 `8156f58f…c734e6`; the same file is in `e0bbb8b4`). Anything else is refused.
- It copies the verified files from `vendor/output-spacing-10/` (or
  `vendor/output-spacing-5/`), keeps the original in
  `.ink-surface-support-install/`, and `--restore` puts it back.
- Spacing 5 and spacing 10 install the same producer file, so use separate
  checkouts or restore before switching.
- The equivalent diffs are in `patches/` if you prefer `patch -p1`.

## Run the grower

Inputs are the upstream packed tracks (`.vctracks`) and their crossing index
(`.npz`), built with the tools in Villa's
[spiral-fitting README](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/README.md#extracting-surface-tracks).

```sh
python spiral-fitting/grow_track_graph.py TRACKS.vctracks CROSSINGS.npz OUT_DIR \
  --random-count 1 --workers 1 --resample-spacing 5 --output-spacing 10 \
  --coarse-mask-mode support
```

- `--coarse-mask-mode erode` (the default) is the upstream behavior.
- Support mode accepts only working spacing 5 with output spacing 10 (or 5 in
  a spacing-5 install) and errors otherwise.
- `--random-count 1` asks for one accepted surface and may try several seeds.
  `--seeds` uses a different size gate, so it does not reproduce the
  random-mode runs in this study.
- `meta.json` records `coarse_mask_mode` and `coarse_support_certificate`.

## Flatten with Lasagna

`tools/flatten.py` runs the exact recipe used here: Villa's `lasagna/fit.py`
with the original `lasagna/configs/flatten_fast.json`, CPU, float32, no
compile, one thread, three 1000-step stages.

```sh
python tools/flatten.py prepare --villa /path/to/villa --source OUT_DIR/patch.tifxyz \
  --run-dir runs/patch1 --python /path/to/lasagna-env/bin/python --origin-zyx Z Y X
python tools/flatten.py run --plan runs/patch1/plan.json
```

- `prepare` checks the 93 Lasagna source files against
  `reproduction/flatten-source-lock.json`, hashes the input and writes a plan.
  It starts nothing.
- `--origin-zyx` is the patch's global origin in voxels. It is only recorded
  for later scoring; coordinates are not shifted.
- `run` launches the CLI and writes `stdout.log`, `stderr.log` and
  `result.json`. Its success state means the run finished and the expected
  files exist, not that the surface is good.

Calling `lasagna/fit.py` directly with `model_init=flatten` and the TIFXYZ in
`external_surfaces` is equivalent; the wrapper just pins the settings.

## Rerun the study on real data

This needs the scan data and a full Villa/Lasagna environment. Everything
needed to identify the inputs is recorded:

- [`reproduction/data-sources.json`](../reproduction/data-sources.json): URLs,
  chunk keys, byte lengths and SHA-256 of the CT and surface-prediction chunks
  and of the three reference surfaces.
- [`reproduction/study-cases.json`](../reproduction/study-cases.json): region
  origins, the 96 seed rows per arm, the gate outcome of each attempt, the
  development cases, and the packed-track and crossing hashes.
- [`reproduction/runtime-observations.json`](../reproduction/runtime-observations.json):
  the effective Lasagna settings and versions of a recorded flatten run.

Coordinate conventions:

- CT and prediction are level 2 of the 2.4 µm scan, so one voxel is 9.6 µm.
  Volumes are ZYX; surface vectors are XYZ.
- Registered reference XYZ is at full resolution: divide by 4 for level 2,
  then subtract the region origin for local coordinates.
- TIFXYZ `meta.scale` describes grid sampling, not an XYZ multiplier.

Steps:

1. Download and verify every listed chunk (never fill missing chunks with
   zeros) and crop each region.
2. Extract and pack tracks and build crossings with the recorded upstream
   settings; check the packed-track and crossing hashes before using seed
   rows, which only mean something for that exact packing.
3. Run the three arms with identical settings in fresh output directories;
   keep failures.
4. Flatten every accepted output with the recipe above.
5. Score with the scripts in `reproduction/evaluation/` (index:
   [`evaluation-source-manifest.json`](../reproduction/evaluation-source-manifest.json)),
   using the distance engine in `reproduction/numeric-sources/spiralcheck/` and
   the SDir code in `reproduction/numeric-sources/villa/`.

The evaluation scripts are the ones used for the study. Machine-specific paths
were replaced by placeholders such as `__HISTORICAL_WORK_ROOT__`, and internal
identifier strings were renamed for publication; logic is unchanged. Bind the
placeholders to your own directories. Some scripts also check the hashes of
the original intermediate files, so they will stop on regenerated inputs until
those checks are updated.

## Repository layout

| Path | Contents |
|---|---|
| `vendor/villa-pinned/` | Unmodified upstream `grow_track_graph.py` |
| `vendor/output-spacing-10/`, `vendor/output-spacing-5/` | Patched producer and support helper per spacing |
| `vendor/SOURCE.json` | Upstream origin and file hashes used by the installer |
| `patches/` | The same changes as unified diffs |
| `tools/` | `install_patch.py`, `flatten.py` |
| `tests/` | Installer, flatten-wrapper and spacing-5 tests |
| `results/` | `evidence.json` and the tables rebuilt from it |
| `figures/`, `assets/` | Plots rendered from `evidence.json` |
| `reproduction/` | `rebuild.py`, `render_figures.py`, input manifests, evaluation scripts, numerical sources |
