# Third-party notices

## Code

**ScrollPrize / Vesuvius Challenge Villa**
([repository](https://github.com/ScrollPrize/villa), revision
`fb8c2c4c2705746eefb42e9bb359560906f391e2`). MIT, copyright (c) 2024 Vesuvius
Challenge; full text in [licenses/VILLA-MIT.txt](licenses/VILLA-MIT.txt). It
covers:

- `vendor/villa-pinned/grow_track_graph.py` (unmodified) and the upstream parts
  of the patched producers in `vendor/output-spacing-10/`,
  `vendor/output-spacing-5/` and `patches/`;
- the unmodified Lasagna files `model.py`, `opt_loss_flatten.py` and
  `dtypes.py` in `reproduction/numeric-sources/villa/`.

**spiralcheck** ([Nicodol/spiralcheck](https://github.com/Nicodol/spiralcheck),
revision `d1b50e2957409a870225fb9f5dcc5e25f7a0f9da`). MIT, copyright (c) 2026
Nicolas Dolegieviez; the license and source hashes sit next to
`reproduction/numeric-sources/spiralcheck/geometry.py`.

Runtime libraries (NumPy, SciPy, PyTorch, tifffile, rustworkx, libigl, ...) are
not bundled and keep their own licenses.

Everything else (support helper, tools, tests, evaluation scripts, docs) is
MIT, copyright (c) 2026 ibarapascal; see [LICENSE](LICENSE).

## Data

The MIT licenses above cover code only.

- The PHercParis4 scan and the surface-prediction volume come from the
  [Vesuvius Challenge data portal](https://scrollprize.org/data), licensed
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) unless an
  asset says otherwise. Credit: Vesuvius Challenge, "Vesuvius Challenge – CT
  Scans of Herculaneum Papyri".
- Files on the older server `dl.ash2txt.org` are under that server's own
  [agreement](https://dl.ash2txt.org/LICENSE.txt).
- The reference segments 20231210121321, 20230702185753 and 20230929220926 are
  historical segmentations by their original authors, registered to this scan.
  For EduceLab-derived material, cite Parsons, Parker, Chapman, Hayashida and
  Seales (2023), [EduceLab-Scrolls](https://doi.org/10.48550/arXiv.2304.02084),
  and acknowledge EduceLab / University of Kentucky.

This repository contains no CT voxels, prediction volumes, surface coordinates
or images. It contains aggregate statistics, file identifiers and hashes;
[reproduction/data-sources.json](reproduction/data-sources.json) says where each
input comes from.
