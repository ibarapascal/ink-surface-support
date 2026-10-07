"""Restricted registered-reference adapter; never a sheet-switch truth oracle.

Distance algorithm: unchanged MIT spiralcheck geometry.py, pinned in provenance.
Reference preparation and candidate evaluation use the same geometric distance core.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
from scipy import ndimage
import tifffile

sys.path.insert(0, str(Path(__file__).parent / 'vendor' / 'spiralcheck'))
from geometry import TriangleSoup, surface_distance

ORIGIN_XYZ = np.array([3840., 5504., 16256.])
SHAPE = 512
HALO = 32
CORE_MARGIN = 64
SEED_RADIUS = 16
SEED_GRID = (382, 300)
TOLERANCES = [0.5, 1., 2., 4., 8.]
UM_PER_VOX = 9.6
SOURCE_SHA = 'd1b50e2957409a870225fb9f5dcc5e25f7a0f9da'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def write_npz(path, **arrays):
    temporary = Path(str(path) + '.partial')
    with open(temporary, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(temporary, path)


def all_quad(mask):
    return mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]


def grid_triangles(xyz, qmask):
    """Villa triangulation, retaining parameter-grid provenance."""
    h, w = xyz.shape[:2]
    r, c = np.nonzero(qmask)
    tl = r * w + c
    tr, bl, br = tl + 1, tl + w, tl + w + 1
    ids = np.concatenate([np.stack([bl, tl, tr], 1),
                          np.stack([bl, tr, br], 1)])
    unique, inverse = np.unique(ids, return_inverse=True)
    verts = xyz.reshape(-1, 3)[unique].astype(np.float64)
    faces = inverse.reshape(-1, 3)
    uv = np.stack([r, c], 1)
    return verts, faces, np.tile(uv, (2, 1)), unique


def face_stats(vertices, faces):
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    centers = (a + b + c) / 3
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    return centers, area


def core_mask(points):
    return ((points >= CORE_MARGIN) & (points <= SHAPE - CORE_MARGIN)).all(1)


def load_grid(path, frame):
    path = Path(path)
    arrays = [tifffile.imread(path / (name + '.tif'), maxworkers=1) for name in 'xyz']
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError('Coordinate shapes differ or are not grids')
    valid = np.ones(arrays[0].shape, bool)
    sentinel = np.ones_like(valid)
    for a in arrays:
        valid &= np.isfinite(a)
        sentinel &= a == -1
    valid &= ~sentinel
    # Native full-scan validity BEFORE shifting into the ROI-local frame.
    if frame == 'fullres_xyz':
        valid &= arrays[2] > 0
    mask_path = path / 'mask.tif'
    if mask_path.exists():
        mask = tifffile.imread(mask_path, maxworkers=1)
        if mask.shape != valid.shape:
            raise ValueError('Mask shape mismatch')
        valid &= mask != 0
    xyz = np.stack(arrays, -1).astype(np.float32, copy=False)
    if frame == 'fullres_xyz':
        xyz /= 4.
        xyz -= ORIGIN_XYZ.astype(np.float32)
    elif frame == 'global_l2_xyz':
        xyz -= ORIGIN_XYZ.astype(np.float32)
    elif frame != 'local_l2_xyz':
        raise ValueError(frame)
    return xyz, valid


def prepare(reference, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xyz, valid = load_grid(reference, 'fullres_xyz')
    seed = xyz[SEED_GRID].astype(float)
    if not valid[SEED_GRID]:
        raise ValueError('Invalid frozen GP seed')
    in_roi = valid & ((xyz >= 0) & (xyz <= SHAPE)).all(2)
    in_halo = valid & ((xyz >= -HALO) & (xyz <= SHAPE + HALO)).all(2)
    roi_quads, halo_quads = all_quad(in_roi), all_quad(in_halo)
    labels, n_components = ndimage.label(roi_quads)
    # Fixed seed quad, not chosen by candidate overlap or score.
    target = int(labels[SEED_GRID])
    if target == 0:
        raise ValueError('Frozen seed quad outside valid ROI')
    vertices, faces, uv, grid_vertices = grid_triangles(xyz, halo_quads)
    centers, area = face_stats(vertices, faces)
    face_component = labels[uv[:, 0], uv[:, 1]]
    boundary_quads = halo_quads & ~ndimage.binary_erosion(halo_quads)
    face_boundary = boundary_quads[uv[:, 0], uv[:, 1]]
    core = core_mask(centers)
    away_seed = np.linalg.norm(centers - seed, axis=1) > SEED_RADIUS
    nonzero = area > 1e-10
    target_core = (face_component == target) & core & nonzero & ~face_boundary
    # Fixed domain before seeing any candidate.
    arrays = dict(vertices=vertices, faces=faces, face_uv=uv,
                  source_grid_vertices=grid_vertices, component=face_component,
                  boundary=face_boundary, centers=centers, area=area,
                  target_core=target_core, target_unseeded=target_core & away_seed,
                  seed_xyz=seed, target_component=np.array(target),
                  source_shape=np.array(valid.shape))
    write_npz(out / 'reference-domain.npz', **arrays)
    source_files = {f: sha(Path(reference) / f) for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json')}
    mask_file = Path(reference) / 'mask.tif'
    if mask_file.exists():
        source_files['mask.tif'] = sha(mask_file)
    info = {
        'schema': 'iss-patch-reference-domain/1',
        'status': 'PROVISIONAL_LOCAL_BRANCH_REFERENCE_ONLY',
        'reference_files_sha256': source_files,
        'domain_sha256': sha(out / 'reference-domain.npz'),
        'code_sha256': sha(__file__), 'geometry_sha256': sha(Path(__file__).parent / 'vendor/spiralcheck/geometry.py'),
        'spiralcheck_source_sha': SOURCE_SHA,
        'frame': 'ROI-local XYZ in 9.6um input voxels',
        'origin_global_l2_xyz': ORIGIN_XYZ.tolist(),
        'roi_size': SHAPE, 'halo': HALO, 'core_margin': CORE_MARGIN,
        'seed_exclusion_radius': SEED_RADIUS, 'tolerances_vox': TOLERANCES,
        'seed_grid': list(SEED_GRID), 'seed_local_xyz': seed.tolist(),
        'target_component': target, 'roi_quad_components': n_components,
        'reference_triangle_count': len(faces), 'reference_compact_vertices': len(vertices),
        'target_core_triangles': int(target_core.sum()),
        'target_core_area_mm2': float(area[target_core].sum() * UM_PER_VOX**2 / 1e6),
        'target_core_unseeded_area_mm2': float(area[target_core & away_seed].sum() * UM_PER_VOX**2 / 1e6),
        'components': [{'id': i, 'roi_faces': int((face_component == i).sum()),
                        'roi_area_mm2': float(area[face_component == i].sum() * UM_PER_VOX**2 / 1e6)}
                       for i in range(1, n_components + 1)],
        'registration_accuracy_independently_verified': False,
        'identity_verdict': 'UNKNOWN',
        'reference_quality': 'Official GP registered mesh; local identity and registration error not independently certified',
        'training_overlap': 'Unknown; seed/ROI selected from reference, report seed-excluded metrics separately',
        'candidate_seen': False,
    }
    write_json(out / 'reference-domain.json', info)
    return info


def summary(dist, weights):
    if not len(dist) or weights.sum() <= 0:
        return {'status': 'INSUFFICIENT_REFERENCE', 'samples': 0, 'coverage': None}
    order = np.argsort(dist, kind='stable')
    cdf = np.cumsum(weights[order]) / weights.sum()
    q = [float(dist[order[min(int(np.searchsorted(cdf, p)), len(order)-1)]]) for p in [.5, .9, .95]]
    return {'status': 'SCORED_GEOMETRIC_DISCREPANCY_ONLY', 'samples': len(dist),
            'weighted_area_mm2': float(weights.sum() * UM_PER_VOX**2 / 1e6),
            'distance_vox_p50_p90_p95': q,
            'distance_um_p50_p90_p95': [v * UM_PER_VOX for v in q],
            'coverage': {str(t): float(weights[dist <= t].sum() / weights.sum()) for t in TOLERANCES}}


def evaluate(domain, candidate, frame, destination):
    domain = Path(domain)
    meta = json.loads((domain / 'reference-domain.json').read_text())
    if sha(domain / 'reference-domain.npz') != meta['domain_sha256']:
        raise ValueError('Frozen reference domain hash differs')
    d = np.load(domain / 'reference-domain.npz', allow_pickle=False)
    report = {'schema': 'iss-patch-evaluation/1', 'candidate': str(candidate),
              'candidate_frame': frame, 'reference_domain_sha256': meta['domain_sha256'],
              'code_sha256': sha(__file__), 'identity_verdict': 'UNKNOWN',
              'registration_accuracy_independently_verified': False,
              'claims': 'Only discrepancy to registered GP reference; no natural sheet-switch truth or held-out generalization'}
    if not Path(candidate).exists():
        report.update(status='NO_PATCH', fixed_target_coverage=0. if meta['target_core_area_mm2'] > 0 else None)
        write_json(destination, report)
        return report
    xyz, valid = load_grid(candidate, frame)
    qmask = all_quad(valid)
    if not qmask.any():
        report.update(status='NO_PATCH', fixed_target_coverage=0. if meta['target_core_area_mm2'] > 0 else None)
        write_json(destination, report)
        return report
    v, f, uv, _ = grid_triangles(xyz, qmask)
    centers, area = face_stats(v, f)
    nonzero = area > 1e-10
    if not nonzero.any():
        report.update(status='NO_PATCH', reason='All candidate triangles degenerate')
        write_json(destination, report)
        return report
    labels, count = ndimage.label(qmask)
    component = labels[uv[:, 0], uv[:, 1]]
    seed_query = surface_distance(d['seed_xyz'][None, :], TriangleSoup(v, f[nonzero]), chunk=1)
    seed_face = np.flatnonzero(nonzero)[seed_query.face_idx[0]]
    seed_component = component[seed_face]
    selected = nonzero & (component == seed_component)
    candidate_soup = TriangleSoup(v, f[selected])
    reference_soup = TriangleSoup(d['vertices'], d['faces'])
    cand_core = selected & core_mask(centers)
    idx = np.flatnonzero(cand_core)
    if not len(idx) or not d['target_core'].any():
        report.update(status='INSUFFICIENT_REFERENCE', candidate_core_samples=len(idx))
        write_json(destination, report)
        return report
    cr = surface_distance(centers[idx], reference_soup, chunk=256)
    reference_boundary = d['boundary'][cr.face_idx]
    # Near reference is a support domain, not a correctness verdict; beyond
    # max predefined tolerance unknown and reported separately.
    supported = (cr.dist <= max(TOLERANCES)) & ~reference_boundary
    report.update(status='SCORED_CONTINUOUS_ONLY', candidate_components=count,
                  seed_attachment_distance_vox=float(seed_query.dist[0]),
                  seed_attachment_status='WITHIN_MAX_TOLERANCE' if seed_query.dist[0] <= max(TOLERANCES) else 'SEED_NOT_ATTACHED',
                  candidate_total_area_mm2=float(area[nonzero].sum()*UM_PER_VOX**2/1e6),
                  candidate_seed_component_area_mm2=float(area[selected].sum()*UM_PER_VOX**2/1e6),
                  candidate_core_area_mm2=float(area[idx].sum()*UM_PER_VOX**2/1e6),
                  candidate_unknown_area_mm2=float(area[idx][~supported].sum()*UM_PER_VOX**2/1e6),
                  candidate_reference_boundary_area_mm2=float(area[idx][reference_boundary].sum()*UM_PER_VOX**2/1e6),
                  candidate_to_all_reference=summary(cr.dist, area[idx]),
                  candidate_within_reference_support=summary(cr.dist[supported], area[idx][supported]),
                  quadrature='One deterministic centroid per triangle, physical area weight; not exact continuous area integrals')
    target_face = d['component'] == int(d['target_component'])
    competitor_face = (d['component'] > 0) & ~target_face
    target_query = surface_distance(centers[idx], TriangleSoup(d['vertices'], d['faces'][target_face]), chunk=256)
    report['candidate_to_seed_reference_branch'] = summary(target_query.dist, area[idx])
    if competitor_face.any():
        other = surface_distance(centers[idx], TriangleSoup(d['vertices'], d['faces'][competitor_face]), chunk=256)
        report['nearest_branch_margin_vox'] = (other.dist-target_query.dist).tolist()
        report['competitor_nearer_area_mm2'] = float(area[idx][other.dist < target_query.dist].sum()*UM_PER_VOX**2/1e6)
        report['competitor_nearer_is_sheet_switch_verdict'] = False
    for name in ('target_core', 'target_unseeded'):
        mask = d[name]
        result = surface_distance(d['centers'][mask], candidate_soup, chunk=256)
        report['reference_to_candidate_' + name] = summary(result.dist, d['area'][mask])
    report['candidate_files_sha256'] = {f: sha(Path(candidate)/f) for f in ('x.tif','y.tif','z.tif')}
    sample_path = Path(destination).with_suffix('.samples.npz')
    write_npz(sample_path, candidate_face_idx=idx, candidate_face_uv=uv[idx],
              nearest_reference_face=cr.face_idx, nearest_reference_uv=d['face_uv'][cr.face_idx],
              distance_vox=cr.dist, candidate_area_vox2=area[idx], supported=supported)
    report['samples_sha256'] = sha(sample_path)
    write_json(destination, report)
    return report


def selftest():
    vertices = np.array([[0.,0.,0.],[10.,0.,0.],[0.,10.,0.],[10.,10.,0.]])
    faces = np.array([[0,1,2],[1,3,2]])
    soup = TriangleSoup(vertices, faces)
    points = np.array([[2.,2.,0.],[2.,2.,3.],[12.,0.,0.]])
    actual = surface_distance(points, soup, chunk=1).dist
    np.testing.assert_allclose(actual, [0.,3.,2.], atol=1e-12)
    centers, area = face_stats(vertices, faces)
    np.testing.assert_allclose(area.sum(), 100.)
    clean = surface_distance(centers, soup, chunk=1).dist
    np.testing.assert_allclose(clean, 0., atol=1e-12)
    moved = TriangleSoup(vertices + [0.,0.,3.], faces)
    assert summary(surface_distance(centers, moved, chunk=1).dist, area)['coverage']['2.0'] == 0.
    assert summary(surface_distance(centers, moved, chunk=1).dist, area)['coverage']['4.0'] == 1.
    # Same fixed reference denominator, candidate loses half of surface.
    reduced = TriangleSoup(vertices, faces[:1])
    assert summary(surface_distance(centers, reduced, chunk=1).dist, area)['coverage']['0.5'] == .5
    assert summary(np.array([]), np.array([]))['status'] == 'INSUFFICIENT_REFERENCE'
    with tempfile.TemporaryDirectory(prefix='patch-eval-interface-') as temporary:
        root = Path(temporary)
        candidate = root / 'candidate'
        candidate.mkdir()
        grid = np.array([[[200.,200.,200.],[210.,200.,200.]],
                         [[200.,210.,200.],[210.,210.,200.]]], np.float32)

        def save_candidate(points):
            for i, name in enumerate('xyz'):
                tifffile.imwrite(candidate / (name + '.tif'), points[..., i])

        v, f, uv, _ = grid_triangles(grid, np.ones((1,1), bool))
        centers, area = face_stats(v, f)
        write_npz(root/'reference-domain.npz', vertices=v, faces=f,
                  centers=centers, area=area, component=np.ones(2, int),
                  boundary=np.zeros(2, bool), target_core=np.ones(2, bool),
                  target_unseeded=np.ones(2, bool), target_component=np.array(1),
                  seed_xyz=np.array([205.,205.,200.]), face_uv=uv)
        write_json(root/'reference-domain.json', {
            'domain_sha256':sha(root/'reference-domain.npz'), 'target_core_area_mm2':.009216})
        save_candidate(grid)
        null = evaluate(root, candidate, 'local_l2_xyz', root/'null.json')
        assert null['reference_to_candidate_target_core']['coverage']['0.5'] == 1.
        save_candidate(grid + [0.,0.,3.])
        shifted = evaluate(root, candidate, 'local_l2_xyz', root/'shifted.json')
        assert shifted['reference_to_candidate_target_core']['coverage']['2.0'] == 0.
        assert shifted['reference_to_candidate_target_core']['coverage']['4.0'] == 1.
        smaller = grid.copy()
        smaller[..., 0] = 200 + (smaller[..., 0] - 200)/2
        save_candidate(smaller)
        shrink = evaluate(root, candidate, 'local_l2_xyz', root/'shrink.json')
        assert shrink['reference_to_candidate_target_core']['coverage']['0.5'] == .5
        save_candidate(np.full_like(grid, -1))
        empty = evaluate(root, candidate, 'local_l2_xyz', root/'empty.json')
        assert empty['status'] == 'NO_PATCH'
        missing = evaluate(root, root/'missing', 'local_l2_xyz', root/'missing.json')
        assert missing['status'] == 'NO_PATCH'
        z_zero = grid.copy()
        z_zero[...,2] = 0
        save_candidate(z_zero)
        _, validity = load_grid(candidate, 'local_l2_xyz')
        assert validity.all(), 'Local z=0 must remain valid'
        full = (grid + ORIGIN_XYZ) * 4
        save_candidate(full)
        restored, validity = load_grid(candidate, 'fullres_xyz')
        np.testing.assert_allclose(restored, grid)
        assert validity.all()
    return {'status':'PASS_INTERFACE_ONLY', 'checks':['Exact plane/offplane/outside distances', 'Physical triangle area', 'Null geometry', 'Translated plane tolerance sensitivity', 'Fixed-denominator crop penalty', 'Empty score not pass', 'End-to-end TIFF null/shift/shrink/invalid/missing candidates', 'Local z=0 remains valid', 'Fullres XYZ transformation roundtrip'],
            'scientific_result': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','evaluate','selftest'])
    parser.add_argument('--reference')
    parser.add_argument('--out', required=True)
    parser.add_argument('--domain')
    parser.add_argument('--candidate')
    parser.add_argument('--candidate-frame', choices=['local_l2_xyz','global_l2_xyz','fullres_xyz'])
    args = parser.parse_args()
    start = time.monotonic()
    if args.action == 'prepare':
        result = prepare(args.reference, args.out)
    elif args.action == 'evaluate':
        if not args.candidate_frame:
            parser.error('--candidate-frame required; no guessed coordinate frame')
        result = evaluate(args.domain, args.candidate, args.candidate_frame, args.out)
    else:
        result = selftest()
        write_json(args.out, result)
    print(json.dumps({'action':args.action,'status':result['status'],
                      'elapsed_s':time.monotonic()-start,
                      'ru_maxrss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'platform':sys.platform}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
