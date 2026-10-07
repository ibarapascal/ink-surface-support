"""Exact point-to-mesh distance, CPU-only.

Two-stage query: a cKDTree on triangle centroids proposes candidates, then the
exact closest-point-on-triangle (same barycentric region logic as villa's
``Patch.project``) decides. The candidate radius is derived from a per-mesh
bound (max centroid-to-vertex distance), which makes the result exact, not
approximate: any triangle that could beat the current best upper bound has its
centroid within ``upper_bound + r_max``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

_EPS = 1e-12


def closest_point_on_triangles(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Element-wise closest point to ``p[i]`` on triangle ``(a[i], b[i], c[i])``.

    All inputs are (K, 3) float arrays; returns (K, 3). Implements the standard
    Voronoi-region case analysis (Ericson, Real-Time Collision Detection),
    matching villa's torch implementation.
    """
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    out = np.empty_like(p)
    done = np.zeros(p.shape[0], dtype=bool)

    def assign(mask: np.ndarray, values: np.ndarray) -> None:
        mask = mask & ~done
        out[mask] = values[mask]
        done[mask] = True

    # Vertex regions
    assign((d1 <= 0) & (d2 <= 0), a)
    assign((d3 >= 0) & (d4 <= d3), b)
    assign((d6 >= 0) & (d5 <= d6), c)

    # Edge AB
    vc = d1 * d4 - d3 * d2
    v_ab = d1 / np.where(np.abs(d1 - d3) < _EPS, _EPS, d1 - d3)
    assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v_ab[:, None] * ab)

    # Edge AC
    vb = d5 * d2 - d1 * d6
    w_ac = d2 / np.where(np.abs(d2 - d6) < _EPS, _EPS, d2 - d6)
    assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w_ac[:, None] * ac)

    # Edge BC
    va = d3 * d6 - d5 * d4
    denom_bc = (d4 - d3) + (d5 - d6)
    w_bc = (d4 - d3) / np.where(np.abs(denom_bc) < _EPS, _EPS, denom_bc)
    assign((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w_bc[:, None] * (c - b))

    # Face region
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < _EPS, _EPS, denom)
    v = vb / denom
    w = vc / denom
    assign(np.ones_like(done), a + v[:, None] * ab + w[:, None] * ac)
    return out


@dataclass
class TriangleSoup:
    """A triangle mesh prepared for repeated exact distance queries."""

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (M, 3) int
    _centroids: np.ndarray | None = field(default=None, repr=False)
    _tree: cKDTree | None = field(default=None, repr=False)
    _r_max: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3), got {self.faces.shape}")
        if len(self.faces) == 0:
            raise ValueError("empty triangle soup")

    @property
    def a(self) -> np.ndarray:
        return self.vertices[self.faces[:, 0]]

    @property
    def b(self) -> np.ndarray:
        return self.vertices[self.faces[:, 1]]

    @property
    def c(self) -> np.ndarray:
        return self.vertices[self.faces[:, 2]]

    @property
    def centroids(self) -> np.ndarray:
        if self._centroids is None:
            self._centroids = (self.a + self.b + self.c) / 3.0
        return self._centroids

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.centroids)
        return self._tree

    @property
    def r_max(self) -> float:
        """Max distance from any triangle's centroid to its vertices."""
        if self._r_max is None:
            cen = self.centroids
            self._r_max = float(
                np.sqrt(
                    max(
                        ((self.a - cen) ** 2).sum(-1).max(),
                        ((self.b - cen) ** 2).sum(-1).max(),
                        ((self.c - cen) ** 2).sum(-1).max(),
                    )
                )
            )
        return self._r_max

    def face_normals(self) -> np.ndarray:
        """Unit normals per face (orientation as given by the face winding)."""
        n = np.cross(self.b - self.a, self.c - self.a)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / np.maximum(norm, _EPS)


@dataclass
class DistanceResult:
    dist: np.ndarray  # (N,)
    closest: np.ndarray  # (N, 3)
    face_idx: np.ndarray  # (N,) int64


def surface_distance(
    points: np.ndarray, soup: TriangleSoup, k_seed: int = 8, chunk: int = 65536
) -> DistanceResult:
    """Exact distance from each point to the nearest point of the soup."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    n = points.shape[0]
    out_dist = np.empty(n, dtype=np.float64)
    out_closest = np.empty((n, 3), dtype=np.float64)
    out_face = np.empty(n, dtype=np.int64)

    k_seed = min(k_seed, len(soup.faces))
    for start in range(0, n, chunk):
        pts = points[start : start + chunk]
        m = pts.shape[0]

        # Stage 1: upper bound from the k nearest centroids, evaluated exactly.
        _, seed_idx = soup.tree.query(pts, k=k_seed)
        seed_idx = seed_idx.reshape(m, -1)
        flat_pts = np.repeat(pts, seed_idx.shape[1], axis=0)
        flat_faces = seed_idx.ravel()
        closest = closest_point_on_triangles(
            flat_pts, soup.a[flat_faces], soup.b[flat_faces], soup.c[flat_faces]
        )
        d = np.linalg.norm(closest - flat_pts, axis=-1).reshape(m, -1)
        best_pos = d.argmin(axis=1)
        rows = np.arange(m)
        ub = d[rows, best_pos]
        best_face = seed_idx[rows, best_pos]
        best_closest = closest.reshape(m, -1, 3)[rows, best_pos]

        # Stage 2: any strictly better triangle has its centroid within ub + r_max.
        radii = ub + soup.r_max + 1e-9
        cand = soup.tree.query_ball_point(pts, r=radii)
        lens = np.fromiter((len(c) for c in cand), dtype=np.int64, count=m)
        need = lens > 0
        if need.any():
            group = np.repeat(np.arange(m), lens)
            flat_faces = np.concatenate([np.asarray(c, dtype=np.int64) for c in cand])
            flat_pts = pts[group]
            closest = closest_point_on_triangles(
                flat_pts, soup.a[flat_faces], soup.b[flat_faces], soup.c[flat_faces]
            )
            d = np.linalg.norm(closest - flat_pts, axis=-1)
            # Segment argmin: stable sort by (group, distance), keep first per group.
            order = np.lexsort((d, group))
            group_sorted = group[order]
            first = np.searchsorted(group_sorted, np.arange(m), side="left")
            starts_valid = first[need.nonzero()[0]]
            winner = order[starts_valid]
            better = d[winner] < ub[need]
            upd = need.nonzero()[0][better]
            ub[upd] = d[winner][better]
            best_face[upd] = flat_faces[winner][better]
            best_closest[upd] = closest[winner][better]

        sl = slice(start, start + m)
        out_dist[sl] = ub
        out_closest[sl] = best_closest
        out_face[sl] = best_face
    return DistanceResult(out_dist, out_closest, out_face)
