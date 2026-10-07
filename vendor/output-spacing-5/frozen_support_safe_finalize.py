"""Development prototype: support-constrained coarse mask finalization.

Only deletes coarse vertices; no XYZ movement, growth/fine-clean/fold changes.
No geometric correctness or upstream bug is implied by support validity.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
EPS=1e-9

def axes(fine_shape,factor):
 if not np.isfinite(factor) or factor<=0:raise ValueError('factor must be finite positive')
 return tuple(np.arange(0,n-1+EPS,factor) for n in fine_shape)

def _upper(values,n):
 lo=np.floor(values).astype(np.int64)
 return np.where(values-lo>EPS,np.minimum(lo+1,n-1),lo)

def footprint_bounds(fine_shape,factor):
 rows,cols=axes(fine_shape,factor)
 return (np.floor(rows[:-1]).astype(np.int64),_upper(rows[1:],fine_shape[0]),np.floor(cols[:-1]).astype(np.int64),_upper(cols[1:],fine_shape[1]))

def full_support(fine_valid,factor):
 fine_valid=np.asarray(fine_valid,dtype=bool);r0,r1,c0,c1=footprint_bounds(fine_valid.shape,factor)
 prefix=np.pad((~fine_valid).astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
 counts=prefix[(r1+1)[:,None],(c1+1)[None,:]]-prefix[r0[:,None],(c1+1)[None,:]]-prefix[(r1+1)[:,None],c0[None,:]]+prefix[r0[:,None],c0[None,:]]
 return counts==0

def _complete(mask):return mask[:-1,:-1]&mask[1:,:-1]&mask[:-1,1:]&mask[1:,1:]

def _quad_area(grid,voxel_um):
 a=grid[:-1,:-1].astype(float);b=grid[:-1,1:].astype(float);c=grid[1:,:-1].astype(float);d=grid[1:,1:].astype(float)
 return .5*(np.linalg.norm(np.cross(c-a,b-a),axis=-1)+np.linalg.norm(np.cross(b-d,c-d),axis=-1))*(float(voxel_um)**2/1e6)

def brute_support_check(fine_valid,factor,keep):
 """Independent per-quad slices, deliberately no prefix sums/helper bounds."""
 nv,nu=np.asarray(fine_valid).shape;rows=np.arange(0,nv-1+1e-9,factor);cols=np.arange(0,nu-1+1e-9,factor);checked=0;bad=[]
 for r,c in zip(*np.where(_complete(keep))):
  lo_r=int(np.floor(rows[r]));lo_c=int(np.floor(cols[c]));end_r=float(rows[r+1]);end_c=float(cols[c+1]);hi_r=int(np.floor(end_r));hi_c=int(np.floor(end_c))
  if end_r-hi_r>1e-9:hi_r=min(hi_r+1,nv-1)
  if end_c-hi_c>1e-9:hi_c=min(hi_c+1,nu-1)
  checked+=1
  if not bool(np.asarray(fine_valid)[lo_r:hi_r+1,lo_c:hi_c+1].all()):bad.append([int(r),int(c)])
 return {'checked_quads':checked,'bad_quads':bad,'passed':not bad}

def finalize(coarse_xyz,fine_valid,factor,voxel_um=9.6):
 grid=np.asarray(coarse_xyz);fine=np.asarray(fine_valid,dtype=bool)
 if grid.ndim!=3 or grid.shape[-1]!=3 or grid.shape[0]*grid.shape[1]>250000:raise ValueError('bounded XYZ grid required')
 rows,cols=axes(fine.shape,factor)
 if grid.shape[:2]!=(len(rows),len(cols)):raise ValueError('coarse/fine/factor lattice mismatch')
 valid=grid[...,0]>=0;keep=valid.copy();safe=full_support(fine,factor);area=_quad_area(grid,voxel_um);initial_complete=_complete(keep);initial_bad=initial_complete&~safe;deleted=[]
 while True:
  complete=_complete(keep);bad=complete&~safe
  if not bad.any():break
  corners=set()
  for r,c in zip(*np.where(bad)):
   corners.update([(int(r),int(c)),(int(r+1),int(c)),(int(r),int(c+1)),(int(r+1),int(c+1))])
  choices=[]
  for r,c in sorted(corners):
   loss=0.
   for qr in (r-1,r):
    for qc in (c-1,c):
     if 0<=qr<safe.shape[0] and 0<=qc<safe.shape[1] and safe[qr,qc] and complete[qr,qc]:loss+=float(area[qr,qc])
   choices.append((loss,r,c))
  loss,r,c=min(choices);keep[r,c]=False;deleted.append({'vertex':[r,c],'destroyed_adjacent_safe_quad_area_mm2':loss})
 before_component=keep.copy();labels,count=ndimage.label(keep)
 if count>1:
  sizes=np.bincount(labels.ravel());sizes[0]=0;keep=labels==sizes.argmax()
 out=grid.copy();out[~keep]=-1.
 assert not np.any(keep&~valid);assert np.array_equal(out[keep],grid[keep])
 independent=brute_support_check(fine,factor,keep)
 if not independent['passed']:raise AssertionError(independent)
 diagnostics={'prototype':'support-safe coarse finalization','factor':float(factor),'epsilon':EPS,'voxel_um':float(voxel_um),'initial_valid_vertices':int(valid.sum()),'initial_complete_quads':int(initial_complete.sum()),'initial_bad_footprint_quads':int(initial_bad.sum()),'deleted_greedily':deleted,'greedy_rule':'global min sum of adjacent current complete safequad physical areas; exact numeric tie then row,column lex','largest_component_rule':'scipy.ndimage.label default4connected; retain first argmax by original source rule','components_before_largest':int(count),'vertices_removed_by_component_rule':int(before_component.sum()-keep.sum()),'final_valid_vertices':int(keep.sum()),'final_complete_quads':int(_complete(keep).sum()),'no_invalid_restored':True,'retained_xyz_unchanged':True,'independent_brute_support':independent,'limitations':'full fine support is not geometric/sheet identity truth; arbitrary-mask examples do not prove a default40clean bug'}
 return out,diagnostics
