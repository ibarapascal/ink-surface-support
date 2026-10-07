"""Same qualified reference geometry, in RAM only, for fixed V3."""
import gc,json,hashlib
from pathlib import Path
import numpy as np
import qualify_disjoint_references as original
from evaluate_expanded_references import domain_geometry
from evaluate_patch import sha
from evaluate_references import numeric_equal
W=Path('__HISTORICAL_WORK_ROOT__/iss-independent-region-prequalification-20261007')
SOURCE_SHA='47cd8aa602281946820e703783d3d58e451df5dbb17007edd67737ce803c90d7'
PLAN_SHA='8a759cff5dfbe117dc7536c61319c06d25fa2e25ac54bd6f055df81ef5dd2668'
CAP=1610612736;UNION=268435456

def array_sha(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()

def load(asset,consumer_bound_bytes):
 assert consumer_bound_bytes<=UNION,'Actualconsumer memory bound exceeds preregistered256MiB allowance'
 assert sha(W/'sources.json')==SOURCE_SHA and sha(W/'plan.json')==PLAN_SHA
 locks=json.loads((W/'source-lock.json').read_text());assert all(sha(p)==h for p,h in locks.items())
 assert sha(original.__file__)==locks['__HISTORICAL_WORK_ROOT__/iss-disjoint-references-20261007/qualify_disjoint_references.py']
 plan=json.loads((W/'plan.json').read_text());s=next(x for x in json.loads((W/'sources.json').read_text())['assets'] if x['asset']==asset)
 prior=json.loads((W/'qualification.json').read_text());arrays,q=original.read_source(s)
 box=np.asarray(plan['input_box_global_L2_zyx'],float)[:,::-1];lo,hi=box[0]-32,box[1]+32
 n=2*sum(len(r) for r,c in original.broad_rows(arrays,q,lo,hi));bound=128*1024**2+2*sum(x.nbytes for x in arrays)+1000*n+consumer_bound_bytes
 assert bound<=CAP,'Actualsource/consumer bound exceeds1.5GiB; nohalo/domain reduction'
 v,f,uv,uniq=original.geometry(arrays,q,(lo,hi));centers,area,labels,components,boundary=domain_geometry(v,f,lo,hi);keep=area>1e-10;f=f[keep];uv=uv[keep]
 data={'vertices_global_l2_xyz':v,'faces':f};report={'asset':asset,'source_hashes':s['hashes'],'source_manifest_sha256':SOURCE_SHA,'plan_sha256':PLAN_SHA,'halo_broadphase_faces':n,'source_plus_consumer_bound_bytes':bound,'actual_consumer_bound_bytes':consumer_bound_bytes,'halo32_box_global_l2_xyz':[lo.tolist(),hi.tolist()],'array_sha256':{'vertices_global_l2_xyz':array_sha(v),'faces':array_sha(f),'face_source_grid_rc':array_sha(uv),'source_vertex_grid_linear_ids':array_sha(uniq)},'domains':{},'new_NPZ_bytes':0}
 del centers,area,labels,components,boundary,keep,uv,uniq,arrays,q;gc.collect()
 for domain,bb in plan['domains_global_L2_xyz'].items():
  blo,bhi=np.asarray(bb,float);c,a,l,comp,b=domain_geometry(v,f,blo,bhi);active=a>1e-10;use=active&~b
  old=next(x for x in prior['rows'] if x['asset']==asset and x['domain']==domain)
  assert int(active.sum())==old['positive_clipped_faces'] and int(use.sum())==old['boundary_excluded_faces'] and len(comp)==old['component_count']
  actualarea=float(a[use].sum()*9.6**2/1e6);delta=actualarea-old['boundary_excluded_area_mm2']
  # Same arithmetic definitions; tiny reduction-order differences reported, not a scientific tolerance.
  numeric_equal(actualarea,old['boundary_excluded_area_mm2'],'existing numerical equality helper/reference denominator')
  for suffix,arr in [('centroids_global_l2_xyz',c),('clipped_area_vox2',a),('component',l),('boundary',b)]:data[domain+'_'+suffix]=arr;report['array_sha256'][domain+'_'+suffix]=array_sha(arr)
  report['domains'][domain]={'box_global_l2_xyz':bb,'boundary_excluded_area_mm2':actualarea,'prequalified_denominator_delta_mm2':delta,'positive_area_faces':int(active.sum()),'boundary_excluded_faces':int(use.sum()),'component_count':len(comp)}
 for name,h in s['hashes'].items():assert sha(Path(s['source_root'])/name)==h
 return data,report
