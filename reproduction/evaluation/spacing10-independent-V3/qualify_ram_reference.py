"""One bounded source-only qualification. No M7, candidate, scoring, or NPZ writes."""
import gc,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import qualify_disjoint_references as ref
from evaluate_expanded_references import domain_geometry
from expanded_references import digest,write_json
W=Path(__file__).parent
SOURCE_MANIFEST=Path('__HISTORICAL_WORK_ROOT__/iss-independent-region-prequalification-20261007/sources.json')
OLD_MANIFEST=Path('__HISTORICAL_DATA_ROOT__/p4_validation_references_20261007/reference-manifest.json')
LOCK=Path('__HISTORICAL_WORK_ROOT__/iss-independent-region-prequalification-20261007/source-lock.json')
EXPECTED_SOURCES='47cd8aa602281946820e703783d3d58e451df5dbb17007edd67737ce803c90d7'
EXPECTED_OLD='583c62bffd3bb02360556539b8c6d3280604f92d56aa5b02bf728d1ba9c76698'
RAM=2*1024**3;FINAL_RAM=1610612736
# No consumer files are opened; reserve an explicit future union allowance.
FUTURE_UNION_ALLOWANCE=256*1024**2
HALO=(np.array([3232.,4192.,12064.]),np.array([4832.,5792.,13664.]))
start=time.monotonic();assert digest(SOURCE_MANIFEST)==EXPECTED_SOURCES and digest(OLD_MANIFEST)==EXPECTED_OLD
source_locks=json.loads(LOCK.read_text());assert all(digest(Path(p))==h for p,h in source_locks.items())
assert digest(Path(ref.__file__))==source_locks['__HISTORICAL_WORK_ROOT__/iss-disjoint-references-20261007/qualify_disjoint_references.py']
sources=json.loads(SOURCE_MANIFEST.read_text())['assets'];old=json.loads(OLD_MANIFEST.read_text())
pre=[]
for s in sources:
 arrays,q=ref.read_source(s);count=2*sum(len(r) for r,c in ref.broad_rows(arrays,q,*HALO))
 base=128*1024**2+2*sum(x.nbytes for x in arrays);estimate=base+1000*count
 pre.append({'asset':s['asset'],'halo32_broadphase_faces':count,'source_xyz_nbytes':sum(x.nbytes for x in arrays),'reference_peak_bound_bytes':estimate,'future_union_allowance_bytes':FUTURE_UNION_ALLOWANCE,'combined_predicted_scoring_peak_bytes':estimate+FUTURE_UNION_ALLOWANCE,'fits_final_1p5GiB_conservative_bound':estimate+FUTURE_UNION_ALLOWANCE<=FINAL_RAM,'optional_original_npz_upper_bytes_not_written':300*count+65536,'halo_global_L2_xyz':[x.tolist() for x in HALO]})
 for n,h in s['hashes'].items():assert digest(Path(s['source_root'])/n)==h
 del arrays,q;gc.collect()
selected=max((x for x in old['assets'] if x['region']=='V1'),key=lambda x:x['geometry']['bytes']);s=next(x for x in sources if x['asset']==selected['asset'])
assert digest(Path(selected['geometry']['path']))==selected['geometry']['sha256'];arrays,q=ref.read_source(s)
lo,hi=np.asarray(selected['domains']['halo32']['box_global_l2_xyz'],float)
count=2*sum(len(r) for r,c in ref.broad_rows(arrays,q,lo,hi));estimate=128*1024**2+2*sum(x.nbytes for x in arrays)+1000*count+selected['geometry']['bytes']
assert estimate<=RAM,'Old equivalence allocation exceeds declared2GiB; do not reduce domain'
v,faces,uv,uniq=ref.geometry(arrays,q,(lo,hi));hc,ha,hl,hcomp,hboundary=domain_geometry(v,faces,lo,hi);keep=ha>1e-10;faces=faces[keep];uv=uv[keep]
del hc,ha,hl,hcomp,hboundary,keep;gc.collect()
checks=[]
def compare(name,a,npz):
 b=npz[name];assert a.shape==b.shape and a.dtype==b.dtype,(name,a.shape,b.shape,a.dtype,b.dtype)
 ah=hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest();bh=hashlib.sha256(memoryview(np.ascontiguousarray(b)).cast('B')).hexdigest()
 equal=np.array_equal(a,b,equal_nan=True);assert equal and ah==bh,('Not byte-equivalent',name)
 checks.append({'field':name,'shape':list(a.shape),'dtype':str(a.dtype),'array_equal':bool(equal),'byte_sha256_equal':True,'sha256':ah})
 del b
with np.load(selected['geometry']['path'],allow_pickle=False) as z:
 for name,a in [('vertices_global_l2_xyz',v),('faces',faces),('face_source_grid_rc',uv),('source_vertex_grid_linear_ids',uniq),('source_shape',np.array(s['shape']))]:compare(name,a,z)
 for domain in ['B1408','A384']:
  blo,bhi=np.asarray(selected['domains'][domain]['box_global_l2_xyz'],float);centers,areas,labels,comps,boundary=domain_geometry(v,faces,blo,bhi)
  for suffix,a in [('centroids_global_l2_xyz',centers),('clipped_area_vox2',areas),('component',labels),('boundary',boundary)]:compare(domain+'_'+suffix,a,z)
  oldareas=z[domain+'_clipped_area_vox2'];oldboundary=z[domain+'_boundary'];active=areas>1e-10;oldactive=oldareas>1e-10
  assert np.array_equal(active,oldactive) and np.array_equal(active&~boundary,oldactive&~oldboundary)
  checks.append({'domain':domain,'active_query_mask_equal':True,'primary_boundary_excluded_mask_equal':True,'active_faces':int(active.sum()),'inactive_faces_preserved':int((~active).sum()),'boundary_faces':int((active&boundary).sum()),'reference_denominator_mm2':float(areas[active&~boundary].sum()*9.6**2/1e6)})
  del centers,areas,labels,comps,boundary,oldareas,oldboundary,active,oldactive;gc.collect()
for n,h in s['hashes'].items():assert digest(Path(s['source_root'])/n)==h
assert all(digest(Path(p))==h for p,h in source_locks.items())
result={'status':'SOURCE_ONLY_RAM_REBUILD_EQUIVALENCE_COMPLETE','new_prediction_or_consumer_read':False,'new_reference_NPZ_bytes':0,'V3_halo_preflight':pre,'final_scoring_initial_cap_bytes':FINAL_RAM,'future_union_allowance_is_not_measured':True,'final_scoring_admission':'All3conservativebounds_fit' if all(x['fits_final_1p5GiB_conservative_bound'] for x in pre) else 'STOP_AND_REPORT_BOUND_NO_SHRINK', 'equivalence_selection':{'region':'V1','asset':selected['asset'],'rule':'Largest oldNPZ amongV1 assets as requested; not globalmaxV2','old_NPZ':selected['geometry'],'old_rebuild_peak_bound_including_npz_bytes':estimate},'array_comparisons':checks,'original_sources_unchanged':True,'source_manifest_sha256':EXPECTED_SOURCES,'old_manifest_sha256':EXPECTED_OLD,'implementation_source_locks':source_locks,'driver_sha256':digest(Path(__file__)),'seconds':time.monotonic()-start,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':['No V3 reference materialized, no scores or newmethod decisions.','Finalfutureexportmesh memory still bounded explicitly; allowance notmeasurement.','Arraysbyteequal includes allinactive/zero/boundary rows; no croppedquery substitution.']}
encoded=json.dumps(result,indent=2)+'\n';assert len(encoded.encode())<=5*1024**2;write_json(W/'ram-reference-qualification.json',result);print(json.dumps({'status':result['status'],'final_scoring_admission':result['final_scoring_admission'],'seconds':result['seconds'],'peak':result['self_peak_rss_bytes']}))
