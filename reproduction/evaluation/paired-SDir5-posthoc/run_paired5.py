"""Exact2same-source original5/support5 SDir pairs. Nooptimizer/GP/CT."""
import json,os,time,resource,gc
from pathlib import Path
import numpy as np
import torch
import model as native_model
import opt_loss_flatten as native_loss
import dtypes
from evaluate_flatten_consumer import stats
from evaluate_patch import all_quad,sha
W=Path(__file__).parent;start=time.monotonic();p=json.loads((W/'protocol.json').read_text());torch.set_num_threads(1)
assert os.environ['LASAGNA_MAX_PRECISION_FLOAT']=='32' and dtypes.torch_float_hi==torch.float32
for module,name in [(native_model,'model.py'),(native_loss,'opt_loss_flatten.py'),(dtypes,'dtypes.py')]:assert sha(module.__file__)==p['source_hashes'][name]
def physical_area(x):
 a,b,c,d=x[:-1,:-1].astype(float),x[1:,:-1].astype(float),x[:-1,1:].astype(float),x[1:,1:].astype(float)
 return (np.linalg.norm(np.cross(b-a,c-a),axis=2)+np.linalg.norm(np.cross(d-b,c-b),axis=2))/2

def read(rec):
 assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
 for name,h in rec['source_files_sha256'].items():assert sha(Path(rec['source_path'])/name)==h
 s=torch.load(rec['checkpoint'],map_location='cpu',weights_only=False);xyz=s['flatten_source_xyz'];valid=s['flatten_source_valid'].numpy().astype(bool);sc=s['flatten_source_cell_valid'];assert xyz.numel()<750000 and xyz.dtype==torch.float32
 ps=[s[k] for k in sorted((k for k in s if k.startswith('flatten_map_ms.')),key=lambda k:int(k.rsplit('.',1)[1]))];uv=native_model.Model3D._integrate_pyramid_3d(ps,pyramid_d=False)[:,0].permute(1,2,0).contiguous();metric=s.get('flatten_source_metric')
 if metric is None or not metric.numel():metric=native_model.Model3D._flatten_source_metric(xyz)
 step=s['flatten_target_step'];assert float(step)==5.0 and uv.dtype==torch.float32
 _,lm,_,eligible=native_loss._flatten_forward_sdir_core(uv,metric,sc,step,1e-8)
 return {'xyz':xyz.numpy(),'valid':valid,'q':all_quad(valid),'cell_valid':sc.numpy().astype(bool),'metric':metric.numpy(),'loss':lm.numpy(),'eligible':eligible.numpy().astype(bool)}

cases=[];pooled_old=[];pooled_new=[];pooled_weights=[];fixed_area=0.;added_area=0.
for pair in p['pairs']:
 a=read(pair['arms']['original5']);b=read(pair['arms']['support5']);weights=physical_area(a['xyz']);common=a['q'];denom=float(weights[common].sum()*9.6**2/1e6);fixed_area+=denom
 row={'case':pair['case'],'region':pair['region'],'source_fixed_quad_count':int(common.sum()),'fixed_original_area_mm2':denom,'checkpoint_provenance':pair['arms'],'exact_shared_source_shape':a['xyz'].shape==b['xyz'].shape}
 if not row['exact_shared_source_shape']:
  row['status']='NO_EXACT_COMMON_GEOMETRY';cases.append(row);continue
 retained=bool(np.all(b['valid'][a['valid']]));same=retained and np.array_equal(a['xyz'][a['valid']],b['xyz'][a['valid']]);row.update(old_valid_vertices_retained=retained,exact_shared_XYZ=same)
 if not same:row['status']='NO_EXACT_COMMON_GEOMETRY';cases.append(row);continue
 assert np.array_equal(a['metric'][common],b['metric'][common]),'Metricdifference despite samecellgeometry'
 use=common&a['eligible']&b['eligible'];wa=weights[use];la=a['loss'][use];lb=b['loss'][use];delta=lb.astype(float)-la.astype(float)
 row.update(status='EXACT_COMMON_SOURCE_PAIRED',paired_quad_count=int(use.sum()),paired_area_mm2=float(wa.sum()*9.6**2/1e6),original_ineligible_fixed_area_mm2=float(weights[common&~a['eligible']].sum()*9.6**2/1e6),support_ineligible_fixed_area_mm2=float(weights[common&~b['eligible']].sum()*9.6**2/1e6),original_common=stats(la,wa),support_common=stats(lb,wa),paired_delta_support_minus_original=stats(delta,wa))
 if len(wa):
  row['paired_weighted_mean_delta']=float(np.sum(delta*wa)/wa.sum());pooled_old.append(la);pooled_new.append(lb);pooled_weights.append(wa)
 else:row['paired_weighted_mean_delta']=None
 added=b['q']&~common;bw=physical_area(b['xyz']);usable_added=added&b['eligible'];aa=float(bw[added].sum()*9.6**2/1e6);added_area+=aa
 row['added_support_region']={'quad_count':int(added.sum()),'area_mm2':aa,'ineligible_area_mm2':float(bw[added&~b['eligible']].sum()*9.6**2/1e6),'SDir':stats(b['loss'][usable_added],bw[usable_added]),'not_mixed_into_common_denominator':True};cases.append(row)
 del a,b,weights,common,wa,la,lb,delta,bw;gc.collect()
assert len(cases)==2
if pooled_weights:
 wa=np.concatenate(pooled_weights);la=np.concatenate(pooled_old);lb=np.concatenate(pooled_new);pooled={'original_common':stats(la,wa),'support_common':stats(lb,wa),'paired_delta_support_minus_original':stats(lb.astype(float)-la.astype(float),wa),'fixed_original_area_mm2_all2':fixed_area,'total_added_support_area_mm2':added_area}
else:pooled={'status':'NO_VALID_WEIGHTED_SAMPLES','fixed_original_area_mm2_all2':fixed_area}
result={'status':'ALL2_PAIRS_ACCOUNTED','cases':cases,'pooled':pooled,'exact_geometry_pairs':sum(r['status']=='EXACT_COMMON_SOURCE_PAIRED' for r in cases),'unmatched_pairs':sum(r['status']!='EXACT_COMMON_SOURCE_PAIRED' for r in cases),'no_5_grid_matching':True,'no_CT_GP_optimizer_download':True,'numeric_source_hashes':p['source_hashes'],'protocol_sha256':sha(W/'protocol.json'),'driver_sha256':sha(__file__),'native_eps':1e-8,'native_precision':'float32; geometricareaweightsfloat64, same aspreviousdiagnostic','seconds':time.monotonic()-start,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':['2pairedcase outputs, not independent2acquisitions.','Commonoriginal5area fixed for both arms;addedregions separatelyreported.','Exactfloat32nativeSDir signedtinynegative values preserved; numericalsign notnewqualitygate.','Cellmetric distortion notCT/fiberidentity orreadability.','No threshold tuned afterresults; median/tails aredescriptive, noPvalue.']}
text=json.dumps(result,indent=2,allow_nan=False)+'\n';assert len(text.encode())<2*1024**2;(W/'result.json').write_text(text);print(json.dumps({'status':result['status'],'exact_pairs':result['exact_geometry_pairs'],'seconds':result['seconds'],'bytes':len(text.encode())}))
