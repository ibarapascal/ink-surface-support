"""One seen-case native option supplement. JSON only; no optimization or NPZ."""
import os,json,time,resource,hashlib
from pathlib import Path
import numpy as np
import torch
import model as native_model
import dtypes
from flatten_cohort_geometry import inspect_export
from evaluate_references import evaluate,numeric_equal
from evaluate_native_consumer import compact
from native32_det_probe import mask,MODEL_SHA
from diagnose_replayed_stages import reference
from evaluate_patch import sha
W=Path(__file__).parent
ARMS=('existing10_to10','support10_to10','existing5_to5','existing10_to5','support10_to5')
SOURCE={'existing10_to10':'existing10_native','support10_to10':'support10_native','existing5_to5':'existing5_native','existing10_to5':'existing10_native','support10_to5':'support10_native'}
MANIFEST_SHA='b084cfc1234a4f0648ac188e5940d2310d3653ffc6a97cda6cac3c8a896ed371'
start=time.monotonic();torch.set_num_threads(1);assert os.environ['LASAGNA_MAX_PRECISION_FLOAT']=='32' and dtypes.torch_float_hi==torch.float32 and dtypes.numpy_float_hi==np.float32
assert sha(native_model.__file__)==MODEL_SHA
oldprotocol=json.loads(Path('__HISTORICAL_WORK_ROOT__/iss-spacing10-followup-20261007/native-consumer-evaluation-protocol.json').read_text())
for n,h in oldprotocol['source_pointer_hashes'].items():assert sha(Path('__HISTORICAL_WORK_ROOT__/iss-patch-eval-20261007')/n)==h
mp=W/'consumer-output-manifest.json';assert sha(mp)==MANIFEST_SHA;m=json.loads(mp.read_text());assert m['all_five_input_output_hash_verified'] is True
assert tuple(r['arm'] for r in m['cases'])==ARMS
pre=Path('__HISTORICAL_DATA_ROOT__/spacing10_followup_20261007/reference/P05-reference.json');expected=json.loads(pre.read_text())
reports={};stages={};qual={};checks={}
for q in m['cases']:
 arm=q['arm'];assert q['case']=='C0-P05' and q['status']=='SUCCESS' and q['returncode']==0
 assert q['source_config_input_hashes_verified_before'] and q['source_config_hashes_verified_after'] and q['input_hashes_unchanged']
 assert q['declared_stage_steps']==[['0','1000'],['1','1000'],['2','1000']]
 assert [x[0] for x in q['stage_completions']]==['0','1','2']
 assert q['effective_runtime_env']['LASAGNA_MAX_PRECISION_FLOAT']=='32' and q['effective_module_dtype']['torch_float_hi']=='torch.float32'
 assert q['expected_native_output_spacing']==q['input']['consumer_output_step'] and q['source_spacing']==q['input']['spacing']
 root=Path(q['checkpoint']).parent;assert Path(q['output_path'])==root/'tifxyz/flatten.tifxyz'
 for f in q['files']:
  p=root/f['relative'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
 for n,h in q['input']['files_sha256'].items():assert sha(Path(q['input']['path'])/n)==h
 assert sha(q['stdout_path'])==q['stdout_sha256'] and sha(q['stderr_path'])==q['stderr_sha256']
 report,ss=inspect_export(q['checkpoint'],q['origin_l2_zyx'])
 state=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);parts=[state[k] for k in sorted((k for k in state if k.startswith('flatten_map_ms.')),key=lambda k:int(k.rsplit('.',1)[1]))]
 uv=native_model.Model3D._integrate_pyramid_3d(parts,pyramid_d=False)[:,0].permute(1,2,0).contiguous();a=mask(uv,state['flatten_source_xyz'],state['flatten_source_cell_valid'],np.float32);b=mask(uv,state['flatten_source_xyz'],state['flatten_source_cell_valid'],np.float64)
 assert int(b.sum())==report['source_export_eligible_quads'];diff=a!=b
 report['native32_vs_legacy64_mask_audit']={'same':bool(np.array_equal(a,b)),'different_cells':int(diff.sum()),'different_rc':np.argwhere(diff).tolist(),'native32_cells':int(a.sum()),'legacy64_cells':int(b.sum()),'model_sha256':MODEL_SHA}
 if diff.any():
  report['legacy64_eligibility_not_native']={k:report[k] for k in ['native_inversion_eligibility','eligible_source_to_export','source_export_eligible_quads']}
  report['native_inversion_eligibility']={'status':'PENDING_NATIVE32_QUALIFICATION'};report['eligible_source_to_export']={'status':'PENDING_NATIVE32_QUALIFICATION'};report['source_export_eligible_quads']=None;report['physical_area_mm2']['source_export_eligible']=None;report['all_stage_geometry']['source_export_eligible']={'status':'PENDING_NATIVE32_QUALIFICATION'};ss.pop('source_export_eligible')
 reports[arm]=report;stages[arm]=ss
 qual[arm]={'source_spacing':q['source_spacing'],'output_step':q['expected_native_output_spacing'],'coordinate_files_verified':True,'seconds':q['seconds'],'maximum_resident_set_size_bytes':q['maximum_resident_set_size_bytes'],'consumer_file_bytes':sum(f['bytes'] for f in q['files']),'new_explicit32_run':True}
 del state,parts,uv,a,b,diff
rv,rf,p,w,l,b,idx,box,info=reference('P05');unique={};keys={}
for arm,ss in stages.items():
 for name,x in ss.items():
  h=hashlib.sha256(x['v'].tobytes()+x['f'].tobytes()).hexdigest();keys[(arm,name)]=h
  if h not in unique:unique[h]=compact(x)
scored=evaluate(unique,rv,rf,p,w,l,b,idx,box)
section={'reference_denominator_mm2':scored['reference_denominator_mm2'],'arms':{arm:{name:scored['arms'][keys[(arm,name)]] for name in ss} for arm,ss in stages.items()}}
for arm in ARMS:
 got=section['arms'][arm]['source_complete'];old=expected['domains'][info['domain']]['arms'][SOURCE[arm]]
 numeric_equal(old['fixed_reference_to_candidate']['covered_area_mm2'],got['fixed_reference_to_candidate']['covered_area_mm2'],'source/referencecoverage')
 for k in ['supported8_mm2','unknown_mm2','clipped_mm2']:numeric_equal(old[k],got[k],'source/'+k)
checks['all_preconsumer_source_metrics_equal']=True
out={'schema':'iss-P05-output-step-seen-supplement-score/1','status':'COMPLETE_SEEN_SUPPLEMENT_EXISTING_OPTION','arms':reports,'qualification':qual,'domains':{info['domain']:section},'reference':info,'checks':checks,'source_reference_score_sha256':sha(pre),'input_consumer_manifest_sha256':MANIFEST_SHA,'source_sha256':sha(__file__),'numerical_core_sha256':{'inspect_export':sha(inspect_export.__code__.co_filename),'evaluate':sha(evaluate.__code__.co_filename),'mask':sha(mask.__code__.co_filename)},'seconds':time.monotonic()-start,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':['SinglealreadyseenP05 existingoption supplement; not independent validation or V3/full64 member.','output_step changes nativeoptimization trajectory and exportgrid; not pure resampling.','Native10/sourcegeometry unchanged for10to5; preconsumer metrics independently reproduced.','SameGP/registration/identity uncertainty. Unknown isnoterror; perstage area gross notcrosspatch unique.','Fixed5newruns no best-arm search/threshold adjustment. Existing option not newmethod.']}
t=json.dumps(out,indent=2,allow_nan=False)+'\n';assert len(t.encode())<=3*1024**2;(W/'evaluation.json').write_text(t);print(json.dumps({'status':out['status'],'seconds':out['seconds'],'jsonbytes':len(t.encode())}))
