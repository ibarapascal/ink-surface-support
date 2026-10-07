"""Fixed earlier-development P05/P29 native5/support5/2.5, original C0 reference cores."""
import argparse,gc,json,time,resource,os
from pathlib import Path
import numpy as np
import torch
import dtypes
from flatten_cohort_geometry import inspect_export
from evaluate_native_consumer import compact
from native32_det_probe import audit as audit_native32
from evaluate_references import evaluate
from diagnose_replayed_stages import reference
from evaluate_patch import sha
W=Path(__file__).parent
ARMS=('original5','support5','original2_5');SPACING={'original5':5.0,'support5':5.0,'original2_5':2.5};CASES=('C0-P05','C0-P29')
MANIFEST=Path('__HISTORICAL_WORK_ROOT__/iss-spacing5-development-consumer-20261007/qualified-outputs.json')
MANIFEST_SHA='36687533a9d4128792f0656a553a3e64caae3df4f0d5d1c40cc5559e44f452c9'
REF=Path('__HISTORICAL_DATA_ROOT__/p4_validation_references_20261007')

def run(case):
 start=time.monotonic();torch.set_num_threads(1);assert os.environ['LASAGNA_MAX_PRECISION_FLOAT']=='32' and dtypes.torch_float_hi==torch.float32 and dtypes.numpy_float_hi==np.float32
 assert sha(MANIFEST)==MANIFEST_SHA;m=json.loads(MANIFEST.read_text());assert m['all_input_output_source_config_hashes_verified']
 assert {(q['case'],q['arm']) for q in m['cases']}=={(c,a) for c in CASES for a in ARMS}
 assert case in CASES
 source_locked_case=case
 records={};meshes={}
 for q in m['cases']:
  if q['case']!=source_locked_case:continue
  case,arm=q['case'],q['arm'];key=case+'/'+arm;assert q['status'] in ('SUCCESS','CACHED_SUCCESS') and q['origin_l2_zyx']==[16000,5248,3584] and q['frame']=='local_l2_xyz'
  assert sha(q['record_path'])==q['record_sha256'];assert q['declared_stage_steps']==[['0','1000'],['1','1000'],['2','1000']]
  assert str(q['effective_float_hi_bits'])=='32';assert q['source_input_hashes_unchanged']
  if q['consumer_origin']=='REUSED_EFFECTIVE32_QUALIFIED':assert q['reuse_input_array_and_scale_parity'] and q['cached_from_current_spacing10_main']
  else:assert q['consumer_origin']=='NEW' and q['raw_consumer_origin']=='NEW_EXPLICIT_DEFAULT32_RUN' and q['effective_runtime_env']['LASAGNA_MAX_PRECISION_FLOAT']=='32'
  root=Path(q['checkpoint']).parent;assert Path(q['output_path'])==root/'tifxyz/flatten.tifxyz'
  for f in q['files']:
   p=root/f['relative'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
  source=Path(q['source_input_path'])
  for n,h in q['source_input_files_sha256'].items():assert sha(source/n)==h
  sm=json.loads((source/'meta.json').read_text());em=json.loads((Path(q['output_path'])/'meta.json').read_text());spacing=SPACING[arm]
  assert np.allclose(sm['scale'],[1/spacing]*2,rtol=0,atol=1e-12) and np.allclose(em['scale'],[1/spacing]*2,rtol=0,atol=1e-12)
  report,stages=inspect_export(q['checkpoint'],q['origin_l2_zyx']);audit_native32(q['checkpoint'],report,W/(case+'-'+arm+'-det-mask-difference.npz'))
  meshes[key]=compact(stages['export']);del stages
  records[key]={'case':case,'arm':arm,'source_spacing_input_L2_voxels':spacing,'consumer_output_step_input_L2_voxels':spacing,'working_grid_spacing_input_L2_voxels':5,'CT_physical_voxel_um':9.6,'note':'2.5isinterpolatedoutputspacingonworking5; not2.5umCT oraddedsourceinformation.','diagnostics':report,'provenance':q}
  gc.collect()
 rv,rf,points,weights,labels,boundary,idx,box,info=reference(case.removeprefix('C0-'))
 result={'schema':'iss-spacing5-development-existing-consumers/1','status':'COMPLETE','role':'SEEN_PREVIOUSLY_FIXED_DEVELOPMENT_GROUP_NOT_HOLDOUT','case':case,'labels':SPACING,'working_grid_spacing':5,'records':records,'reference':info,'domains':{info['domain']:evaluate(meshes,rv,rf,points,weights,labels,boundary,idx,box)},'qualified_consumer_manifest_sha256':MANIFEST_SHA,'source_sha256':sha(__file__),'numeric_core_sha256':{'inspect':sha(inspect_export.__code__.co_filename),'evaluate':sha(evaluate.__code__.co_filename),'reference':sha(reference.__code__.co_filename)},'no_optimizer_calls':True}
 result.update(seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=['PreviouslyfixeddevelopmentP05/P29 group;no newseed or candidateoutcome selection. SeparatefromP10/P06/P12resourcegroup, notholdout.','P05original5 reusesexplicit32P05-output-step existing5_to5; P29baseline rerunnew32 ratherthanreuseoldprecisionunknown. New4candidates; currentmanifestpreserves actualprovenance.','Individualreferencecoverage, no acrosscases/assets sum;unknown noterror, components notwindingtruth.','Source andconsumer spacings explicitly5/5/2.5; sameworking5information, nothigherresolutioninput.','ThisGPscorerusespercase nativecore; exactsame-sourcepairedSDir runsunderseparatepre-scoreaddendum.'])
 text=json.dumps(result,indent=2,allow_nan=False)+'\n';assert len(text.encode())<4*1024**2;(W/(case+'-score.json')).write_text(text);assert sum(p.stat().st_size for p in W.rglob('*') if p.is_file())<16*1024**2
 print(json.dumps({'status':'COMPLETE','case':case,'seconds':result['seconds'],'bytes':len(text.encode())}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,required=True);run(p.parse_args().case)
