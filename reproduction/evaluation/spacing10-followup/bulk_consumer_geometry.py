"""Spacing10 cohort bookkeeping around the unchanged validated inspect_export."""
import argparse,hashlib,json,time,os
from pathlib import Path
import numpy as np
import torch
from flatten_cohort_geometry import inspect_export
from evaluate_patch import sha
from native32_det_probe import audit as audit_native32
import native32_det_probe
import evaluate_flatten_consumer
import dtypes

ARMS=('existing10_native','support10_native','existing5_native')


def write(path,record):
    text=json.dumps(record,indent=2,allow_nan=False)+'\n'
    assert len(text.encode())<2*1024**2
    temp=path.with_suffix('.partial');temp.write_text(text);temp.replace(path)


def key(item):return item['case']+'-'+item['arm']


def prepare(args):
    torch.set_num_threads(1)
    assert os.environ.get('LASAGNA_MAX_PRECISION_FLOAT')=='32'
    assert dtypes.torch_float_hi==torch.float32 and dtypes.numpy_float_hi==np.float32
    numeric_sources={
      'wrapper':sha(__file__), 'probe':sha(native32_det_probe.__file__),
      'inspect_core':sha(inspect_export.__code__.co_filename),
      'inspect_helpers':sha(evaluate_flatten_consumer.__file__),
      'native_model':native32_det_probe.MODEL_SHA,'native_dtypes':sha(dtypes.__file__)}
    effective_eval={'LASAGNA_MAX_PRECISION_FLOAT':os.environ['LASAGNA_MAX_PRECISION_FLOAT'],
      'torch_float_hi':str(dtypes.torch_float_hi),'numpy_float_hi':np.dtype(dtypes.numpy_float_hi).name,
      'torch_intraop_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
      'torch_version':torch.__version__,'numpy_version':np.__version__,
      'LASAGNA_COMPILE_FLATTEN':os.environ.get('LASAGNA_COMPILE_FLATTEN'),
      'LASAGNA_FUSED_FLATTEN_ADAM_CLAMP':os.environ.get('LASAGNA_FUSED_FLATTEN_ADAM_CLAMP'),
      'dispatch':'eager original diagnostic cores; optimizer not invoked'}
    qualification_version='strict-new-default32-v1_native-det32-vs64-v1'
    assert sha(args.inputs)==args.inputs_sha256
    inputs=json.loads(args.inputs.read_text());manifest=json.loads(args.manifest.read_text())
    assert manifest['all_input_output_source_config_hashes_verified'] is True
    assert manifest['manifest_sha256']==args.inputs_sha256
    sources={r['id']:r for r in inputs['rows'] if r['region']==args.region}
    absent=[r for r in inputs['original_no_consumer_input_attempts'] if r['region']==args.region]
    for arm in ARMS:
        a=[r for r in sources.values() if r['arm']==arm];b=[r for r in absent if r['arm']==arm]
        assert len(a)+len(b)==32
        assert {r['attempt_id'] for r in a+b}=={args.region+f'-P{i:02d}' for i in range(1,33)}
    supplied={key(r):r for r in manifest['cases'] if r['case'].startswith(args.region+'-')}
    assert len(supplied)==len([r for r in manifest['cases'] if r['case'].startswith(args.region+'-')])
    assert set(supplied)<=set(sources)
    out=args.output_root/args.region;out.mkdir(parents=True,exist_ok=True)
    records=[];pending=[];mesh_bytes=0
    for identity,source in sources.items():
        q=supplied.get(identity)
        if q is None or q['status'] in ('PENDING','STARTED','RUNNING','QUEUED'):
            pending.append(identity);continue
        for n,h in source['files_sha256'].items():assert sha(Path(source['path'])/n)==h
        record={'id':identity,'attempt_id':source['attempt_id'],'arm':source['arm'],'region':args.region,
                'origin_l2_zyx':source['origin_l2_zyx'],'consumer_status':q['status'],'source_record':source,
                'consumer_record':q,'new_or_cache':'CURRENT_APPROVED_PILOT' if source['mode']=='QUALIFIED_PILOT_REUSE' else 'NEW_EXPLICIT_DEFAULT32',
                'timing_scope':'Consumer record preserves actual original or new job timing; retrieval isnotcompute.'}
        if q['status'] not in ('SUCCESS','CACHED_SUCCESS'):
            assert q['status']=='FAILED_OR_INCOMPLETE', 'Unknown consumer status needs qualification, not silent zero credit'
            record['status']='NO_EXPORT_CONSUMER_FAILED';records.append(record);continue
        if source['mode']=='QUALIFIED_PILOT_REUSE':
            assert identity==inputs['pilot_id'] and str(source['cache']['effective_float_hi_bits'])=='32'
            assert source['cache']['effective_runtime_env']['LASAGNA_MAX_PRECISION_FLOAT'] in (None,'32')
            assert q['checkpoint']==source['cache']['checkpoint'] and q['files']==source['cache']['files']
            record['precision_evidence']='Current approved largest5 pilot: recorded unset+locked default32; no invented module observation.'
        else:
            assert source['mode']=='NEW'
            assert q['consumer_origin'] in ('NEW','NEW_EXPLICIT_DEFAULT32_RUN')
            assert sha(q['record_path'])==q['record_sha256']
            raw=json.loads(Path(q['record_path']).read_text())
            assert raw['consumer_origin']=='NEW_EXPLICIT_DEFAULT32_RUN' and raw['status']=='SUCCESS'
            assert raw['checkpoint']==q['checkpoint'] and raw['files']==q['files']
            assert raw['effective_runtime_env']==q['effective_runtime_env'] and raw['effective_module_dtype']==q['effective_module_dtype']
            assert q['effective_runtime_env']==inputs['effective_environment']
            mod=q['effective_module_dtype']
            assert mod['torch_float_hi']=='torch.float32' and mod['numpy_float_hi']=='float32' and mod['torch_intraop_threads']==1
            record['precision_evidence']={'recorded_environment':q['effective_runtime_env'],'recorded_modules':mod}
        root=Path(q['checkpoint']).parent
        assert Path(q['output_path'])==root/'tifxyz/flatten.tifxyz'
        assert len(q['stage_completions'])==3
        if 'declared_stage_steps' in q:assert q['declared_stage_steps']==[['0','1000'],['1','1000'],['2','1000']]
        for f in q['files']:
            path=root/f['relative'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
        scientific_key=hashlib.sha256(json.dumps({'input':source,'numeric_sources':numeric_sources,'effective_evaluation':effective_eval,'qualification_version':qualification_version,'checkpoint_sha256':sha(q['checkpoint']),
             'export_hashes':{n:sha(Path(q['output_path'])/n) for n in ['x.tif','y.tif','z.tif','meta.json']}},sort_keys=True).encode()).hexdigest()
        case_record=out/(identity+'.json')
        if case_record.exists():
            cached=json.loads(case_record.read_text())
            assert cached['scientific_key']==scientific_key
            assert sha(cached['mesh']['path'])==cached['mesh']['sha256']
            cached['consumer_record']=q;cached['consumer_status']=q['status'];cached['new_or_cache']=record['new_or_cache']
            records.append(cached);mesh_bytes+=cached['mesh']['bytes'];continue
        diagnostic,meshes=inspect_export(q['checkpoint'],source['origin_l2_zyx']);export=meshes['export']
        audit_native32(q['checkpoint'],diagnostic,out/(identity+'-native32-mask-difference.npz'))
        mesh_path=out/(identity+'.npz')
        assert export['v'].nbytes+export['f'].nbytes+mesh_bytes<16*1024**2
        np.savez_compressed(mesh_path,vertices_global_l2_xyz=export['v'],faces=export['f'])
        mesh_bytes+=mesh_path.stat().st_size
        record.update(scientific_key=scientific_key,status='EXPORT_SURFACE' if len(export['f']) else 'EMPTY_EXPORT',
          diagnostics=diagnostic,mesh={'path':str(mesh_path),'sha256':sha(mesh_path),'bytes':mesh_path.stat().st_size,'faces':len(export['f'])})
        write(case_record,record);records.append(record)
    state='PARTIAL_NOT_SCORABLE' if pending else 'COMPLETE'
    index={'schema':'iss-spacing10-cohort-consumer-geometry/1','status':state,'region':args.region,
           'original_attempts_per_arm':32,'input_manifest_sha256':args.inputs_sha256,'consumer_manifest_sha256':sha(args.manifest),
           'original_no_consumer_input_attempts':absent,'records':records,'pending':pending,'mesh_file_bytes':mesh_bytes,
           'source_sha256':sha(__file__),'numeric_diagnostic':'Unchanged inspect_export plus native32-vs64 eligibility audit; differences marked pending, actual export geometry retained.',
           'numeric_source_hashes':numeric_sources,'effective_evaluation':effective_eval,'qualification_version':qualification_version,
           'density_policy':'native10/10/5; no optimization by evaluator'}
    write(out/'geometry-index.json',index)
    print(json.dumps({'status':state,'region':args.region,'completed':len(records),'pending':len(pending),'mesh_bytes':mesh_bytes}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--region',choices=['V1','V2'],required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    prepare(p.parse_args())
