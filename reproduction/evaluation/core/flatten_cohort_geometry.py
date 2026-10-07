"""Frozen cohort consumer geometry preparation; exact six-case diagnostic core.

Only checkpoint/output reads. No optimizer, growth or reference mesh calls.
"""
from __future__ import annotations
import argparse,ast,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import model as native_model
import opt_loss_flatten as native_loss
from evaluate_flatten_consumer import mesh,stats,load_xyz
from evaluate_patch import all_quad,surface_distance,sha,write_json
from score_frozen_validation import measure,mm2


def inspect_export(checkpoint, origin):
    checkpoint=Path(checkpoint);root=checkpoint.parent
    assert checkpoint.stat().st_size<100000000
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    sx=state['flatten_source_xyz'];sv=state['flatten_source_valid'];sc=state['flatten_source_cell_valid']
    assert sx.ndim==3 and sx.shape[-1]==3 and sx.numel()<750000
    source=sx.numpy();valid=sv.numpy().astype(bool);filtered=sc.numpy().astype(bool)
    sourceq=all_quad(valid);assert not np.any(filtered&~sourceq)
    pyramids=[state[k] for k in sorted((k for k in state if k.startswith('flatten_map_ms.')),key=lambda k:int(k.rsplit('.',1)[1]))]
    assert pyramids
    uv=native_model.Model3D._integrate_pyramid_3d(pyramids,pyramid_d=False)[:,0].permute(1,2,0).contiguous()
    assert tuple(uv.shape[:2])==tuple(sx.shape[:2])
    metric=state.get('flatten_source_metric')
    if metric is None or not metric.numel():metric=native_model.Model3D._flatten_source_metric(sx)
    step=state['flatten_target_step'];loss,lm,lmask,lvalid=native_loss._flatten_forward_sdir_core(uv,metric,sc,step,1e-8)
    u=uv.numpy().astype(np.float64)
    a,b,c,d=u[:-1,:-1],u[1:,:-1],u[:-1,1:],u[1:,1:]
    s,t=b-a,c-a;u1,v1=d-b,c-b
    det0=s[...,0]*t[...,1]-s[...,1]*t[...,0]
    det1=u1[...,0]*v1[...,1]-u1[...,1]*v1[...,0]
    finite=all_quad(np.isfinite(u).all(2)&np.isfinite(source).all(2))
    eligible=filtered&finite&np.isfinite(det0)&np.isfinite(det1)&(det0>1e-10)&(det1>1e-10)
    exportpath=root/'tifxyz/flatten.tifxyz';out,ov=load_xyz(exportpath)
    cm=state['mesh_flat'].numpy()[:,0].transpose(1,2,0).copy();pm=state['flatten_point_mask'].numpy().astype(bool);cm[~pm]=-1
    assert out.shape==cm.shape and out.tobytes()==cm.tobytes(),'native checkpoint/export coordinate mismatch'
    stages={'source_complete':mesh(source,sourceq,origin),'source_angle_filtered':mesh(source,filtered,origin),
            'source_export_eligible':mesh(source,eligible,origin),'export':mesh(out,all_quad(ov),origin)}
    # Native metric loss is per quad; area weights come from the same 3D corners.
    v00,v10,v01,v11=source[:-1,:-1].astype(float),source[1:,:-1].astype(float),source[:-1,1:].astype(float),source[1:,1:].astype(float)
    qa=(np.linalg.norm(np.cross(v10-v00,v01-v00),axis=2)+np.linalg.norm(np.cross(v11-v10,v01-v10),axis=2))/2
    meta=json.loads((exportpath/'meta.json').read_text())
    report={'checkpoint_sha256':sha(checkpoint),'export_path':str(exportpath),'export_hashes':{p.name:sha(p) for p in exportpath.iterdir() if p.is_file() and p.name in ['x.tif','y.tif','z.tif','meta.json']},
      'source_shape':list(source.shape),'export_shape':list(out.shape),'source_valid_vertices':int(valid.sum()),'export_valid_vertices':int(ov.sum()),
      'source_complete_quads':int(sourceq.sum()),'source_angle_filter_retained_quads':int(filtered.sum()),'source_export_eligible_quads':int(eligible.sum()),
      'checkpoint_export_XYZ_bytes_equal':True,'native_export_metadata_area_vx2_not_physical_triangle_area':meta.get('area_vx2'),
      'physical_area_mm2':{k:mm2(m['area']) for k,m in stages.items()},
      'native_SDir':{'unweighted_native_loss':float(loss),'area_weighted_diagnostic':stats(lm.numpy()[lvalid.numpy()],qa[lvalid.numpy()]),'excluded_source_cell_area_mm2':mm2(qa[filtered&~lvalid.numpy()]),'native_eps':1e-8,'not_new_rejection_gate':True},
      'native_inversion_eligibility':{'threshold':1e-10,'filtered_cell_area_rejected_mm2':mm2(qa[filtered&~eligible]),'both_triangle_dets_positive_required':True},
      'export_bytes':sum(p.stat().st_size for p in exportpath.iterdir() if p.is_file()),'all_stage_geometry':{k:{'quads':m['quads'],'triangles':len(m['f']),'zeroarea_triangles':m['zeroarea_triangles']} for k,m in stages.items()}}
    e=stages['export'];q=stages['source_export_eligible'];dist=surface_distance(q['p'],e['soup'],chunk=128).dist if e['soup'] is not None else np.full(len(q['p']),np.inf)
    report['eligible_source_to_export']=measure(dist,q['area'])
    report['area_difference_not_exact_identity_loss']=True
    return report,stages



INPUT_SHA='cf51d9eeeee27e59f8753a8fe174d51702e3519c7b5836c7bca85dd84f659b8b'
ROOT=Path('__HISTORICAL_WORK_ROOT__/iss-consumer-init-20261007/cohort47')
OUTPUT=Path('__HISTORICAL_DATA_ROOT__/flatten_cohort_evaluation_20261007')
ARMS=('old20','candidate20','existing10')


def verify_qualified(item):
    assert item['status'] in ('SUCCESS','CACHED_SUCCESS')
    root=Path(item['checkpoint']).parent
    assert Path(item['path'])==root
    assert Path(item['output_path'])==root/'tifxyz/flatten.tifxyz'
    assert len(item['stage_completions'])==3
    for file in item['files']:
        path=root/file['relative']
        assert path.stat().st_size==file['bytes'] and sha(path)==file['sha256']
    return root


def prepare(region,manifest_path):
    started=time.monotonic();torch.set_num_threads(1)
    ip=ROOT/'immutable-inputs.json';assert sha(ip)==INPUT_SHA;inputs=json.loads(ip.read_text())
    native=json.loads(Path(manifest_path).read_text())
    selected=[r for r in inputs['rows'] if r['region']==region]
    want={r['id']:r for r in selected};assert len(want)==(23 if region=='V1' else 24)*3
    supplied={r['case']+'-'+r['arm']:r for r in native['cases'] if r['case'].startswith(region+'-')}
    assert set(supplied)==set(want),'No replacements or missing accepted-source identities'
    out=OUTPUT/region;out.mkdir(parents=True,exist_ok=True)
    records=[];mesh_bytes=0
    for identity,source in want.items():
        item=supplied[identity]
        assert item['arm']==source['arm']
        for name,h in source['files_sha256'].items():assert sha(Path(source['path'])/name)==h
        row={'id':identity,'attempt_id':source['attempt_id'],'arm':source['arm'],'region':region,'origin_l2_zyx':source['origin_l2_zyx'],'consumer_status':item['status'],'source_record':source}
        if item['status'] not in ('SUCCESS','CACHED_SUCCESS'):
            row.update(status='NO_EXPORT_CONSUMER_FAILED',consumer_record=item);records.append(row);continue
        verify_qualified(item)
        report,meshes=inspect_export(item['checkpoint'],source['origin_l2_zyx'])
        export=meshes['export'];mesh_path=out/(identity+'.npz')
        # Coordinates already promoted and translated once; do not rescale.
        assert mesh_bytes+export['v'].nbytes+export['f'].nbytes<8*1024**2
        np.savez_compressed(mesh_path,vertices_global_l2_xyz=export['v'],faces=export['f'])
        mesh_bytes+=mesh_path.stat().st_size
        row.update(status='EXPORT_SURFACE' if len(export['f']) else 'EMPTY_EXPORT',diagnostics=report,
                   mesh={'path':str(mesh_path),'sha256':sha(mesh_path),'bytes':mesh_path.stat().st_size,'faces':len(export['f'])})
        records.append(row)
    original_absent=[r for r in inputs['original_no_consumer_input_attempts'] if r['region']==region]
    assert len(original_absent)==(32-len(want)//3)*3
    result={'schema':'iss-flatten-cohort-geometry/1','status':'COMPLETE','region':region,'original_attempts_per_arm':32,
      'input_manifest_sha256':INPUT_SHA,'consumer_manifest_path':str(manifest_path),'consumer_manifest_sha256':sha(manifest_path),
      'original_no_consumer_input_attempts':original_absent,'records':records,'mesh_file_bytes':mesh_bytes,
      'source_sha256':sha(__file__),'diagnostic_base_sha256':sha(Path(__file__).with_name('evaluate_flatten_consumer.py')),
      'new_optimization':False,'elapsed_seconds':time.monotonic()-started}
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n';assert len(encoded.encode())<1024**2
    (out/'geometry-index.json').write_text(encoded)
    print(json.dumps({'status':'COMPLETE','region':region,'records':len(records),'mesh_bytes':mesh_bytes}),flush=True)


def selftest():
    """Existing six actual caches plus tiny union identity, never optimize."""
    torch.set_num_threads(1)
    import evaluate_flatten_consumer as original
    from evaluate_patch import TriangleSoup
    from score_frozen_validation import merge
    source=Path(original.__file__).read_text();new=Path(__file__).read_text()
    old_body=source.split('    state=torch.load(checkpoint,map_location=')[1].split('\n\ndef score_ref')[0]
    new_body=new.split('    state=torch.load(checkpoint,map_location=')[1].split('\n\n\nINPUT_SHA')[0]
    assert old_body.rstrip()==new_body.rstrip()
    six=Path('__HISTORICAL_WORK_ROOT__/iss-consumer-init-20261007/flatten-output-manifest.json')
    assert sha(six)=='56bf04b12dd8148e80165debaaad78d3656118ede525638ffac82cc9825e9246'
    records=[]
    for case,origin in [('P05',[16000,5248,3584]),('V1-P06',[16128,3584,3840])]:
        previous=Path('__HISTORICAL_DATA_ROOT__/p4_patch_eval_20261007')/('flatten-P05-evaluation.json' if case=='P05' else 'flatten-P06-20231210121321-evaluation.json')
        expected=json.loads(previous.read_text())
        for arm in ARMS:
            report,_=inspect_export(original.BASE/'flatten'/(case+'-'+arm)/'model_final.pt',origin)
            assert report==expected['arms'][arm]
            records.append(case+'-'+arm)
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    a=TriangleSoup(v,np.array([[0,1,2]]));b=TriangleSoup(v+np.array([3.,0.,0.]),np.array([[0,1,2]]))
    points=np.array([[.2,.2,1.],[3.2,.2,2.],[1.,0.,0.]])
    dist=surface_distance(points,merge([a,b]),chunk=2).dist
    expected=np.minimum(surface_distance(points,a,chunk=2).dist,surface_distance(points,b,chunk=2).dist)
    assert np.array_equal(dist,expected)
    duplicate=surface_distance(points,merge([a,b,a]),chunk=2).dist
    assert np.array_equal(duplicate,dist)
    result={'status':'PASSED','six_cached_diagnostics_exact_equal':records,'numerical_core_source_bytes_identical':True,
       'merged_equals_per_surface_min':True,'duplicate_output_union_unchanged':True,'empty_union_is_none':merge([]) is None,
       'source_sha256':sha(__file__),'new_optimization':False}
    write_json(OUTPUT/'selftest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--region',choices=['V1','V2']);p.add_argument('--manifest');a=p.parse_args()
    if a.selftest:OUTPUT.mkdir(parents=True,exist_ok=True);selftest()
    else:assert a.region and a.manifest;prepare(a.region,a.manifest)
