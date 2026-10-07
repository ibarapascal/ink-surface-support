"""Read-only evaluation of the six fixed actual native flatten exports."""
from __future__ import annotations
import argparse,gc,json,time,resource,hashlib
from pathlib import Path
import numpy as np
import tifffile
import torch
import model as native_model
import opt_loss_flatten as native_loss
from evaluate_patch import all_quad,grid_triangles,face_stats,TriangleSoup,surface_distance,sha,write_json,TOLERANCES
from evaluate_patch_batch import restrict_summary
from evaluate_expanded_references import domain_geometry
from diagnose_replayed_stages import reference
from score_frozen_validation import mm2,measure

BASE=Path('__HISTORICAL_WORK_ROOT__/iss-consumer-init-20261007')
REF=Path('__HISTORICAL_DATA_ROOT__/p4_validation_references_20261007')
ARMS=('old20','candidate20','existing10')


def mesh(grid,qmask,origin):
    xyz=grid.astype(float).copy();xyz+=np.asarray(origin[::-1],float)
    v,f,_,_=grid_triangles(xyz,qmask);p,a=face_stats(v,f);use=a>1e-10
    return {'v':v,'f':f[use],'p':p[use],'area':a[use],
            'soup':TriangleSoup(v,f[use]) if use.any() else None,
            'quads':int(qmask.sum()),'zeroarea_triangles':int((~use).sum())}


def load_xyz(path):
    for c in 'xyz':
        with tifffile.TiffFile(path/(c+'.tif')) as t:
            shape=t.series[0].shape;assert len(shape)==2 and np.prod(shape)<250000
    a=np.stack([tifffile.imread(path/(c+'.tif'),maxworkers=1) for c in 'xyz'],2)
    valid=np.isfinite(a).all(2)&~(a==-1).all(2)
    return a,valid


def stats(values,weights):
    take=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    v,w=values[take],weights[take]
    if not len(v):return {'status':'NO_VALID_WEIGHTED_SAMPLES'}
    order=np.argsort(v,kind='stable');cdf=np.cumsum(w[order])/w.sum()
    return {'n':len(v),'source_area_mm2':mm2(w),'area_weighted_mean':float(np.sum(v*w)/w.sum()),
            'area_weighted_p50_p90_p95':[float(v[order[min(np.searchsorted(cdf,p),len(v)-1)]]) for p in [.5,.9,.95]],'min':float(v.min()),'max':float(v.max())}


def inputs(case,arm,origin):
    root=BASE/'flatten'/(case+'-'+arm);checkpoint=root/'model_final.pt'
    mp=BASE/'flatten-output-manifest.json'
    assert sha(mp)=='56bf04b12dd8148e80165debaaad78d3656118ede525638ffac82cc9825e9246'
    qualified=json.loads(mp.read_text());item=next(x for x in qualified['cases'] if x['case']==case and x['arm']==arm)
    assert item['source_input_hashes_unchanged'] and len(item['stage_completions'])==3
    for entry in item['files']:assert sha(root/entry['relative'])==entry['sha256']
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


def score_ref(rv,rf,rpoints,weights,labels,boundary,ridx,box,stages):
    soup=TriangleSoup(rv,rf);report={};cache={}
    for arm,meshes in stages.items():
        ar={}
        for stage,m in meshes.items():
            digest=hashlib.sha256(m['v'].tobytes()+m['f'].tobytes()).hexdigest()
            if digest in cache:
                ar[stage]=cache[digest];continue
            dist=surface_distance(rpoints,m['soup'],chunk=128).dist if m['soup'] is not None else np.full(len(rpoints),np.inf)
            reverse=measure(dist,weights)
            row={'fixed_reference_to_surface':reverse,'reference_components':{str(int(c)):measure(dist[labels[ridx]==c],weights[labels[ridx]==c]) for c in np.unique(labels[ridx])},'status':'INCONCLUSIVE_NO_REFERENCE_INTERSECTION' if not np.any(dist<=8) else 'GEOMETRIC_SUPPORT_ONLY'}
            if len(m['f']):
                cp,area,_,_,cb=domain_geometry(m['v'],m['f'],box[0],box[1]);take=area>1e-10
                if take.any():
                    near=surface_distance(cp[take],soup,chunk=128);known=(labels[near.face_idx]>0)&~boundary[near.face_idx];supported=known&(near.dist<=8)
                    row.update(surface_to_reference=restrict_summary(near.dist,area[take]),supported8_mm2=mm2(area[take][supported]),unknown_mm2=mm2(area[take][~supported]))
                else:row.update(supported8_mm2=0.,unknown_mm2=0.)
                row.update(clipped_surface_mm2=mm2(area),boundary_surface_mm2=mm2(area[cb]))
            else:row.update(clipped_surface_mm2=0.,boundary_surface_mm2=0.,supported8_mm2=0.,unknown_mm2=0.)
            ar[stage]=row;cache[digest]=row
        report[arm]=ar
    return report


def run(case,asset,out):
    t=time.monotonic();torch.set_num_threads(1)
    origin=[16000,5248,3584] if case=='P05' else [16128,3584,3840]
    plan=json.loads((Path(__file__).parent/'flatten-consumer-evaluation-plan.json').read_text())
    for name,entry in plan['source_pointers'].items():assert sha(BASE/'official-cli'/name)==entry['sha256']
    result={'schema':'iss-actual-native-flatten-evaluation/1','case':case,'asset':asset,'status':'COMPLETE','source_sha256':sha(__file__),'arms':{},'domains':{},'limits':plan['limits'],'new_optimization':False}
    stages={}
    for arm in ARMS:
        result['arms'][arm],stages[arm]=inputs(case,arm,origin)
        print(json.dumps({'case':case,'arm':arm,'area':result['arms'][arm]['physical_area_mm2'],'source_to_export':result['arms'][arm]['eligible_source_to_export']['coverage']}),flush=True)
    if case=='P05':
        assert asset=='20230702185753'
        rv,rf,p,a,l,b,idx,box,info=reference(case)
        result['reference']=info;result['domains']['B896']=score_ref(rv,rf,p,a,l,b,idx,box,stages)
    else:
        manifest=json.loads((REF/'reference-manifest.json').read_text());entry=next(e for e in manifest['assets'] if e['region']=='V1' and e['asset']==asset)
        rp=Path(entry['geometry']['path']);assert sha(rp)==entry['geometry']['sha256'];d=np.load(rp,allow_pickle=False);rv=d['vertices_global_l2_xyz'];rf=d['faces'];result['reference']=entry['geometry']
        for domain in ['B1408','A384']:
            a=d[domain+'_clipped_area_vox2'];l=d[domain+'_component'];b=d[domain+'_boundary'];idx=np.flatnonzero((a>1e-10)&~b);points=d[domain+'_centroids_global_l2_xyz'][idx];box=np.asarray(entry['domains'][domain]['box_global_l2_xyz'])
            result['domains'][domain]=score_ref(rv,rf,points,a[idx],l,b,idx,box,stages)
    result['elapsed_seconds']=time.monotonic()-t;result['peak_rss_darwin_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out,result);print(json.dumps({'status':'COMPLETE','case':case,'asset':asset,'seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['P05','V1-P06'],required=True);p.add_argument('--asset',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.case,a.asset,a.out)
