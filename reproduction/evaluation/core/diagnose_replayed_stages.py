"""Two frozen development cases; parity-gated geometry diagnostics only."""
from __future__ import annotations
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
from evaluate_patch import (TriangleSoup,surface_distance,grid_triangles,all_quad,
    face_stats,sha,write_json,write_npz,ORIGIN_XYZ,TOLERANCES,UM_PER_VOX)
from evaluate_patch_batch import restrict_summary

ROOT=Path('__HISTORICAL_DATA_ROOT__/p4_patch_eval_20261007')
REPLAY=Path('__HISTORICAL_WORK_ROOT__/iss-case-replay-20261007')


def reference(case):
    if case=='P29':
        meta=json.loads((ROOT/'reference-domain.json').read_text())
        assert sha(ROOT/'reference-domain.npz')==meta['domain_sha256']
        d=np.load(ROOT/'reference-domain.npz',allow_pickle=False)
        v=d['vertices']+ORIGIN_XYZ;f=d['faces']
        p=d['centers']+ORIGIN_XYZ
        box=(np.array([3904.,5568.,16320.]),np.array([4288.,5952.,16704.]))
        active=((p>=box[0])&(p<=box[1])).all(1)&(d['component']>0)&~d['boundary']&(d['area']>1e-10)
        idx=np.flatnonzero(active)
        return v,f,p[active],d['area'][active],d['component'],d['boundary'],idx,box,{'asset':'20231210121321','domain':'original_frozen_core384','geometry_sha256':meta['domain_sha256'],'no_domain_extension':True}
    ref=Path('__HISTORICAL_DATA_ROOT__/expanded_reference_20261007/20230702185753/qualified-geometry.npz')
    assert sha(ref)=='cff7978966c2513791e90cf1ec8514cc3803665b4da3e7bf788c129952c6a1d8'
    source_report=json.loads((ROOT/'C0-development32-20230702185753-evaluation.json').read_text())
    samples=ROOT/'C0-development32-20230702185753-evaluation-B_input_core896-C0-development-samples.npz'
    expected=source_report['domains']['B_input_core896']['conditions']['C0-development']['samples_sha256']
    assert sha(samples)==expected
    d=np.load(ref,allow_pickle=False);s=np.load(samples,allow_pickle=False)
    v,f=d['vertices_global_l2_xyz'],d['faces'];labels=np.zeros(len(f),np.int32);bounds=np.ones(len(f),bool)
    labels[s['reference_face_indices']]=s['component'];bounds[s['reference_face_indices']]=s['boundary']
    active=~s['boundary'];idx=s['reference_face_indices'][active]
    box=(np.array([3648.,5312.,16064.]),np.array([4544.,6208.,16960.]))
    return v,f,s['clipped_centers_global_l2_xyz'][active],s['clipped_area_vox2'][active],labels,bounds,idx,box,{'asset':'20230702185753','domain':'existing_exploratory_B896','geometry_sha256':sha(ref),'query_samples_sha256':expected,'boundary_faces_excluded':True}


def describe(values):
    if not len(values):return {'n':0}
    return {'n':len(values),'p10_p50_p90':np.quantile(values,[.1,.5,.9]).tolist(),
            'min':float(values.min()),'max':float(values.max())}


def summarize_points(points,soup,normals,labels,boundary,box):
    keep=((points>=box[0])&(points<=box[1])).all(1)
    points=points[keep]
    if not len(points):return {'points_in_domain':0,'status':'OUTSIDE_DOMAIN_UNKNOWN'}
    r=surface_distance(points,soup,chunk=128)
    signed=np.sum((points-r.closest)*normals[r.face_idx],axis=1)
    component=labels[r.face_idx]
    known=(component>0)&~boundary[r.face_idx]
    supported=known&(r.dist<=max(TOLERANCES))
    ids,count=np.unique(component,return_counts=True)
    top=np.argsort(r.dist,kind='stable')[-min(8,len(points)):]
    return {'points_in_domain':len(points),'distance_vox':describe(r.dist),
            'signed_nearest_GP_normal_residual_vox':describe(signed),
            'nearest_component_counts':{str(int(i)):int(n) for i,n in zip(ids,count)},
            'qualified_nearest_reference_count':int(known.sum()),
            'within8vox_qualified_count':int(supported.sum()),
            'unknown_count':int((~supported).sum()),
            'by_nearest_component':{str(int(c)):{'distances':describe(r.dist[component==c]),'signed':describe(signed[component==c])} for c in ids},
            'largest_distance_examples':[{'xyz':points[i].tolist(),'distance_vox':float(r.dist[i]),'signed_normal_vox':float(signed[i]),'nearest_reference_face':int(r.face_idx[i]),'nearest_component':int(component[i])} for i in top],
            'point_sampling_is_not_area_estimate':True,'nearest_component_is_not_identity_truth':True}


def grid_edges(xyz,q):
    rows,cols=np.nonzero(q);w=xyz.shape[1]
    tl=rows*w+cols;tr=tl+1;bl=tl+w;br=tl+w+1
    edges=np.concatenate([np.stack([tl,tr],1),np.stack([tl,bl],1),np.stack([tr,br],1),np.stack([bl,br],1)])
    edges.sort(1);edges=np.unique(edges,axis=0)
    points=xyz.reshape(-1,3)
    t=np.linspace(0,1,9)
    samples=points[edges[:,0],None,:]*(1-t[None,:,None])+points[edges[:,1],None,:]*t[None,:,None]
    return np.unique(samples.reshape(-1,3),axis=0),len(edges)


def run(case,out):
    started=time.monotonic()
    # Tiny independent interface checks run on the same compute node.
    test_grid=np.array([[[0.,0.,0.],[1.,0.,0.]],[[0.,1.,0.],[1.,1.,0.]]])
    test_edges,nedges=grid_edges(test_grid,np.ones((1,1),bool))
    assert nedges==4 and len(test_edges)==32
    tv,tf,_,_=grid_triangles(test_grid,np.ones((1,1),bool));ts=TriangleSoup(tv,tf)
    tr=surface_distance(np.array([[.5,.5,1.]]),ts,chunk=1)
    assert np.allclose(tr.dist,1.) and np.allclose(np.sum((np.array([[.5,.5,1.]])-tr.closest)*ts.face_normals()[tr.face_idx],axis=1),1.)
    manifest_path=REPLAY/case/'result.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['same_output_interpretation_admitted'] is True
    # Replay schema records error messages as a list, not an integer count.
    assert isinstance(manifest['observer_errors'],list) and not manifest['observer_errors']
    assert manifest['old_outputs_unchanged'] and manifest['inputs_unchanged']
    rv,rf,rpoints,rarea,rlabels,rboundary,ridx,box,refinfo=reference(case)
    rsoup=TriangleSoup(rv,rf);rnormals=rsoup.face_normals()
    rows=[];cache={};query_history=[]
    for record in manifest['stages']:
        if sha(record['path'])!=record['sha256']:raise ValueError('Stage hash changed')
        assert record['frame']=='local_l2_xyz' and record['origin_l2_zyx']==[16000,5248,3584]
        saved=np.load(record['path'],allow_pickle=False)
        xyz=saved['xyz'].astype(float);valid=saved['valid'].astype(bool)
        if xyz.ndim==2:
            pts=xyz[valid]+np.array([3584.,5248.,16000.])
            rows.append({'stage':record['stage'],'stage_sha256':record['sha256'],
                         'kind':'point_cloud_not_surface','point_diagnostic':summarize_points(pts,rsoup,rnormals,rlabels,rboundary,box)})
            continue
        assert valid.shape==xyz.shape[:2] and int(valid.sum())==record['valid_count']
        xyz[valid]+=np.array([3584.,5248.,16000.])
        h=hashlib.sha256();h.update(str(valid.shape).encode());h.update(valid.tobytes());h.update(xyz[valid].tobytes());key=h.hexdigest()
        row={'stage':record['stage'],'stage_sha256':record['sha256'],'semantic_geometry_sha256':key,
             'valid_vertices':int(valid.sum()),'explicit_saved_valid_mask_used':True}
        if key in cache:
            previous,stats,dist,signed=cache[key]
            row['same_geometry_as_stage']=previous;row['analysis']=stats
            query_history.append((record['stage'],dist,signed))
            rows.append(row);continue
        q=all_quad(valid)
        if not q.any():row['status']='NO_VALID_SURFACE';rows.append(row);continue
        cv,cf,uv,_=grid_triangles(xyz,q);cent,area=face_stats(cv,cf)
        nonzero=area>1e-10;cf=cf[nonzero];cent=cent[nonzero];area=area[nonzero]
        csoup=TriangleSoup(cv,cf)
        edgepoints,nedges=grid_edges(xyz,q)
        stats={'full_quads':int(q.sum()),'triangles':len(cf),
               'total_surface_area_mm2':float(area.sum()*UM_PER_VOX**2/1e6),
               'vertices':summarize_points(xyz[valid],rsoup,rnormals,rlabels,rboundary,box),
               'original_quad_edges':nedges,'edge9_unique':summarize_points(edgepoints,rsoup,rnormals,rlabels,rboundary,box),
               'face_centroids':summarize_points(cent,rsoup,rnormals,rlabels,rboundary,box)}
        reverse=surface_distance(rpoints,csoup,chunk=128)
        # Same fixed GP point and normal at every stage, so this is not
        # confounded by a different candidate sampling density or alignment fit.
        signed=np.sum((reverse.closest-rpoints)*rnormals[ridx],axis=1)
        stats['fixed_reference_to_stage']=restrict_summary(reverse.dist,rarea)
        stats['reference_components']={str(int(c)):restrict_summary(reverse.dist[rlabels[ridx]==c],rarea[rlabels[ridx]==c]) for c in np.unique(rlabels[ridx])}
        cache[key]=(record['stage'],stats,reverse.dist,signed)
        query_history.append((record['stage'],reverse.dist,signed));row['analysis']=stats;rows.append(row)
        print(json.dumps({'stage':record['stage'],'case':case,'valid':int(valid.sum()),'faces':len(cf)}),flush=True)
    common=np.ones(len(ridx),bool)
    for _,distance,_ in query_history:common&=distance<=max(TOLERANCES)
    common_summary={'definition':'Same fixed boundary-excluded reference points within8voxelofEVERY recorded valid surface stage; conditionaldiagnostic only',
                    'points':int(common.sum()),'area_mm2':float(rarea[common].sum()*UM_PER_VOX**2/1e6),
                    'per_stage':[]}
    for name,distance,signed in query_history:
        common_summary['per_stage'].append({'stage':name,'distance_vox':describe(distance[common]),'signed_fixed_GP_normal_vox':describe(signed[common])})
    samples=Path(out).with_suffix('.reference-stage-samples.npz')
    payload={'reference_face_idx':ridx,'reference_area_vox2':rarea,'component':rlabels[ridx],'common_within8':common}
    for i,(name,distance,signed) in enumerate(query_history):payload[f's{i}_distance']=distance;payload[f's{i}_signed']=signed
    write_npz(samples,**payload)
    report={'schema':'iss-fixed-case-stage-diagnostic/1','case':case,'status':'COMPLETE_DIAGNOSTIC_ONLY',
            'source_sha256':sha(__file__),'replay_manifest_sha256':sha(manifest_path),'replay_final_parity':manifest['coordinates_parity'],
            'reference':refinfo,'boundary_excluded_reference_area_mm2':float(rarea.sum()*UM_PER_VOX**2/1e6),
            'known_answer_checks':'Originalquad4edges9samplesdedup32;exactplaneoffset1andnativeGPnormalsignedresidual1',
            'stages':rows,'common_reference_support':common_summary,'sample_file_sha256':sha(samples),
            'sampling':'Every validvertex,9uniformpoints per originalquadedge deduplicatedXYZ,trianglecentroids; fixedreferencepointareaweightedcoverage. No translations or newthresholds.',
            'limits':['SamefinalXYZproven; no claimnewmethod','Point/edge distributions are countweighted and samplingdensity changesacrossstages','Fixedreference distances/normalresiduals permitconditionalstagecomparison,notregistration-error estimation','Unknown/off-reference andnearestothercomponent are notpaperidentityerrors','Existingreference/sourceframe uncertainty retained'],
            'elapsed_seconds':time.monotonic()-started,'peak_rss_darwin_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out,report);print(json.dumps({'status':report['status'],'case':case,'seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['P05','P29'],required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.case,a.out)
