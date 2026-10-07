"""All-component batch evaluation against the unchanged frozen reference.

Distance engine is unchanged pinned spiralcheck. Reference-point union takes
one minimum distance over every output face; duplicate outputs never increase
reference coverage. This complements, and does not alter, point-query scores.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import resource
import tempfile
import numpy as np
import tifffile

from evaluate_patch import (ORIGIN_XYZ,TOLERANCES,UM_PER_VOX,load_grid,
    all_quad,grid_triangles,face_stats,core_mask,TriangleSoup,surface_distance,
    summary,sha,write_json,write_npz)


def restrict_summary(dist, weights):
    if not len(weights) or weights.sum() <= 0:
        return {'status':'INSUFFICIENT_REFERENCE','samples':0,'coverage':None}
    if np.isinf(dist).all():
        return {'status':'NO_PRODUCT','samples':len(weights),
                'weighted_area_mm2':float(weights.sum()*UM_PER_VOX**2/1e6),
                'coverage':{str(t):0. for t in TOLERANCES},
                'distance_vox_p50_p90_p95':None}
    return summary(dist,weights)


def update_union(dist, points, soup):
    new=surface_distance(points,soup,chunk=256).dist
    np.minimum(dist,new,out=dist)


def read_candidate(record):
    path=Path(record['path'])
    # Refuse unbounded materialization before reading the TIFF arrays.
    for name in 'xyz':
        with tifffile.TiffFile(path/(name+'.tif')) as f:
            shape=f.series[0].shape
            if len(shape)!=2 or np.prod(shape)>250000:
                raise ValueError('Candidate exceeds250000gridvertex admission; replan resources')
    expected=record.get('files_sha256',{})
    for name,digest in expected.items():
        file=path/Path(name).name
        if sha(file)!=digest:
            raise ValueError(f'Candidate hash differs: {file}')
    frame=record['frame']
    xyz,valid=load_grid(path,frame)
    if frame=='local_l2_xyz':
        if 'origin_l2_zyx' not in record:
            raise ValueError('Every local batch candidate requires explicit origin_l2_zyx')
        delta=np.asarray(record['origin_l2_zyx'],float)[::-1]-ORIGIN_XYZ
        # Shift valid coordinates only, including coordinates now below zero.
        xyz[valid]+=delta.astype(xyz.dtype)
    return xyz,valid


def evaluate_batch(domain, manifest_path, output):
    root=Path(domain)
    meta=json.loads((root/'reference-domain.json').read_text())
    if sha(root/'reference-domain.npz')!=meta['domain_sha256']:
        raise ValueError('Frozen reference domain changed')
    d=np.load(root/'reference-domain.npz',allow_pickle=False)
    manifest=json.loads(Path(manifest_path).read_text())
    records=manifest['patches']
    target=int(d['target_component'])
    masks={'target_core':d['target_core'], 'target_unseeded':d['target_unseeded']}
    masks['other_known_core']=(d['component']>0)&(d['component']!=target)&core_mask(d['centers'])&~d['boundary']&(d['area']>1e-10)
    query_mask=masks['target_core']|masks['other_known_core']
    query_idx=np.flatnonzero(query_mask)
    points=d['centers'][query_mask]
    union_distance=np.full(len(points),np.inf)
    subset={name:mask[query_mask] for name,mask in masks.items()}
    soups={'all':TriangleSoup(d['vertices'],d['faces']),
           'target':TriangleSoup(d['vertices'],d['faces'][d['component']==target])}
    competitor=(d['component']>0)&(d['component']!=target)
    if competitor.any():
        soups['competitor']=TriangleSoup(d['vertices'],d['faces'][competitor])
    results=[]
    seen_geometry={}
    qualified_sample_seen=False
    for record in records:
        xyz,valid=read_candidate(record)
        qmask=all_quad(valid)
        item={'attempt_id':record['attempt_id'],'path':record['path'],
              'frame':record['frame'],'origin_l2_zyx':record.get('origin_l2_zyx'),
              'valid_vertices':int(valid.sum()),'full_quads':int(qmask.sum()),
              'selection':'ALL valid components; no GPquery-nearest selection',
              'files_sha256':{n:sha(Path(record['path'])/n) for n in ['x.tif','y.tif','z.tif']}}
        geom_key=json.dumps(item['files_sha256'],sort_keys=True)+str(record.get('origin_l2_zyx'))+record['frame']
        if geom_key in seen_geometry:
            item['exact_duplicate_of_attempt']=seen_geometry[geom_key]
        else:
            seen_geometry[geom_key]=record['attempt_id']
        if not qmask.any():
            item.update(status='NO_VALID_SURFACE',triangle_count=0)
            results.append(item)
            continue
        v,f,uv,_=grid_triangles(xyz,qmask)
        centers,area=face_stats(v,f)
        nonzero=area>1e-10
        if not nonzero.any():
            item.update(status='NO_VALID_SURFACE',triangle_count=0)
            results.append(item)
            continue
        from scipy import ndimage
        _,ncomp=ndimage.label(qmask)
        item.update(status='SCORED_CONTINUOUS_ONLY',quad_components=ncomp,
                    triangle_count=int(nonzero.sum()),
                    emitted_area_mm2=float(area[nonzero].sum()*UM_PER_VOX**2/1e6))
        candidate_soup=TriangleSoup(v,f[nonzero])
        # Union update uses every component of this output, even if none is
        # near our reference seed; minimum over files is duplicate-safe.
        update_union(union_distance,points,candidate_soup)
        core=nonzero&core_mask(centers)
        core_area=area[core]
        item['candidate_core_area_mm2']=float(core_area.sum()*UM_PER_VOX**2/1e6)
        item['outside_fixed_eval_core_area_mm2']=float(area[nonzero&~core].sum()*UM_PER_VOX**2/1e6)
        item['outside_fixed_eval_core_is_error']=False
        if core.any():
            rr={k:surface_distance(centers[core],s,chunk=256) for k,s in soups.items()}
            all_distance=rr['all'].dist
            boundary=d['boundary'][rr['all'].face_idx]
            ref_components=d['component'][rr['all'].face_idx]
            qualified=(all_distance<=max(TOLERANCES))&~boundary&(ref_components>0)
            qualified_sample_seen |= bool(qualified.any())
            target_nearer=np.ones(len(core_area),bool)
            ties=np.zeros(len(core_area),bool)
            if 'competitor' in rr:
                delta=rr['competitor'].dist-rr['target'].dist
                target_nearer=delta>1e-8
                ties=np.abs(delta)<=1e-8
                item['competitor_minus_target_margin_vox']=delta.tolist()
            target_area=core_area[qualified&target_nearer].sum()
            competitor_area=core_area[qualified&~target_nearer&~ties].sum()
            tie_area=core_area[qualified&ties].sum()
            item['core_correspondence_area_mm2']={
                'all_reference_supported':float(core_area[qualified].sum()*UM_PER_VOX**2/1e6),
                'target_nearer':float(target_area*UM_PER_VOX**2/1e6),
                'competitor_nearer':float(competitor_area*UM_PER_VOX**2/1e6),
                'numerical_tie':float(tie_area*UM_PER_VOX**2/1e6),
                'unknown':float(core_area[~qualified].sum()*UM_PER_VOX**2/1e6),
                'reference_boundary':float(core_area[boundary].sum()*UM_PER_VOX**2/1e6)}
            item['correspondence_is_identity_truth']=False
            item['candidate_to_reference']={k:summary(r.dist,core_area) for k,r in rr.items()}
        else:
            item['core_status']='OUTSIDE_FIXED_REFERENCE_CORE_UNKNOWN'
        results.append(item)
    known_intersection=qualified_sample_seen or bool((union_distance<=max(TOLERANCES)).any())
    union={name:restrict_summary(union_distance[m],d['area'][query_mask][m])
           for name,m in subset.items()}
    report={'schema':'iss-all-components-batch-union/1',
            'status':'SCORED_REFERENCE_INTERSECTION_ONLY' if known_intersection else 'INCONCLUSIVE_NO_KNOWN_INTERSECTION',
            'source_sha256':sha(__file__),'unchanged_distance_adapter_sha256':sha(Path(__file__).with_name('evaluate_patch.py')),
            'manifest_sha256':sha(manifest_path),'reference_domain_sha256':meta['domain_sha256'],
            'input_manifest':manifest,'patches':results,
            'union_reference_coverage':union,
            'union_rule':'Pointwise min exact distance over all nondegenerate faces of all valid components of every manifest output; each fixed reference triangle weight counted once.',
            'candidate_area_rule':'Per-output emitted/core/correspondence areas are not a physical surface union; do not sum duplicate or overlapping products as recovered area.',
            'reference_query_faces':len(query_idx),'fixed_target_faces':int(d['target_core'].sum()),
            'identity_verdict':'UNKNOWN','registration_error_independently_bounded':False,
            'unknown_is_error':False,'no_reference_intersection_is_success':False,
            'old_point_query_scores_unchanged':True,
            'sampling':'Frozen reference triangle centroids with area weights; candidate-direction trianglecentroids with physicalarea weights, not exact continuousarea integrals.'}
    samples=Path(output).with_suffix('.union-samples.npz')
    write_npz(samples,reference_face_idx=query_idx,union_distance_vox=union_distance,
              fixed_area_vox2=d['area'][query_mask])
    report['union_samples_sha256']=sha(samples)
    write_json(output,report)
    return report


def selftest():
    # Four equal reference cells; independent half-squares cover2 each.
    pts=np.array([[.25,.25,0],[.75,.25,0],[1.25,.25,0],[1.75,.25,0]])
    f=np.array([[0,1,2],[1,3,2]])
    a=TriangleSoup(np.array([[0.,0,0],[1.,0,0],[0.,1,0],[1.,1,0]]),f)
    b=TriangleSoup(a.vertices+[1.,0.,0.],f)
    dist=np.full(4,np.inf)
    update_union(dist,pts,a)
    first=dist.copy()
    assert int((dist<=.1).sum())==2
    update_union(dist,pts,a)
    assert np.array_equal(dist,first),'Duplicate must not increase union coverage'
    update_union(dist,pts,b)
    assert int((dist<=.1).sum())==4
    # One output has two disconnected components: both belong to union.
    allcomponents=TriangleSoup(np.concatenate([a.vertices,b.vertices]),np.concatenate([f,f+4]))
    both=np.full(4,np.inf)
    update_union(both,pts,allcomponents)
    assert np.array_equal(both,dist)
    assert restrict_summary(np.full(4,np.inf),np.ones(4))['status']=='NO_PRODUCT'
    assert restrict_summary(np.array([]),np.array([]))['status']=='INSUFFICIENT_REFERENCE'
    with tempfile.TemporaryDirectory(prefix='batch-eval-known-answer-') as tmp:
        root=Path(tmp)
        vertices=np.concatenate([a.vertices,b.vertices])*10+[200.,200.,200.]
        faces=np.concatenate([f,f+4])
        centers,area=face_stats(vertices,faces)
        write_npz(root/'reference-domain.npz',vertices=vertices,faces=faces,centers=centers,
                  area=area,component=np.ones(4,int),boundary=np.zeros(4,bool),
                  target_core=np.ones(4,bool),target_unseeded=np.ones(4,bool),
                  target_component=np.array(1))
        write_json(root/'reference-domain.json',{'domain_sha256':sha(root/'reference-domain.npz')})
        candidate=root/'candidate';candidate.mkdir()
        # Two valid square components separated in parameter space by sentinel
        # row; XYZ supplied in expanded1024 local coordinates.
        grid=np.full((5,2,3),-1.,np.float32)
        grid[0]=vertices[[0,1]]+256;grid[1]=vertices[[2,3]]+256
        grid[3]=vertices[[4,5]]+256;grid[4]=vertices[[6,7]]+256
        for i,name in enumerate('xyz'):
            tifffile.imwrite(candidate/(name+'.tif'),grid[...,i])
        record={'attempt_id':'knownA','path':str(candidate),'frame':'local_l2_xyz',
                'origin_l2_zyx':[16000,5248,3584]}
        manifest={'attempts':[{'attempt_id':'knownA','status':'ACCEPTED'}],
                  'patches':[record]}
        write_json(root/'manifest.json',manifest)
        report=evaluate_batch(root,root/'manifest.json',root/'report.json')
        assert report['patches'][0]['quad_components']==2
        assert report['union_reference_coverage']['target_core']['coverage']['0.5']==1.
        manifest['patches'].append(dict(record,attempt_id='knownDuplicate'))
        write_json(root/'manifest.json',manifest)
        duplicate=evaluate_batch(root,root/'manifest.json',root/'duplicate.json')
        assert duplicate['union_reference_coverage']==report['union_reference_coverage']
        assert duplicate['patches'][1]['exact_duplicate_of_attempt']=='knownA'
        empty={'attempts':[{'attempt_id':'empty','status':'DISCARDED'}],'patches':[]}
        write_json(root/'empty-manifest.json',empty)
        no_output=evaluate_batch(root,root/'empty-manifest.json',root/'empty.json')
        assert no_output['status']=='INCONCLUSIVE_NO_KNOWN_INTERSECTION'
    return {'status':'PASS_BATCH_UNION_INTERFACE_ONLY','checks':['Halfpatch covers2/4','Duplicate keeps2/4','Complementaryunion covers4/4','Two disconnected components both included','No product not success','No reference not success','End-to-end1024-origin TIFF sentinel+allcomponent conversion','Duplicate manifests leave fixed union exactly unchanged','Emptyattempt manifest inconclusive'],'scientific_efficacy_claim':False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['evaluate','selftest'])
    p.add_argument('--domain')
    p.add_argument('--manifest')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    start=time.monotonic()
    if a.action=='selftest':
        r=selftest();write_json(a.out,r)
    else:
        r=evaluate_batch(a.domain,a.manifest,a.out)
    print(json.dumps({'status':r['status'],'elapsed_s':time.monotonic()-start,
                      'rss_darwin_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


if __name__=='__main__':
    main()
