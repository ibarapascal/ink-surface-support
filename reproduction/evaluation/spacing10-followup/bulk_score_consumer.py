"""One frozen ROI/reference job: exact union of all native flatten exports.

No per-output full-reference traversal. No optimizer, new reference or domain.
"""
from __future__ import annotations
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from evaluate_patch import TriangleSoup,surface_distance,face_stats,sha,TOLERANCES
from evaluate_patch_batch import restrict_summary
from evaluate_expanded_references import domain_geometry
from score_frozen_validation import merge,measure,mm2

ROOT=Path('__HISTORICAL_DATA_ROOT__/spacing10_followup_20261007/full64-consumer-evaluation')
REF=Path('__HISTORICAL_DATA_ROOT__/p4_validation_references_20261007')
ARMS=('existing10_native','support10_native','existing5_native')


def run(region,asset):
    start=time.monotonic();ip=ROOT/region/'geometry-index.json';index=json.loads(ip.read_text())
    assert index['status']=='COMPLETE' and index['region']==region and index['original_attempts_per_arm']==32
    assert not index.get('pending')
    assert len(index['records'])+len(index['original_no_consumer_input_attempts'])==32*3
    records={};meshes={};unions={}
    for arm in ARMS:
        selected=[r for r in index['records'] if r['arm']==arm]
        assert len({r['attempt_id'] for r in selected})==len(selected)
        absent=[r for r in index['original_no_consumer_input_attempts'] if r['arm']==arm]
        assert len(selected)+len(absent)==32
        assert {r['attempt_id'] for r in selected+absent}=={region+f'-P{i:02d}' for i in range(1,33)}
        records[arm]=selected;meshes[arm]={}
        for row in selected:
            if row['status'] not in ('EXPORT_SURFACE','EMPTY_EXPORT'):continue
            m=row['mesh'];assert sha(m['path'])==m['sha256']
            with np.load(m['path'],allow_pickle=False) as d:v,f=d['vertices_global_l2_xyz'],d['faces']
            assert len(f)==m['faces']
            if not len(f):continue
            assert np.isfinite(v).all();_,area=face_stats(v,f);assert (area>1e-10).all()
            meshes[arm][row['attempt_id']]={'v':v,'f':f,'soup':TriangleSoup(v,f),'whole_area_mm2':mm2(area)}
        unions[arm]=merge([m['soup'] for m in meshes[arm].values()])
    mp=REF/'reference-manifest.json';assert sha(mp)=='583c62bffd3bb02360556539b8c6d3280604f92d56aa5b02bf728d1ba9c76698'
    manifest=json.loads(mp.read_text());entry=next(e for e in manifest['assets'] if e['region']==region and e['asset']==asset)
    path=Path(entry['geometry']['path']);assert path.stat().st_size<=230000000 and sha(path)==entry['geometry']['sha256']
    d=np.load(path,allow_pickle=False);rv,rf=d['vertices_global_l2_xyz'],d['faces'];ref=TriangleSoup(rv,rf)
    result={'schema':'iss-spacing10-fixed-cohort-reference-union/1','status':'COMPLETE','region':region,'asset':asset,'geometry_index_sha256':sha(ip),'reference_geometry_sha256':sha(path),'source_sha256':sha(__file__),'domains':{},'attempt_accounting':{}}
    for arm in ARMS:
        expected=len(records[arm])
        result['attempt_accounting'][arm]={'original_attempts':32,'no_consumer_input':32-expected,'qualified_source_outputs':expected,'consumer_failed':sum(r['status']=='NO_EXPORT_CONSUMER_FAILED' for r in records[arm]),'empty_exports':sum(r['status']=='EMPTY_EXPORT' for r in records[arm]),'surface_exports':len(meshes[arm]),'all_records':[{k:r[k] for k in ['id','attempt_id','status','consumer_status']} for r in records[arm]]}
    for domain in ['B1408','A384']:
        weights=d[domain+'_clipped_area_vox2'];labels=d[domain+'_component'];boundary=d[domain+'_boundary'];active=weights>1e-10;idx=np.flatnonzero(active);points=d[domain+'_centroids_global_l2_xyz'][active];a=weights[active];l=labels[active];b=boundary[active];lo,hi=np.asarray(entry['domains'][domain]['box_global_l2_xyz'],float)
        assert np.isfinite(points).all()
        section={'primary':domain=='B1408','reference_boundary_excluded_mm2':mm2(a[~b]),'reference_boundary_mm2':mm2(a[b]),'box_global_l2_xyz':[lo.tolist(),hi.tolist()],'arms':{}}
        for arm in ARMS:
            union=unions[arm]
            distances=surface_distance(points,union,chunk=128).dist if union is not None else np.full(len(points),np.inf)
            if union is not None:assert np.isfinite(distances).all()
            status='INSUFFICIENT_REFERENCE' if not (~b).any() else 'NO_PRODUCT' if union is None else 'INCONCLUSIVE_NO_REFERENCE_INTERSECTION' if not np.any(distances[~b]<=8) else 'GEOMETRIC_SUPPORT_ONLY'
            row={'status':status,'all_clipped':measure(distances,a),'boundary_excluded':measure(distances[~b],a[~b]),'components':{},'outputs':[]}
            for component in np.unique(l):
                use=(l==component)&~b;row['components'][str(int(component))]=measure(distances[use],a[use])
            for identity,m in meshes[arm].items():
                cp,area,_,_,cb=domain_geometry(m['v'],m['f'],lo,hi);keep=area>1e-10
                clipped_mm2=mm2(area); outside_signed=m['whole_area_mm2']-clipped_mm2
                output={'attempt_id':identity,'whole_export_triangle_area_mm2':m['whole_area_mm2'],
                  'clipped_physical_mm2':clipped_mm2,'outside_domain_mm2':max(0.0,outside_signed),
                  'whole_minus_clipped_signed_mm2':outside_signed,
                  'outside_roundoff_note':'Outside is max(0,whole-clipped); signed difference retained. No tolerance or new gate. All areas gross per-output, not unique across patches.',
                  'candidate_boundary_mm2':mm2(area[cb]),'unknown_is_error':False,'outside_is_unscored_not_zero_unknown':True}
                if keep.any():
                    q=surface_distance(cp[keep],ref,chunk=128);known=(labels[q.face_idx]>0)&~boundary[q.face_idx];support=known&(q.dist<=8)
                    output.update(surface_to_reference=restrict_summary(q.dist,area[keep]),supported8_mm2=mm2(area[keep][support]),unknown_mm2=mm2(area[keep][~support]),known_support_curve_mm2={str(t):mm2(area[keep][known&(q.dist<=t)]) for t in TOLERANCES})
                else:output.update(status='OUTSIDE_DOMAIN_UNKNOWN',supported8_mm2=0.,unknown_mm2=0.)
                row['outputs'].append(output)
            section['arms'][arm]=row
            print(json.dumps({'region':region,'asset':asset,'domain':domain,'arm':arm,'covered4_mm2':row['boundary_excluded']['covered_area_mm2']['4.0'],'status':status}),flush=True)
        result['domains'][domain]=section
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_darwin_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      limits=['Reference union avoids duplicate coverage, but per-output area is gross.','Do not sum across assets; unknown is noterror, components are not winding identity.','No new optimization/ref/domain/threshold/seed; crop and registration uncertainty unchanged.','Seen full64 spacing10 followup after fixed3pilot; not a fresh independent validation.'])
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n';assert len(encoded.encode())<=2*1024**2,'perjob report storage cap'
    (ROOT/(region+'-'+asset+'-score.json')).write_text(encoded)
    print(json.dumps({'status':'COMPLETE','region':region,'asset':asset,'seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--region',choices=['V1','V2'],required=True);p.add_argument('--asset',required=True);a=p.parse_args();run(a.region,a.asset)
