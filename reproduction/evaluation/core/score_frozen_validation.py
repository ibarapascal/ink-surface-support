"""Frozen V1/V2 validation: one compact reference per job, exact merged unions."""
from __future__ import annotations
import argparse,gc,json,resource,time
from pathlib import Path
import numpy as np
import tifffile
from evaluate_patch import TriangleSoup,surface_distance,all_quad,grid_triangles,face_stats,sha,write_json,write_npz,UM_PER_VOX,TOLERANCES
from evaluate_patch_batch import restrict_summary
from evaluate_expanded_references import domain_geometry

HELPER='473f7d70797123d2ecc04db797084b0b73b4892dd4e241a3a69daffc5712776c'
BASE=Path('__HISTORICAL_WORK_ROOT__/iss-validation-producer-20261007')
REF=Path('__HISTORICAL_DATA_ROOT__/p4_validation_references_20261007')
OUT=Path('__HISTORICAL_DATA_ROOT__/p4_patch_eval_20261007/holdout')


def mm2(values):return float(np.sum(values)*UM_PER_VOX**2/1e6)


def measure(dist,weight):
    r=restrict_summary(dist,weight)
    r['covered_area_mm2']={str(t):mm2(weight[dist<=t]) for t in TOLERANCES}
    return r


def read_patch(item,origin):
    assert item['frame']=='local_l2_xyz' and item['origin_l2_zyx']==origin
    path=Path(item['path'])
    for name,h in item['files_sha256'].items():assert sha(path/name)==h
    arrays=[]
    for name in 'xyz':
        with tifffile.TiffFile(path/(name+'.tif')) as file:
            shape=file.series[0].shape
            if len(shape)!=2 or np.prod(shape)>250000:raise ValueError('Candidate grid requires new resource review')
        arrays.append(tifffile.imread(path/(name+'.tif'),maxworkers=1))
    assert len({a.shape for a in arrays})==1
    local=np.stack(arrays,2)
    valid=(local[...,0]>=0)&np.isfinite(local).all(2)
    assert int(valid.sum())==item['valid_vertices']
    assert int(all_quad(valid).sum())==item['full_quads']
    # Promote BEFORE translation: no extra float32 origin-rounding.
    global_xyz=local.astype(np.float64)
    global_xyz[valid]+=np.asarray(origin[::-1],float)
    return local,global_xyz,valid


def quads_area(grid):
    tl,tr,bl,br=grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1],grid[1:,1:]
    return (np.linalg.norm(np.cross(tl-bl,tr-bl),axis=2)+np.linalg.norm(np.cross(tr-bl,br-bl),axis=2))/2


def merge(soups):
    vv=[];ff=[];offset=0
    for soup in soups:
        used,inverse=np.unique(soup.faces,return_inverse=True)
        v=soup.vertices[used];f=inverse.reshape(-1,3)
        vv.append(v);ff.append(f+offset);offset+=len(v)
    return TriangleSoup(np.concatenate(vv),np.concatenate(ff)) if vv else None


def run(region,asset,protocol):
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    rules=json.loads(Path(protocol).read_text());region_rule=rules['exact_domains_freeze']['regions'][region]
    origin=region_rule['input_box_global_L2_zyx'][0]
    index=json.loads((REF/'reference-manifest.json').read_text())
    entry=next(a for a in index['assets'] if a['region']==region and a['asset']==asset)
    path=Path(entry['geometry']['path']);assert path.stat().st_size<=230000000
    assert sha(path)==entry['geometry']['sha256']
    # Read only compact halo geometry and one domain at a time, not31M sourcefaces.
    d=np.load(path,allow_pickle=False);rv=d['vertices_global_l2_xyz'];rf=d['faces']
    assert list(rf.shape)==entry['npz_header_audit']['faces.npy']['shape']
    assert len(rf)<=1200000
    ref=TriangleSoup(rv,rf)
    cohorts={};geometry={};unions={};gross=[]
    for arm in ('old20','candidate20','existing10'):
        mp=BASE/region/'paired'/(arm+'-manifest.json');m=json.loads(mp.read_text())
        assert m['status']=='COMPLETE' and m['case']==region and m['completed_attempts']==32
        assert m['helper_sha256']==HELPER and m['origin_l2_zyx']==origin
        cohorts[arm]={'manifest':m,'sha256':sha(mp)};patches={}
        for item in m['patches']:
            local,xyz,valid=read_patch(item,origin)
            v,f,_,_=grid_triangles(xyz,all_quad(valid));_,area=face_stats(v,f);f=f[area>1e-10]
            if len(f):patches[item['attempt_id']]={'record':item,'local':local,'xyz':xyz,'valid':valid,'vertices':v,'faces':f,'soup':TriangleSoup(v,f)}
        geometry[arm]=patches;unions[arm]=merge([p['soup'] for p in patches.values()])
    assert len({cohorts[a]['manifest']['prefix_seal_sha256'] for a in cohorts})==1
    assert [a['seed_row'] for a in cohorts['old20']['manifest']['attempts']]==[a['seed_row'] for a in cohorts['candidate20']['manifest']['attempts']]
    for aid in sorted(set(geometry['old20'])|set(geometry['candidate20'])):
        a=geometry['old20'].get(aid);b=geometry['candidate20'].get(aid)
        if not a or not b:
            qa=all_quad(a['valid']) if a else np.zeros((0,0),bool)
            qb=all_quad(b['valid']) if b else np.zeros((0,0),bool)
            gross.append({'attempt_id':aid,'paired_surface_available':False,
                'old_quads':int(qa.sum()),'candidate_quads':int(qb.sum()),'retained_quads':0,
                'lost_old_quads':int(qa.sum()),'added_quads':int(qb.sum()),
                'old_quads_all_included':not bool(qa.any()),'retained_gross_area_mm2':0.,
                'lost_gross_area_mm2':mm2(quads_area(a['xyz'])[qa]) if a else 0.,
                'added_gross_area_mm2':mm2(quads_area(b['xyz'])[qb]) if b else 0.,
                'gross_is_not_unique_surface_area':True});continue
        assert a['local'].shape==b['local'].shape
        shared=a['valid']&b['valid'];assert a['local'][shared].tobytes()==b['local'][shared].tobytes()
        qa,qb=all_quad(a['valid']),all_quad(b['valid']);aa,ab=quads_area(a['xyz']),quads_area(b['xyz'])
        gross.append({'attempt_id':aid,'paired_surface_available':True,'shared_XYZ_bytes_equal':True,
          'old_quads':int(qa.sum()),'candidate_quads':int(qb.sum()),'retained_quads':int((qa&qb).sum()),
          'lost_old_quads':int((qa&~qb).sum()),'added_quads':int((qb&~qa).sum()),
          'old_quads_all_included':bool(not np.any(qa&~qb)),
          'retained_gross_area_mm2':mm2(aa[qa&qb]),'lost_gross_area_mm2':mm2(aa[qa&~qb]),
          'added_gross_area_mm2':mm2(ab[qb&~qa]),'gross_is_not_unique_surface_area':True})
    result={'schema':'iss-frozen-holdout-reference-score/1','status':'COMPLETE','region':region,'asset':asset,
      'source_sha256':sha(__file__),'protocol_sha256':sha(protocol),'reference_manifest_sha256':sha(REF/'reference-manifest.json'),
      'geometry_sha256':entry['geometry']['sha256'],'geometry_header_audit':entry['npz_header_audit'],
      'reference_registration_precision_independently_verified':False,'component_is_winding':False,
      'arms':{a:{'manifest_sha256':c['sha256'],'attempts':c['manifest']['attempts'],
                 'outputs':len(geometry[a]),'coordinate_TIFF_bytes':sum(p['coordinate_bytes'] for p in c['manifest']['patches']),
                 'valid_vertices':sum(p['valid_vertices'] for p in c['manifest']['patches']),
                 'full_quads':sum(p['full_quads'] for p in c['manifest']['patches'])} for a,c in cohorts.items()},
      'old_candidate_quad_accounting':gross,'domains':{},'union_implementation':'One mergedTriangleSoup ofallvalidnondegeneratefaces, originalsurface_distance unchanged;no distancecensoring'}
    for domain,rule_key in [('B1408','primary_B1408_global_L2_xyz'),('A384','secondary_A384_global_L2_xyz')]:
        lo,hi=np.asarray(region_rule[rule_key],float)
        assert entry['domains'][domain]['box_global_l2_xyz']==region_rule[rule_key]
        area=d[domain+'_clipped_area_vox2'];labels=d[domain+'_component'];boundary=d[domain+'_boundary']
        active=area>1e-10;idx=np.flatnonzero(active);points=d[domain+'_centroids_global_l2_xyz'][active]
        assert np.isfinite(points).all();primary=~boundary[active]
        report={'bbox_global_L2_xyz':[lo.tolist(),hi.tolist()],
                'clipped_area_mm2':mm2(area[active]),'boundary_excluded_area_mm2':mm2(area[active][primary]),
                'boundary_area_mm2':mm2(area[active][~primary]),'arms':{}}
        for arm in cohorts:
            u=unions[arm]
            dist=surface_distance(points,u,chunk=128).dist if u is not None and len(points) else np.full(len(points),np.inf)
            if u is not None:assert np.isfinite(dist).all()
            status=('INSUFFICIENT_REFERENCE' if not len(points) or not primary.any()
                    else 'NO_PRODUCT' if u is None else 'SCORED_GEOMETRY_ONLY'
                    if np.any(dist[primary]<=max(TOLERANCES))
                    else 'INCONCLUSIVE_NO_KNOWN_REFERENCE_INTERSECTION')
            ar={'status':status,
                'all_clipped':measure(dist,area[active]),'boundary_excluded':measure(dist[primary],area[active][primary]),
                'components':{},'outputs':[]}
            for c in np.unique(labels[active]):
                cm=labels[active]==c;pm=cm&primary
                ar['components'][str(int(c))]={'all_clipped':measure(dist[cm],area[active][cm]),'boundary_excluded':measure(dist[pm],area[active][pm])}
            for aid,p in geometry[arm].items():
                cp,ca,_,_,cb=domain_geometry(p['vertices'],p['faces'],lo,hi);take=ca>1e-10
                row={'attempt_id':aid,'candidate_clipped_area_mm2':mm2(ca),'candidate_domain_boundary_area_mm2':mm2(ca[cb]),'unknown_is_error':False}
                if take.any():
                    nearest=surface_distance(cp[take],ref,chunk=128)
                    known=(labels[nearest.face_idx]>0)&~boundary[nearest.face_idx]
                    support=known&(nearest.dist<=8)
                    row.update(candidate_to_reference=restrict_summary(nearest.dist,ca[take]),
                        supported8_area_mm2=mm2(ca[take][support]),unknown_area_mm2=mm2(ca[take][~support]),
                        nearest_reference_boundary_area_mm2=mm2(ca[take][boundary[nearest.face_idx]]))
                else:row.update(status='OUTSIDE_DOMAIN_UNKNOWN',supported8_area_mm2=0.,unknown_area_mm2=0.)
                ar['outputs'].append(row)
            sample=OUT/f'{region}-{asset}-{domain}-{arm}-distances.npz'
            write_npz(sample,reference_face_idx=idx,area_vox2=area[active],component=labels[active],boundary=boundary[active],distance_vox=dist)
            ar['distance_samples_sha256']=sha(sample);report['arms'][arm]=ar
            print(json.dumps({'region':region,'asset':asset,'domain':domain,'arm':arm,'covered4_mm2':ar['boundary_excluded']['covered_area_mm2']['4.0']}),flush=True)
        result['domains'][domain]=report
        del area,labels,boundary,points;gc.collect()
    result.update(elapsed_seconds=time.monotonic()-started,peak_rss_darwin_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      limits=['B1408primary;A384context-localityonly,nofallback','Referenceassets/componentsnotindependentwindinglabels;nocrossassetareatotal','Unknown/outside-refisnoterror;boundaries/cropandregistrationlimitsremain','Quadareaaccountsaregrossperfile,nouniquephysicalarea','No method/threshold/seed/ref/domain changes'])
    write_json(OUT/f'{region}-{asset}-score.json',result)
    print(json.dumps({'status':'COMPLETE','region':region,'asset':asset,'seconds':result['elapsed_seconds']}),flush=True)


def measure(dist,weight):
    r=restrict_summary(dist,weight);r['covered_area_mm2']={str(t):mm2(weight[dist<=t]) for t in TOLERANCES};return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--region',choices=['V1','V2'],required=True);p.add_argument('--asset',choices=['20231210121321','20230702185753','20230929220926'],required=True);p.add_argument('--protocol',required=True);a=p.parse_args();run(a.region,a.asset,a.protocol)
