"""Fixed three cached cases, frozen reference definitions; no grow or flatten."""
import argparse,gc,json,time,resource
from pathlib import Path
import numpy as np
import tifffile
from evaluate_patch import TriangleSoup,surface_distance,grid_triangles,all_quad,face_stats,sha,write_json,ORIGIN_XYZ,TOLERANCES
from evaluate_patch_batch import read_candidate,restrict_summary
from evaluate_expanded_references import domain_geometry
from diagnose_replayed_stages import reference
from score_frozen_validation import quads_area,mm2

BASE=Path('__HISTORICAL_WORK_ROOT__/iss-spacing10-followup-20261007')
OUT=Path('__HISTORICAL_DATA_ROOT__/spacing10_followup_20261007/reference')
REF=Path('__HISTORICAL_DATA_ROOT__/p4_validation_references_20261007')
ARMS=('existing10_native','support10_native','existing5_native')


def measure(d,w):
    r=restrict_summary(d,w);r['covered_area_mm2']={str(t):mm2(w[d<=t]) for t in TOLERANCES};return r


def load_case(case,protocol):
    mp=BASE/'fixed-cache-results/summary.json';assert sha(mp)==protocol['pilot_summary_sha256'];m=json.loads(mp.read_text())
    name='C0-'+case if case in ['P05','P29'] else case
    record=next(r for r in m['records'] if r['case']==name);assert record['inputs_hash_unchanged'] and record['raw20_bit_parity']
    meshes={};grids={}
    for arm in ARMS:
        a=record['arms'][arm];assert a['status']=='ACCEPTED'
        item={'path':a['path'],'files_sha256':a['files_sha256'],'frame':record['frame'],'origin_l2_zyx':record['origin_l2_zyx']}
        if case in ['P05','P29']:
            xyz,valid=read_candidate(item);xyz=xyz.astype(float);xyz[valid]+=ORIGIN_XYZ
        else:
            for n,h in a['files_sha256'].items():assert sha(Path(a['path'])/n)==h
            raw=np.stack([tifffile.imread(Path(a['path'])/(c+'.tif'),maxworkers=1) for c in 'xyz'],2)
            assert raw.shape[0]*raw.shape[1]<250000
            valid=(raw[...,0]>=0)&np.isfinite(raw).all(2);xyz=raw.astype(float);xyz[valid]+=np.asarray(record['origin_l2_zyx'][::-1],float)
        assert int(valid.sum())==a['valid_vertices'] and int(all_quad(valid).sum())==a['full_quads']
        v,f,_,_=grid_triangles(xyz,all_quad(valid));p,area=face_stats(v,f);good=area>1e-10
        meshes[arm]={'v':v,'f':f[good],'soup':TriangleSoup(v,f[good]) if good.any() else None,'surface_area_mm2':mm2(area[good]),'record':a};grids[arm]=(xyz,valid)
    x,va=grids['existing10_native'];y,vb=grids['support10_native'];assert x.shape==y.shape and np.array_equal(x[va&vb],y[va&vb])
    qa,qb=all_quad(va),all_quad(vb)
    quad={'retained':int((qa&qb).sum()),'lost':int((qa&~qb).sum()),'added':int((qb&~qa).sum()),'all_existing10_quads_retained':not bool((qa&~qb).any()),'lost_gross_mm2':mm2(quads_area(x)[qa&~qb]),'added_gross_mm2':mm2(quads_area(y)[qb&~qa]),'not_identity_truth':True}
    return record,meshes,quad


def evaluate(meshes,rv,rf,points,weights,labels,boundary,idx,box):
    ref=TriangleSoup(rv,rf);arms={}
    for arm,m in meshes.items():
        d=surface_distance(points,m['soup'],chunk=128).dist if m['soup'] is not None else np.full(len(points),np.inf)
        row={'status':'INSUFFICIENT_REFERENCE' if not len(weights) else 'NO_PRODUCT' if m['soup'] is None else 'GEOMETRIC_SUPPORT_ONLY' if np.any(d<=8) else 'INCONCLUSIVE_NO_REFERENCE_INTERSECTION','fixed_reference_to_candidate':measure(d,weights),'components':{str(int(c)):measure(d[labels[idx]==c],weights[labels[idx]==c]) for c in np.unique(labels[idx])},'surface_area_mm2':m['surface_area_mm2']}
        cp,area,_,_,cb=domain_geometry(m['v'],m['f'],box[0],box[1]);active=area>1e-10
        row.update(clipped_mm2=mm2(area),boundary_mm2=mm2(area[cb]),outside_mm2=m['surface_area_mm2']-mm2(area))
        if active.any():
            q=surface_distance(cp[active],ref,chunk=128);known=(labels[q.face_idx]>0)&~boundary[q.face_idx];support=known&(q.dist<=8)
            row.update(candidate_distance=restrict_summary(q.dist,area[active]),supported8_mm2=mm2(area[active][support]),unknown_mm2=mm2(area[active][~support]))
        else:row.update(supported8_mm2=0.,unknown_mm2=0.,candidate_distance={'status':'OUTSIDE_DOMAIN_UNKNOWN'})
        row['unknown_is_error']=False;arms[arm]=row
        print(json.dumps({'arm':arm,'covered4_mm2':row['fixed_reference_to_candidate']['covered_area_mm2']['4.0'],'unknown_mm2':row['unknown_mm2']}),flush=True)
    return {'reference_denominator_mm2':mm2(weights),'arms':arms}


def numeric_equal(a,b,path=''):
    if isinstance(a,dict):
        for k,v in a.items():numeric_equal(v,b[k],path+'/'+k)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(v,w) in enumerate(zip(a,b)):numeric_equal(v,w,path+'/'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert abs(float(a)-float(b))<=1e-9,(path,a,b)
    else:assert a==b,(path,a,b)


def check_expected(case,asset,domain,row,expected):
    if case in ['P05','P29']:
        ex=expected[case]
        numeric_equal(ex['reference_covered_area_mm2'],row['fixed_reference_to_candidate']['covered_area_mm2'],'coverage')
        for k in ['unknown_mm2','supported8_mm2','clipped_mm2']:numeric_equal(ex[k],row[k],k)
        numeric_equal(ex['candidate_distance'],row['candidate_distance'],'distance')
    else:
        ex=expected[case+'-'+asset][domain]
        numeric_equal(ex['fixed_reference_to_surface']['covered_area_mm2'],row['fixed_reference_to_candidate']['covered_area_mm2'],'coverage')
        for k in ['unknown_mm2','supported8_mm2']:numeric_equal(ex[k],row[k],k)
        numeric_equal(ex['clipped_surface_mm2'],row['clipped_mm2'],'clipped')
        if 'surface_to_reference' in ex:numeric_equal(ex['surface_to_reference'],row['candidate_distance'],'distance')


def run(case,asset):
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    pp=BASE/'reference-protocol.json';protocol=json.loads(pp.read_text());ep=BASE/'existing10-expected-reference.json';assert sha(ep)==protocol['existing10_control_check']['expected_file_sha256'];expected=json.loads(ep.read_text())
    record,meshes,quad=load_case(case,protocol)
    result={'schema':'iss-spacing10-fixed-cache-reference-evaluation/1','status':'COMPLETE_CACHED_STAGE_ONLY','case':case,'asset':asset,'protocol_sha256':sha(pp),'source_sha256':sha(__file__),'pilot_summary_sha256':protocol['pilot_summary_sha256'],'source_record':record,'same_grid10_quad_comparison':quad,'domains':{},'all_existing10_cache_metric_checks_passed':False,'end_to_end_claim':False,'selection':'seen development/mechanism cases, not independent validation'}
    if case in ['P05','P29']:
        rv,rf,points,weights,labels,boundary,idx,box,info=reference(case);result['reference']=info
        domain=info['domain'];section=evaluate(meshes,rv,rf,points,weights,labels,boundary,idx,box);check_expected(case,None,domain,section['arms']['existing10_native'],expected);result['domains'][domain]=section
    else:
        mp=REF/'reference-manifest.json';assert sha(mp)=='583c62bffd3bb02360556539b8c6d3280604f92d56aa5b02bf728d1ba9c76698'
        entry=next(e for e in json.loads(mp.read_text())['assets'] if e['region']=='V1' and e['asset']==asset);rp=Path(entry['geometry']['path']);assert sha(rp)==entry['geometry']['sha256'];result['reference']=entry['geometry']
        with np.load(rp,allow_pickle=False) as f:
            rv,rf=f['vertices_global_l2_xyz'],f['faces']
            for domain in ['B1408','A384']:
                a=f[domain+'_clipped_area_vox2'];labels=f[domain+'_component'];boundary=f[domain+'_boundary'];idx=np.flatnonzero((a>1e-10)&~boundary);points=f[domain+'_centroids_global_l2_xyz'][idx];box=np.asarray(entry['domains'][domain]['box_global_l2_xyz'],float)
                section=evaluate(meshes,rv,rf,points,a[idx],labels,boundary,idx,box);check_expected(case,asset,domain,section['arms']['existing10_native'],expected);result['domains'][domain]=section
    result.update(all_existing10_cache_metric_checks_passed=True,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=protocol['limits'])
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n';assert len(encoded.encode())<1048576
    name=case+('-'+asset if asset else '')+'-reference.json';(OUT/name).write_text(encoded);print(json.dumps({'status':result['status'],'case':case,'asset':asset,'seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['P05','P29','V1-P06'],required=True);p.add_argument('--asset');a=p.parse_args();run(a.case,a.asset)
