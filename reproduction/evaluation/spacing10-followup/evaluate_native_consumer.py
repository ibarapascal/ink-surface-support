"""Read fixed real native10/support10/5 flatten outputs using validated diagnostics."""
import argparse,gc,json,time,resource,hashlib
from pathlib import Path
import numpy as np
import torch
from flatten_cohort_geometry import inspect_export
from evaluate_references import evaluate,numeric_equal
from evaluate_patch import sha,write_json
from diagnose_replayed_stages import reference

BASE=Path('__HISTORICAL_WORK_ROOT__/iss-spacing10-followup-20261007')
CONSUMER=Path('__HISTORICAL_WORK_ROOT__/iss-spacing10-consumer-20261007')
OUT=Path('__HISTORICAL_DATA_ROOT__/spacing10_followup_20261007/consumer')
REF=Path('__HISTORICAL_DATA_ROOT__/p4_validation_references_20261007')
ARMS=('existing10_native','support10_native','existing5_native')


def qualify(case,arm):
    manifest=CONSUMER/'consumer-manifest.json'
    assert sha(manifest)=='0abf050e6ad46e22feb4f081139551591f63e8e0d5448104c335c77b0edfd79f'
    path=CONSUMER/'cases'/(case+'-'+arm)/'result.json';q=json.loads(path.read_text())
    assert q['manifest_sha256']==sha(manifest)
    assert q['status']=='SUCCESS' and q['scientific_calls']==1
    assert q['case']==case and q['arm']==arm and q['native_spacing_verified']
    assert q['source_config_input_hashes_verified_before'] and q['source_config_hashes_verified_after'] and q['input_hashes_unchanged']
    assert q['declared_stage_steps']==[['0','1000'],['1','1000'],['2','1000']]
    assert [x[0] for x in q['stage_completions']]==['0','1','2']
    root=Path(q['checkpoint']).parent;assert Path(q['output_path'])==root/'tifxyz/flatten.tifxyz'
    for f in q['files']:
        p=root/f['relative'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    for c,h in q['input']['files_sha256'].items():assert sha(Path(q['input']['path'])/c)==h
    assert sha(q['stdout_path'])==q['stdout_sha256']
    return q,sha(path)


def compact(m):
    return {'v':m['v'],'f':m['f'],'soup':m['soup'],'surface_area_mm2':float(m['area'].sum()*9.6**2/1e6)}


def score_stages(stages,rv,rf,p,w,l,b,idx,box):
    unique={};keys={}
    for arm,ss in stages.items():
        for name,m in ss.items():
            h=hashlib.sha256(m['v'].tobytes()+m['f'].tobytes()).hexdigest();keys[(arm,name)]=h
            if h not in unique:unique[h]=compact(m)
    scored=evaluate(unique,rv,rf,p,w,l,b,idx,box)
    return {'reference_denominator_mm2':scored['reference_denominator_mm2'],'arms':{a:{s:scored['arms'][keys[(a,s)]] for s in stages[a]} for a in ARMS}}


def check_source(section,expected):
    for arm in ARMS:
        got=section['arms'][arm]['source_complete'];old=expected['arms'][arm]
        numeric_equal(old['fixed_reference_to_candidate']['covered_area_mm2'],got['fixed_reference_to_candidate']['covered_area_mm2'],'source/reference coverage')
        for key in ['supported8_mm2','unknown_mm2','clipped_mm2']:numeric_equal(old[key],got[key],'source/'+key)
        if old['candidate_distance'].get('status')!='OUTSIDE_DOMAIN_UNKNOWN':numeric_equal(old['candidate_distance'],got['candidate_distance'],'source/distance')


def run(case,asset):
    t=time.monotonic();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    pp=BASE/'native-consumer-evaluation-protocol.json';protocol=json.loads(pp.read_text())
    for name,h in protocol['source_pointer_hashes'].items():assert sha(Path('__HISTORICAL_WORK_ROOT__/iss-patch-eval-20261007')/name)==h
    reports={};stages={};qualification={}
    for arm in ARMS:
        q,h=qualify(case,arm);reports[arm],stages[arm]=inspect_export(q['checkpoint'],q['origin_l2_zyx']);qualification[arm]={'result_sha256':h,'source_config_input_hash_verified':True,'native_spacing':q['expected_native_output_spacing'],'seconds':q['seconds'],'rss_bytes':q['maximum_resident_set_size_bytes'],'prior_existing10_export_comparison':q.get('prior_existing10_export_comparison')}
        print(json.dumps({'case':case,'arm':arm,'area':reports[arm]['physical_area_mm2'],'source_to_export':reports[arm]['eligible_source_to_export']['coverage']}),flush=True)
    short=case.removeprefix('C0-');name=short+('-'+asset if asset else '')
    prepath=Path('__HISTORICAL_DATA_ROOT__/spacing10_followup_20261007/reference')/(name+'-reference.json');expected=json.loads(prepath.read_text())
    out={'schema':'iss-spacing10-fixed-native-consumer-evaluation/1','case':case,'asset':asset,'status':'COMPLETE_SEEN_CASE_NATIVE_CONSUMER','protocol_sha256':sha(pp),'source_sha256':sha(__file__),'source_reference_score_sha256':sha(prepath),'arms':reports,'qualification':qualification,'domains':{},'all_preconsumer_source_metrics_verified':False,'new_optimizer_calls':0,'not_independent_holdout':True}
    if case.startswith('C0-'):
        rv,rf,p,w,l,b,idx,box,info=reference(short);dom=info['domain'];section=score_stages(stages,rv,rf,p,w,l,b,idx,box);check_source(section,expected['domains'][dom]);out['domains'][dom]=section;out['reference']=info
    else:
        mp=REF/'reference-manifest.json';assert sha(mp)=='583c62bffd3bb02360556539b8c6d3280604f92d56aa5b02bf728d1ba9c76698';entry=next(x for x in json.loads(mp.read_text())['assets'] if x['region']=='V1' and x['asset']==asset);rp=Path(entry['geometry']['path']);assert sha(rp)==entry['geometry']['sha256'];out['reference']=entry['geometry']
        with np.load(rp,allow_pickle=False) as d:
            rv,rf=d['vertices_global_l2_xyz'],d['faces']
            for dom in ['B1408','A384']:
                a=d[dom+'_clipped_area_vox2'];l=d[dom+'_component'];b=d[dom+'_boundary'];idx=np.flatnonzero((a>1e-10)&~b);p=d[dom+'_centroids_global_l2_xyz'][idx];box=np.asarray(entry['domains'][dom]['box_global_l2_xyz'],float);section=score_stages(stages,rv,rf,p,a[idx],l,b,idx,box);check_source(section,expected['domains'][dom]);out['domains'][dom]=section
    out.update(all_preconsumer_source_metrics_verified=True,elapsed_seconds=time.monotonic()-t,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=protocol['scope_limits'])
    text=json.dumps(out,indent=2,allow_nan=False)+'\n';assert len(text.encode())<2*1024**2;(OUT/(name+'-native-consumer.json')).write_text(text);print(json.dumps({'status':out['status'],'case':case,'seconds':out['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['C0-P05','C0-P29','V1-P06'],required=True);p.add_argument('--asset');a=p.parse_args();run(a.case,a.asset)
