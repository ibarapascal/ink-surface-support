#!/usr/bin/env python3
"""Offline REPORT reconstruction only. Does not run scientific evaluation or fetch data."""
from __future__ import annotations
import argparse,ast,csv,hashlib,io,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TOLS=('0.5','1.0','2.0','4.0','8.0')

def need(ok,msg='consistency check failed'):
 if not ok:raise SystemExit('FAIL: '+msg)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9)
def csv_text(fields,rows):
 buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return buf.getvalue()
def ref_values(arm):return arm['reference']['covered_area_mm2']

def validate(e,m):
 need(digest(ROOT/'results/evidence.json')==m['evidence_sha256'],'Evidence bytes differ from source manifest')
 ids={s['id'] for s in m['original_sources']};need(len(ids)==len(m['original_sources']))
 source_index=json.loads((ROOT/'reproduction/evaluation-source-manifest.json').read_text())
 same_ast_version=list(sys.version_info[:2])==source_index.get('ast_fingerprint_python',[])[:2]
 for rec in source_index['files']:
  path=ROOT/'reproduction/evaluation'/rec['file'];need(digest(path)==rec['shareable_sha256'],'evaluation source hash differs: '+rec['file'])
  if same_ast_version:
   tree=ast.parse(path.read_text());tree_hash=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
   need(tree_hash==rec['shareable_module_AST_sha256'],'evaluation source AST differs: '+rec['file'])
   need(tree_hash==rec.get('expected_transformed_module_AST_sha256',rec['normalized_original_module_AST_sha256']))
   funcs=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))];need(len(funcs)==len(rec['functions']))
   for node,fn in zip(funcs,rec['functions']):
    actual=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest();need(actual==fn['shareable_AST_sha256'])
    if fn['exact_AST_equal']:need(actual==fn['original_AST_sha256'])
 for rec in source_index['unmodified_upstream']:
  need(digest(ROOT/'reproduction'/rec['file'])==rec['sha256'])
  need((ROOT/'reproduction'/rec['license_file']).is_file(),'Upstream license notice missing')
 need(e['main_counts']['consumer_outputs']==195 and e['main_counts']['successful_source_pairs']==65)
 need(len(e['consumer_diagnostics'])==195 and len(e['native_noinput'])==93)
 for region in ('V1','V2','V3'):
  for arm in ('existing10_native','support10_native','existing5_native'):
   accepted=[r for r in e['consumer_diagnostics'] if r['region']==region and r['arm']==arm]
   absent=[r for r in e['native_noinput'] if r['region']==region and r['arm']==arm]
   need(len(accepted)+len(absent)==32)
   need({r['case'] for r in accepted+absent}=={f'{region}-P{i:02d}' for i in range(1,33)})
 for row in e['coverage_rows']:
  need(row['source_id'] in ids);den=row['reference_area_mm2'];need(den>=0)
  for a in row['arms'].values():
   vals=ref_values(a);need(set(vals)==set(TOLS))
   yy=[vals[t] for t in TOLS];need(all(math.isfinite(x) and 0<=x<=den+1e-9 for x in yy))
   need(all(a<=b+1e-9 for a,b in zip(yy,yy[1:])),'Tolerance curve must be monotone')
   if den==0:need(not any(yy) and a['status']=='INSUFFICIENT_REFERENCE')
   if a['status']=='INCONCLUSIVE_NO_REFERENCE_INTERSECTION':need(not any(yy))
   comp=a.get('components',{})
   if comp:
    need(close(sum(c.get('weighted_area_mm2',0) for c in comp.values()),den))
    for t in TOLS:need(close(sum(c['covered_area_mm2'][t] for c in comp.values()),vals[t]),'Component conservation failed')
   for o in a.get('output_gross',[]):
    need(close(o['supported8_mm2']+o['unknown_mm2'],o['clipped_physical_mm2']))
    need(close(o['whole_export_triangle_area_mm2']-o['clipped_physical_mm2'],o['outside_domain_mm2']))
 for group in e['paired_distortion']:
  need(group['exact_pairs']==len(group['cases']) and group['unmatched_pairs']==0)
  for c in group['cases']:
   need(c['exact_shared_XYZ'] and c['old_valid_vertices_retained'])
   old,new,delta=(c[k] for k in ('original_common','support_common','paired_delta_support_minus_original'))
   need(close(old['source_area_mm2'],new['source_area_mm2']))
   need(close(new['area_weighted_mean']-old['area_weighted_mean'],delta['area_weighted_mean']))
   for s in (old,new,delta):
    q=s['area_weighted_p50_p90_p95'];need(q==sorted(q))
 main=next(x for x in e['paired_distortion'] if x['study']=='main10-common65')
 need(main['pooled']['original_common']['n']==72066)
 need(close(main['pooled']['original_common']['source_area_mm2'],664.2356142380206))
 need(all(c['paired_weighted_mean_delta']>0 for c in main['cases']))
 # Independent explicit anchors transcribed from the retained raw-score JSONs.
 anchors=[('main10','V1',None,'20231210121321','B1408','support10_native','4.0',40.562754061553164),('main10','V3',None,'20230702185753','B1408','support10_native','4.0',15.427895358288037),('main10','V3',None,'20230929220926','B1408','support10_native','8.0',0.0),('spacing5-development','C0','C0-P05','20230702185753','existing_exploratory_B896','support5','4.0',1.2893267999065225),('spacing5-development','C0','C0-P29','20231210121321','original_frozen_core384','support5','8.0',2.0308199054844427)]
 for study,region,case,asset,domain,arm,tol,value in anchors:
  row=next(r for r in e['coverage_rows'] if (r['study'],r['region'],r['case'],r['asset'],r['domain'])==(study,region,case,asset,domain));need(ref_values(row['arms'][arm])[tol]==value)
 return {'AST_fingerprint_rechecked':same_ast_version,'AST_fingerprint_python':source_index.get('ast_fingerprint_python'),'byte_hashes_checked_on_all_versions':True,'coverage_rows':len(e['coverage_rows']),'main_consumer_outputs':195,'main_native_noinput_slots':93,'paired_main_cells':72066,'checks':['source hash','attempt accounting','all five thresholds','monotonicity','reference denominators','component conservation','unknown/inside/outside conservation','paired shared denominator and delta','explicit raw-score anchors','published numerical-source hashes and AST fingerprints'],'scope':'Report integrity and arithmetic consistency, not independent scientific validation.'}

def build(e):
 coverage=[]
 for r in e['coverage_rows']:
  for arm,a in r['arms'].items():
   den=r['reference_area_mm2'];gross=a.get('output_gross',[])
   for t in TOLS:
    v=ref_values(a)[t];coverage.append({'study':r['study'],'region':r['region'],'case':r['case'] or '', 'asset':r['asset'],'domain':r['domain'],'domain_label':e['domain_definitions'][r['domain']]['label'],'arm':arm,'arm_label':e['arm_definitions'][arm]['label'],'tolerance_input_voxels':t,'reference_area_mm2':den,'covered_reference_mm2':v,'fraction':v/den if den>0 else 'NA','status':a['status'],'gross_unknown_mm2':math.fsum(x['unknown_mm2'] for x in gross) if gross else a.get('unknown_mm2',''),'gross_outside_mm2':math.fsum(x['outside_domain_mm2'] for x in gross) if gross else '', 'source_id':r['source_id']})
 paired=[]
 for group in e['paired_distortion']:
  for c in group['cases']:
   old,new,delta=[c[k] for k in ('original_common','support_common','paired_delta_support_minus_original')]
   paired.append({'study':group['study'],'case':c['case'],'baseline_label':e['arm_definitions']['existing10_native' if group['study']=='main10-common65' else 'original5']['label'],'candidate_label':e['arm_definitions']['support10_native' if group['study']=='main10-common65' else 'support5']['label'],'fixed_original_area_mm2':c['fixed_original_area_mm2'],'paired_area_mm2':c['paired_area_mm2'],'paired_quads':c['paired_quad_count'],'original_mean':old['area_weighted_mean'],'support_mean':new['area_weighted_mean'],'paired_delta_mean':delta['area_weighted_mean'],'original_median':old['area_weighted_p50_p90_p95'][0],'support_median':new['area_weighted_p50_p90_p95'][0],'original_p90':old['area_weighted_p50_p90_p95'][1],'support_p90':new['area_weighted_p50_p90_p95'][1],'original_p95':old['area_weighted_p50_p90_p95'][2],'support_p95':new['area_weighted_p50_p90_p95'][2],'original_ineligible_mm2':c['original_ineligible_fixed_area_mm2'],'support_ineligible_mm2':c['support_ineligible_fixed_area_mm2'],'source_id':group['source_id']})
 costfields=['study','region','case','arm','arm_label','cpu_seconds','wall_seconds','all_files_bytes','export_payload_bytes','coordinate_tiff_bytes','max_process_rss_bytes','precision_evidence','execution_wrapper_state','timing_scope','source_id']
 costrows=[{f:(e['arm_definitions'][r['arm']]['label'] if f=='arm_label' else r.get(f,'')) for f in costfields} for r in e['costs']]
 diag=[]
 for r in e['consumer_diagnostics']:
  diag.append({'region':r['region'],'case':r['case'],'arm':r['arm'],'arm_label':e['arm_definitions'][r['arm']]['label'],'source_quads':r['source_complete_quads'],'export_quads':r['all_stage_geometry']['export']['quads'],'gross_source_mm2':r['physical_area_mm2']['source_complete'],'gross_export_mm2':r['physical_area_mm2']['export'],'angle_rejected_quads':r['source_complete_quads']-r['source_angle_filter_retained_quads'],'det_rejected_quads':r['source_angle_filter_retained_quads']-r['source_export_eligible_quads'],'export_zero_area_triangles':r['all_stage_geometry']['export']['zeroarea_triangles'],'source_SDir_mean':r['native_SDir']['area_weighted_diagnostic'].get('area_weighted_mean',''),'native32_vs64_mask_equal':r['native32_vs64_mask_audit']['masks_equal'],'source_id':r['source_id']})
 summary={'schema':1,'scope':'Tables reconstructed from recorded measurements.','arm_definitions':e['arm_definitions'],'domain_definitions':e['domain_definitions'],'main_counts':e['main_counts'],'main_B4':[{'region':r['region'],'asset':r['asset'],'denominator_mm2':r['reference_area_mm2'],'arms':{a:ref_values(v)['4.0'] for a,v in r['arms'].items()},'statuses':{a:v['status'] for a,v in r['arms'].items()}} for r in e['coverage_rows'] if r['study']=='main10' and r['domain']=='B1408'],'paired_distortion':[{'study':g['study'],'pairs':len(g['cases']),'cases_with_positive_mean_delta':sum(c['paired_weighted_mean_delta']>0 for c in g['cases']),'pooled':g['pooled']} for g in e['paired_distortion']],'limitations':e['limitations']}
 return {'summary.json':json.dumps(summary,indent=2,allow_nan=False)+'\n','reference-coverage.csv':csv_text(list(coverage[0]),coverage),'paired-distortion.csv':csv_text(list(paired[0]),paired),'cost.csv':csv_text(costfields,costrows),'consumer-diagnostics.csv':csv_text(list(diag[0]),diag),'native-noinput.csv':csv_text(list(e['native_noinput'][0])+['arm_label'],[dict(r,arm_label=e['arm_definitions'][r['arm']]['label']) for r in e['native_noinput']])}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true',help='Verify evidence and exact generated table contents; does not write.');parser.add_argument('--out',type=Path,help='Write derived report files here (or check this directory). Input evidence is never overwritten.');args=parser.parse_args()
 e=json.loads((ROOT/'results/evidence.json').read_text());m=json.loads((ROOT/'results/source-manifest.json').read_text());checks=validate(e,m);outputs=build(e);dest=args.out or ROOT/'results'
 if args.check:
  for name,text in outputs.items():need((dest/name).read_text()==text,f'Generated file differs: {name}')
 elif args.out:
  dest.mkdir(parents=True,exist_ok=True)
  for name,text in outputs.items():(dest/name).write_text(text)
 print(json.dumps({'status':'PASS','mode':'check' if args.check else 'write' if args.out else 'validate','derived_files':list(outputs),'validation':checks},indent=2))
if __name__=='__main__':main()
