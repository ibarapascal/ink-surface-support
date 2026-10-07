"""Read-only existing references in frozen held-out boxes; no candidate access."""
import argparse,gc,json,os,resource,time
from pathlib import Path
import numpy as np
import tifffile
from expanded_references import digest,write_json,selftest
from evaluate_expanded_references import domain_geometry
RAM_CAP=1610612736
INTERMEDIATE_CAP=2147483648
COMPACT_CAP=1610612736

def read_source(source):
 root=Path(source['source_root'])
 for name,h in source['hashes'].items():
  if digest(root/name)!=h:raise ValueError('Originalreference hash mismatch '+str(root/name))
 arrays=[tifffile.imread(root/f'{axis}.tif',maxworkers=1) for axis in 'xyz']
 if any(a.shape!=tuple(source['shape']) or a.dtype!=np.float32 for a in arrays):raise ValueError('Originalshape/dtype changed')
 valid=(arrays[2]>0)&np.isfinite(arrays[2])
 if np.any(valid&(~np.isfinite(arrays[0])|~np.isfinite(arrays[1])|(arrays[0]<0)|(arrays[1]<0))):raise ValueError('InvalidXY')
 q=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
 return arrays,q

def bounds(region):
 box=np.asarray(region['input_box_global_L2_zyx'],float)[:,::-1]
 return {'input1536':(box[0],box[1]),'halo32':(box[0]-32,box[1]+32),'B1408':tuple(np.asarray(region['primary_B1408_global_L2_xyz'],float)),'A384':tuple(np.asarray(region['secondary_A384_global_L2_xyz'],float))}

def broad_rows(arrays,q,lo,hi):
 h,w=arrays[0].shape
 for r0 in range(0,h-1,32):
  r1=min(h-1,r0+32);m=q[r0:r1].copy()
  for axis,a in enumerate(arrays):
   blocks=[a[r0:r1,:-1],a[r0:r1,1:],a[r0+1:r1+1,:-1],a[r0+1:r1+1,1:]]
   low=np.minimum.reduce(blocks)/4.;high=np.maximum.reduce(blocks)/4.;m&=(high>=lo[axis])&(low<=hi[axis])
  r,c=np.nonzero(m);yield r+r0,c

def geometry(arrays,q,box):
 h,w=arrays[0].shape;ids=[];uv=[]
 for r,c in broad_rows(arrays,q,*box):
  if not len(r):continue
  tl=r*w+c;tr=tl+1;bl=tl+w;br=tl+w+1
  ids.append(np.concatenate([np.stack([bl,tl,tr],1),np.stack([bl,tr,br],1)]));uv.append(np.tile(np.stack([r,c],1),(2,1)))
 ids=np.concatenate(ids) if ids else np.empty((0,3),np.int64);uv=np.concatenate(uv) if uv else np.empty((0,2),np.int64)
 uniq,inv=np.unique(ids,return_inverse=True);v=np.column_stack([a.reshape(-1)[uniq] for a in arrays]).astype(float)/4.
 return v,inv.reshape(-1,3),uv,uniq

def main(a):
 start=time.monotonic();a.out.mkdir(parents=True,exist_ok=True);manifest=json.loads(a.sources.read_text());protocol=json.loads(a.protocol.read_text());frozen=protocol['exact_domains_freeze'];assert frozen['assets_only']==[x['asset'] for x in manifest['assets']]
 pre=[]
 for source in manifest['assets']:
  arrays,q=read_source(source)
  for region_id,region in frozen['regions'].items():
   boxes=bounds(region);faces=2*sum(len(r) for r,c in broad_rows(arrays,q,*boxes['halo32']))
   # Conservative bound covers full source arrays, centroidpayloads, native
   # adjacency/sort work; exact intersections cannot exceed broadphase faces.
   peak=128*1024**2+2*sum(x.nbytes for x in arrays)+faces*1000
   pre.append({'region':region_id,'asset':source['asset'],'full_grid_triangles_upper':2*int(q.size),'valid_full_grid_triangles':2*int(q.sum()),'halo_broadphase_face_upper':faces,'compact_bytes_upper':faces*300+65536,'peak_bytes_estimate':peak})
  del arrays,q;gc.collect()
 cap=sum(x['compact_bytes_upper'] for x in pre);info={'rows':pre,'total_compact_bytes_upper':cap,'compact_cap_bytes':COMPACT_CAP,'total_intermediate_cap_bytes':INTERMEDIATE_CAP,'ram_cap_bytes':RAM_CAP,'method':'Countactualsourcequadbboxoverlapbefore allocatingcompactmesh; no candidateoutputs, noassetexclusions.'};write_json(a.out/'size-preflight.json',info)
 if cap>COMPACT_CAP or any(x['peak_bytes_estimate']>RAM_CAP for x in pre):raise ValueError('Referencecrop preflight exceedsfixedcap; preserveallregions/assetlist andstop')
 print(json.dumps({'preflight':'PASS','compact_upper_bytes':cap,'max_estimated_peak':max(x['peak_bytes_estimate'] for x in pre)}),flush=True)
 tests=selftest();reports=[];actual_bytes=0
 for source in manifest['assets']:
  arrays,q=read_source(source)
  for region_id,region in frozen['regions'].items():
   boxes=bounds(region);out=a.out/region_id/source['asset'];out.mkdir(parents=True,exist_ok=True)
   v,faces,uv,uniq=geometry(arrays,q,boxes['halo32'])
   hc,ha,hl,hcomp,hboundary=domain_geometry(v,faces,*boxes['halo32']);keep=ha>1e-10;faces=faces[keep];uv=uv[keep]
   payload={'vertices_global_l2_xyz':v,'faces':faces,'face_source_grid_rc':uv,'source_vertex_grid_linear_ids':uniq,'source_shape':np.array(source['shape'])}
   report={'region':region_id,'asset':source['asset'],'source_root':source['source_root'],'source_original_files_sha256':source['hashes'],'source_frame':'PHercParis4/20260411134726 fullresXYZ','geometry_frame':'globalL2XYZ = originalXYZ/4; metascale notcoordinate multiplier','coordinate_sources_copied':False,'candidate_or_scores_seen':False,'registration_precision_independently_verified':False,'component_is_winding':False,'cross_asset_area_addition_allowed':False,'domains':{}}
   for name,box in boxes.items():
    if name=='halo32':centers,areas,labels,comp,boundary=hc[keep],ha[keep],hl[keep],hcomp,hboundary[keep]
    else:centers,areas,labels,comp,boundary=domain_geometry(v,faces,*box)
    active=areas>1e-10;primary=active&~boundary
    payload[name+'_clipped_area_vox2']=areas;payload[name+'_component']=labels;payload[name+'_boundary']=boundary;payload[name+'_centroids_global_l2_xyz']=centers
    report['domains'][name]={'box_global_l2_xyz':[x.tolist() for x in box],'positive_area_faces':int(active.sum()),'components':comp,'component_count':len(comp),'clipped_area_mm2':float(areas.sum()*9.6**2/1e6),'boundary_faces':int((boundary&active).sum()),'boundary_excluded_area_mm2':float(areas[primary].sum()*9.6**2/1e6),'query_rule':'areas>1e-10; primarydenominator additionallyexcludesboundary. Storedcentroidforinactivefacesisnot aquery.','zero_area_retained':True}
   path=out/'qualified-geometry.npz';tmp=path.with_suffix('.npz.partial')
   with tmp.open('wb') as f:np.savez(f,**payload)
   os.replace(tmp,path);actual_bytes+=path.stat().st_size
   if actual_bytes>COMPACT_CAP:raise ValueError('Actualcompactbudget exceeded')
   report['geometry']={'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)};write_json(out/'qualification.json',report);reports.append(report)
   print(json.dumps({'region':region_id,'asset':source['asset'],'faces':len(faces),'bytes':path.stat().st_size,'B_area_mm2':report['domains']['B1408']['clipped_area_mm2'],'A_area_mm2':report['domains']['A384']['clipped_area_mm2']}),flush=True)
   del v,faces,uv,uniq,payload,hc,ha,hl,hcomp,hboundary,centers,areas,labels,comp,boundary;gc.collect()
  for name,h in source['hashes'].items():
   if digest(Path(source['source_root'])/name)!=h:raise ValueError('Originalreference mutated')
  del arrays,q;gc.collect()
 result={'schema':'iss-disjoint-reference-qualification/1','state':'SIX_REFERENCE_DOMAINS_QUALIFIED_NO_EFFECT_SCORES','assets':reports,'size_preflight':info,'actual_compact_bytes':actual_bytes,'tests':tests,'sources_manifest_sha256':digest(a.sources),'protocol_sha256':digest(a.protocol),'code_sha256':digest(Path(__file__)),'original_coordinate_files_unchanged':True,'no_download_or_original_TIFF_copy':True,'no_candidate_outputs_or_coverage_scores':True,'seconds':time.monotonic()-start,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};write_json(a.out/'reference-manifest.json',result);print(json.dumps({'complete':True,'bytes':actual_bytes,'seconds':result['seconds'],'peak':result['self_peak_rss_bytes']}),flush=True)
def finalize_existing(a):
 import zipfile
 started=time.monotonic();sources=json.loads(a.sources.read_text());protocol=json.loads(a.protocol.read_text());info=json.loads((a.out/'size-preflight.json').read_text());reports=[];total=0
 for source in sources['assets']:
  for name,h in source['hashes'].items():
   if digest(Path(source['source_root'])/name)!=h:raise ValueError('Original source mismatch infinalaudit')
  for region_id,region in protocol['exact_domains_freeze']['regions'].items():
   q=a.out/region_id/source['asset']/'qualification.json';report=json.loads(q.read_text());geometry_path=q.parent/'qualified-geometry.npz'
   if report['source_original_files_sha256']!=source['hashes'] or digest(geometry_path)!=report['geometry']['sha256']:raise ValueError('Qualifiedartifact hash mismatch')
   if report['candidate_or_scores_seen'] or report['component_is_winding']:raise ValueError('Invalidreference scope')
   for name,box in bounds(region).items():
    if report['domains'][name]['box_global_l2_xyz']!=[x.tolist() for x in box]:raise ValueError('Frozendomain mismatch')
   headers={}
   with zipfile.ZipFile(geometry_path) as z:
    for name in z.namelist():
     with z.open(name) as f:
      version=np.lib.format.read_magic(f)
      shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
      headers[name]={'shape':list(shape),'dtype':str(dtype),'fortran':fortran}
   count=headers['faces.npy']['shape'][0]
   for name in bounds(region):
    assert headers[name+'_clipped_area_vox2.npy']['shape']==[count]
    assert headers[name+'_component.npy']['shape']==[count]
    assert headers[name+'_boundary.npy']['shape']==[count]
    assert headers[name+'_centroids_global_l2_xyz.npy']['shape']==[count,3]
   report['npz_header_audit']=headers;total+=geometry_path.stat().st_size;reports.append(report)
 if len(reports)!=6 or total>COMPACT_CAP:raise ValueError('Finalcardinality/budget mismatch')
 result={'schema':'iss-disjoint-reference-qualification/1','state':'SIX_REFERENCE_DOMAINS_QUALIFIED_NO_EFFECT_SCORES','assets':reports,'size_preflight':info,'actual_compact_bytes':total,'tests':selftest(),'sources_manifest_sha256':digest(a.sources),'protocol_sha256':digest(a.protocol),'code_sha256':digest(Path(__file__)),'original_coordinate_files_unchanged':True,'no_download_or_original_TIFF_copy':True,'no_candidate_outputs_or_coverage_scores':True,'finalization_mode':'Existing6completedartifacts verified; no geometry recomputation','seconds':time.monotonic()-started,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};write_json(a.out/'reference-manifest.json',result);print(json.dumps({'finalized':True,'bytes':total,'seconds':result['seconds']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--finalize-existing',action='store_true');a=p.parse_args();finalize_existing(a) if a.finalize_existing else main(a)
