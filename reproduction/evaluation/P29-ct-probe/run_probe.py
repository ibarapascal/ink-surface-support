"""One fixed P29 CT intersection/visual diagnostic; no inference/optimization."""
import os,json,hashlib,time,resource
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import label,map_coordinates
from evaluate_patch import all_quad,grid_triangles,face_stats,TriangleSoup,surface_distance,sha
from evaluate_expanded_references import domain_geometry
from score_frozen_validation import measure,mm2
W=Path(__file__).parent
start=time.monotonic();plan=json.loads((W/'protocol.json').read_text());cap=plan['resources']['new_root_cap_bytes']
ctpath=Path(plan['CT']['path']);assert ctpath.stat().st_size==plan['CT']['bytes'] and sha(ctpath)==plan['CT']['sha256']
ct=np.load(ctpath,mmap_mode='r');assert ct.shape==(512,512,512) and ct.dtype==np.uint8
ctorigin=np.array(plan['CT']['origin_global_L2_zyx'][::-1],float);origin=np.array(plan['source_origin_L2_zyx'][::-1],float)
lo=ctorigin+8;hi=ctorigin+503
meshes={};grids={};files={}
def load(path):
 a=np.stack([tifffile.imread(path/(n+'.tif'),maxworkers=1) for n in 'xyz'],2);assert a.shape[2]==3 and a.size<750000
 valid=np.isfinite(a).all(2)&~(a==-1).all(2);xyz=a.astype(np.float64);xyz[valid]+=origin
 q=all_quad(valid);v,f,uv,_=grid_triangles(xyz,q);_,area=face_stats(v,f);keep=area>1e-10
 return xyz,valid,{'v':v,'f':f[keep],'uv':uv[keep],'qmask':q,'soup':TriangleSoup(v,f[keep]) if keep.any() else None}
for arm,item in plan['case_records'].items():
 r=item['record'];assert r['status']=='SUCCESS' and r['origin_l2_zyx']==plan['source_origin_L2_zyx']
 for n,h in r['input']['files_sha256'].items():assert sha(Path(r['input']['path'])/n)==h
 out=Path(r['output_path']);cr=Path(r['checkpoint']).parent
 for x in r['files']:
  if x['relative'].startswith('tifxyz/flatten.tifxyz/') and x['relative'].split('/')[-1] in ['x.tif','y.tif','z.tif','meta.json']:assert sha(cr/x['relative'])==x['sha256']
 for stage,path in [('source',Path(r['input']['path'])),('export',out)]:
  xyz,valid,m=load(path);key=arm+'_'+stage;grids[key]=(xyz,valid);meshes[key]=m;files[key]={'path':str(path),'sha256':{n:sha(path/n) for n in ['x.tif','y.tif','z.tif','meta.json']}}
old,ov=grids['existing10_native_source'];new,nv=grids['support10_native_source'];assert old.shape==new.shape
assert not np.any(ov&~nv) and np.array_equal(old[ov],new[ov]),'Do not assume source superset/coordinate parity'
added=all_quad(nv)&~all_quad(ov);v,f,uv,_=grid_triangles(new,added);_,area=face_stats(v,f)
centers,clipped,_,_,boundary=domain_geometry(v,f,lo,hi);use=clipped>1e-10
quad_area=np.zeros(added.shape,float)
for (r,c),a in zip(uv,clipped):quad_area[r,c]+=a
labels,ncomponents=label(added&(quad_area>1e-10),structure=np.array([[0,1,0],[1,1,1],[0,1,0]],bool))
components=[]
for i in range(1,ncomponents+1):
 cells=np.argwhere(labels==i);components.append({'label':i,'area_vox2':float(quad_area[labels==i].sum()),'first_rc':list(min(tuple(x) for x in cells)),'cells':cells})
components.sort(key=lambda x:(-x['area_vox2'],x['first_rc']))
ledger=[]
for (r,c) in np.argwhere(added):ledger.append({'quad_rc':[int(r),int(c)],'source_area_mm2':mm2(area[(uv[:,0]==r)&(uv[:,1]==c)]),'CT_guard_clipped_area_mm2':mm2(np.array([quad_area[r,c]])),'positive_CT_component':int(labels[r,c])})
stages={}
for k,m in meshes.items():
 cc,aa,_,_,bb=domain_geometry(m['v'],m['f'],lo,hi)
 stages[k]={'fullquad_count':int(m['qmask'].sum()),'triangle_count':len(m['f']),'physical_gross_mm2':mm2(face_stats(m['v'],m['f'])[1]),'CT_guard_clipped_mm2':mm2(aa),'CT_guard_boundary_face_mm2':mm2(aa[bb])}
retention={}
for arm in ['existing10_native','support10_native']:
 soup=meshes[arm+'_export']['soup'];dist=surface_distance(centers[use],soup,chunk=128).dist if soup is not None else np.full(int(use.sum()),np.inf)
 retention[arm]=measure(dist,clipped[use])
# Exact plane/linear-triangle intersections are overlays, not a CT acceptance score.
def lines(mesh,center,tangent,normal,other):
 out=[]
 for tri in mesh['v'][mesh['f']]:
  dd=(tri-center)@other;pts=[]
  for i,j in [(0,1),(1,2),(2,0)]:
   if dd[i]==0:pts.append(tri[i])
   if dd[i]*dd[j]<0:pts.append(tri[i]+(tri[j]-tri[i])*(-dd[i]/(dd[j]-dd[i])))
  if len(pts)>=2:
   pts=np.unique(np.asarray(pts),axis=0)
   if len(pts)>=2:out.append(np.column_stack([(pts[:2]-center)@tangent,(pts[:2]-center)@normal]))
 return out
plots=[];selections=[]
for comp in components[:3]:
 cells=comp['cells'];r,c=min((tuple(x) for x in cells),key=lambda rc:(-quad_area[rc],rc));choose=(uv[:,0]==r)&(uv[:,1]==c)&use
 chosen_face=int(np.flatnonzero(choose)[np.argmax(clipped[choose])]);center=centers[chosen_face]
 e1=new[r,c+1]-new[r,c];e1/=np.linalg.norm(e1);row=new[r+1,c]-new[r,c];e2=row-e1*np.dot(row,e1);e2/=np.linalg.norm(e2);normal=np.cross(e1,e2);normal/=np.linalg.norm(normal)
 selection={'component':comp['label'],'quad_rc':[int(r),int(c)],'selected_added_face_index':chosen_face,'clipped_component_mm2':mm2(np.array([comp['area_vox2']])),'global_L2_xyz':center.tolist(),'CT_local_zyx':(center-ctorigin)[::-1].tolist(),'basis_e1_e2_normal':[e1.tolist(),e2.tolist(),normal.tolist()],'planes':[],'point_distances_not_coverage':{}}
 for k,m in meshes.items():selection['point_distances_not_coverage'][k]=float(surface_distance(center[None],m['soup'],chunk=1).dist[0]) if m['soup'] else None
 keys=['existing10_native_source','support10_native_source','existing10_native_export','support10_native_export'];titles=['Original 10 source','Support 10 source','Original 10 actual export','Support 10 actual export']
 for rr,(tangent,other,axisname) in enumerate([(e1,e2,'column tangent'),(e2,e1,'row tangent')]):
  xx,zz=np.meshgrid(np.arange(-24,25,dtype=float),np.arange(-8,9,dtype=float));points=center+xx[...,None]*tangent+zz[...,None]*normal;local=(points-ctorigin)[...,::-1];inside=((local>=0)&(local<=511)).all(2)
  img=map_coordinates(ct,local.reshape(-1,3).T,order=1,output=np.float32,mode='constant',cval=0,prefilter=False).reshape(xx.shape).astype(float);img[~inside]=np.nan
  plane={'axis':axisname,'samples':int(img.size),'within_CT':int(inside.sum()),'interpolation':'linear, native9.6um CT; displaystep1L2voxel','display_limits':[0,255],'CT_intensity_float32':np.where(inside,img,0).tolist(),'inside_CT_mask':inside.tolist(),'overlays':{}}
  for col,(k,title) in enumerate(zip(keys,titles)):
   plane['overlays'][k]={'title':title,'segments_tangent_normal_vox':[seg.tolist() for seg in lines(meshes[k],center,tangent,normal,other)]}
  selection['planes'].append(plane)
 selections.append(selection)
result={'status':'CT_INTERSECTION_AND_SECTION_SAMPLES_COMPLETE' if use.any() else 'NO_ADDED_FACE_CT_INTERSECTION_STOP','role':'SEEN_P29_DIAGNOSTIC_NOT_METHOD_VALIDATION','CT':plan['CT'],'CT_guard_box_global_L2_xyz':[lo.tolist(),hi.tolist()],'source_shared_XYZ_exact_and_old_valid_retained':True,'added_source_quads':int(added.sum()),'added_source_area_mm2':mm2(area),'added_source_CT_guard_area_mm2':mm2(clipped),'added_triangles_CT_intersect':int(use.sum()),'CT_positive_added_quad_components':ncomponents,'all_added_quad_coverage_ledger':ledger,'stages':stages,'added_CT_source_to_actual_export_geometry':retention,'selected_sections':selections,'plots':plots,'input_files':files,'protocol_sha256':sha(W/'protocol.json'),'driver_sha256':sha(__file__),'no_volume_copy_or_download':True,'no_optimizer_inference_or_new_GT':True,'limits':['Only one alreadyseen P29 CT-overlap region; not195cohort CTvalidation.','Rawintensity/line proximity notfibertruth orsamewinding/identity proof.','Selection uses CTavailablegeometry only, notCTcontrast/reference or successfulconsumer appearance.','Masksampling uses native9.6um, not higherresolution; visualonly, no newCT threshold.','FixedhistoricalP29export precisionprovenance limited; geometry actualfiles verified but notnewstrict195consumer.'],'seconds':time.monotonic()-start,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
text=json.dumps(result,indent=2,allow_nan=False)+'\n';assert len(text.encode())<=4*1024**2;(W/'result.json').write_text(text);size=sum(p.stat().st_size for p in W.rglob('*') if p.is_file());assert size<=cap
print(json.dumps({'status':result['status'],'added_CT_mm2':result['added_source_CT_guard_area_mm2'],'selected_sections':len(selections),'root_bytes':size,'seconds':result['seconds'],'peak':result['self_peak_rss_bytes']}))
