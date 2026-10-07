"""Qualify two preselected official meshes geometrically; no candidate scores."""
from __future__ import annotations
import argparse, json, os, resource, time
from pathlib import Path
import numpy as np
import tifffile
from fetch_verified import fetch, digest
ORIGIN=np.array([3584.,5248.,16000.])
OLD_ORIGIN=np.array([3840.,5504.,16256.])
BOXES={'input1024':(ORIGIN,ORIGIN+1024),'old512':(OLD_ORIGIN,OLD_ORIGIN+512),'old_core384':(OLD_ORIGIN+64,OLD_ORIGIN+448),'halo1024':(ORIGIN-32,ORIGIN+1056)}

def write_json(p,d):
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(t,p)

def polygon_area(p):
    if len(p)<3:return 0.
    p=np.asarray(p,dtype=np.float64)
    return float(np.linalg.norm(np.cross(p[1:-1]-p[0],p[2:]-p[0]),axis=1).sum()/2.)

def clipped_area(triangle,lo,hi):
    if np.any(triangle.max(0)<lo) or np.any(triangle.min(0)>hi):return 0.
    if np.all(triangle>=lo) and np.all(triangle<=hi):return polygon_area(triangle)
    poly=list(triangle)
    for axis in range(3):
        for bound,keep_lower in ((lo[axis],True),(hi[axis],False)):
            if not poly:return 0.
            output=[];previous=poly[-1];pv=previous[axis]-bound;pin=pv>=0 if keep_lower else pv<=0
            for current in poly:
                cv=current[axis]-bound;cin=cv>=0 if keep_lower else cv<=0
                if pin!=cin:
                    t=(bound-previous[axis])/(current[axis]-previous[axis]);output.append(previous+t*(current-previous))
                if cin:output.append(current)
                previous=current;pin=cin
            poly=output
    return polygon_area(poly)

def edge_positive_clip(a,b,lo,hi):
    delta=b-a;t0=0.;t1=1.
    for k in range(3):
        if abs(delta[k])<1e-12:
            if a[k]<lo[k] or a[k]>hi[k]:return False
        else:
            ta=(lo[k]-a[k])/delta[k];tb=(hi[k]-a[k])/delta[k]
            t0=max(t0,min(ta,tb));t1=min(t1,max(ta,tb))
            if t1<=t0:return False
    return t1-t0>1e-12 and np.linalg.norm(delta)>1e-12

def components(vertices,faces,areas,lo,hi):
    active=np.flatnonzero(areas>1e-10)
    labels=np.zeros(len(faces),dtype=np.int32)
    if not len(active):return labels,[],np.zeros(len(faces),bool)
    fs=faces[active];edge=np.concatenate((fs[:,[0,1]],fs[:,[1,2]],fs[:,[2,0]]));edge.sort(axis=1)
    ids=np.tile(np.arange(len(active),dtype=np.int64),3);order=np.lexsort((edge[:,1],edge[:,0]));edge=edge[order];ids=ids[order]
    parent=np.arange(len(active));boundary=np.zeros(len(active),bool)
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return int(x)
    i=0
    while i<len(edge):
        j=i+1
        while j<len(edge) and np.array_equal(edge[j],edge[i]):j+=1
        a,b=vertices[edge[i]]
        if edge_positive_clip(a,b,lo,hi):
            if j-i==1:boundary[ids[i]]=True
            else:
                first=root(ids[i])
                for other in ids[i+1:j]:
                    r=root(other)
                    if r!=first:parent[r]=first
        i=j
    roots=np.array([root(i) for i in range(len(active))]);unique,inv=np.unique(roots,return_inverse=True);labels[active]=inv+1
    allp=vertices[fs];boundary|=np.any((allp<lo)|(allp>hi),axis=(1,2))
    bfaces=np.zeros(len(faces),bool);bfaces[active]=boundary
    report=[{'component':i+1,'intersecting_triangles':int(np.count_nonzero(inv==i)),'clipped_area_mm2':float(areas[active[inv==i]].sum()*9.6**2/1e6),'boundary_faces':int(np.count_nonzero(boundary[inv==i]))} for i in range(len(unique))]
    return labels,report,bfaces

def selftest():
    tri=np.array([[0.,0,0],[2,0,0],[0,2,0]])
    assert abs(clipped_area(tri,np.array([0.,0,-1]),np.array([1.,1,1]))-1)<1e-10
    assert clipped_area(tri,np.array([3.,3,-1]),np.array([4.,4,1]))==0
    assert abs(clipped_area(tri,np.array([-1.,-1,-1]),np.array([3.,3,1]))-2)<1e-10
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
    f=np.array([[0,1,2],[1,3,2]])
    lab,report,b=components(v,f,np.array([.5,.5]),np.array([-1.,-1,-1]),np.array([2.,2,1]))
    assert list(lab)==[1,1] and len(report)==1
    return {'triangle_box_area_known_answers':True,'shared_parameter_edge_components':True,'touch_only_edges_do_not_merge':not edge_positive_clip(np.array([-1.,0,0]),np.array([0.,0,0]),np.zeros(3),np.ones(3))}

def qualify(root,entry):
    root.mkdir(parents=True,exist_ok=True);receipts_path=root/'downloads.json';receipts=json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    base='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
    for item in entry['files']:
        name=item['key'].rsplit('/',1)[-1]
        if name not in ('meta.json','x.tif','y.tif','z.tif'):continue
        path=root/name;expected=receipts.get(name,{}).get('sha256')
        rec=fetch(base+item['key'],path,expected_size=item['bytes'],expected_sha=expected,max_bytes=32000000,rate=2800000)
        receipts[name]=rec;write_json(receipts_path,receipts)
    meta=json.loads((root/'meta.json').read_text())
    if meta!=entry['meta']:raise ValueError('Metadata changed after frozen directory selection')
    print(json.dumps({'event':'downloaded','asset':root.name,'bytes':sum(x['bytes'] for x in receipts.values())}),flush=True)
    arrays=[tifffile.imread(root/f'{c}.tif',maxworkers=1) for c in 'xyz'];x,y,z=arrays
    if x.ndim!=2 or x.shape!=y.shape or x.shape!=z.shape or any(a.dtype.kind!='f' for a in arrays):raise ValueError('Coordinate grid type/shape')
    valid=(z>0)&np.isfinite(z)
    if np.any(valid & (~np.isfinite(x)|~np.isfinite(y)|(x<0)|(y<0))):raise ValueError('InvalidXY at official-valid points')
    bbox=[[float(a[valid].min()) for a in arrays],[float(a[valid].max()) for a in arrays]]
    if not np.allclose(bbox,meta['bbox'],atol=.01,rtol=0):raise ValueError('BBox differs from coordinate payload')
    qvalid=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    h,w=x.shape;lo,hi=BOXES['halo1024'];global_ids=[];uvs=[]
    # Broadphase by full actual quad XYZ bounds, never metadata bbox as coverage.
    for r0 in range(0,h-1,32):
        r1=min(h-1,r0+32);mask=qvalid[r0:r1].copy()
        for a in arrays:
            chunks=[a[r0:r1,:-1],a[r0:r1,1:],a[r0+1:r1+1,:-1],a[r0+1:r1+1,1:]]
            amin=np.minimum.reduce(chunks)/4.;amax=np.maximum.reduce(chunks)/4.
            axis=0 if a is x else (1 if a is y else 2)
            mask&=(amax>=lo[axis])&(amin<=hi[axis])
        rr,cc=np.nonzero(mask);rr+=r0;tl=rr*w+cc;tr=tl+1;bl=tl+w;br=tl+w+1
        if len(rr):
            global_ids.append(np.concatenate([np.stack([bl,tl,tr],1),np.stack([bl,tr,br],1)]));uvs.append(np.tile(np.stack([rr,cc],1),(2,1)))
    global_ids=np.concatenate(global_ids) if global_ids else np.empty((0,3),dtype=np.int64)
    uv=np.concatenate(uvs) if uvs else np.empty((0,2),dtype=np.int64)
    unique,inverse=np.unique(global_ids,return_inverse=True)
    vertices=np.column_stack([a.reshape(-1)[unique] for a in arrays]).astype(np.float64)/4.;faces=inverse.reshape(-1,3)
    areas={name:np.zeros(len(faces),dtype=np.float64) for name in BOXES}
    for i,face in enumerate(faces):
        triangle=vertices[face]
        for name,(blo,bhi) in BOXES.items():areas[name][i]=clipped_area(triangle,blo,bhi)
    keep=areas['halo1024']>1e-10;faces=faces[keep];uv=uv[keep];areas={k:a[keep] for k,a in areas.items()}
    result={'asset':root.name,'source':entry['meta_url'],'lineage':entry.get('lineage'),'frame':'global L2 XYZ, scan20260411134726','fullres_to_L2_divisor':4,'meta_scale_not_coordinate_multiplier':True,'shape':list(x.shape),'dtype':str(x.dtype),'valid_points':int(valid.sum()),'valid_grid_quads_total':int(qvalid.sum()),'measured_fullres_bbox_xyz':bbox,'source_metadata_bbox_matches':True,'halo_intersecting_triangles':len(faces),'source_hashes':receipts,'source_lineage_provisional':True,'registration_accuracy_independently_verified':False,'sheet_identity_truth':'UNKNOWN','domains':{}}
    payload={'vertices_global_l2_xyz':vertices,'faces':faces,'face_source_grid_rc':uv,'source_vertex_grid_linear_ids':unique,'source_shape':np.array(x.shape)}
    for name,(blo,bhi) in BOXES.items():
        labels,comps,boundary=components(vertices,faces,areas[name],blo,bhi)
        payload[name+'_clipped_area_vox2']=areas[name];payload[name+'_component']=labels;payload[name+'_boundary']=boundary
        result['domains'][name]={'bbox_global_l2_xyz':[blo.tolist(),bhi.tolist()],'triangles_with_positive_clipped_area':int(np.count_nonzero(areas[name]>1e-10)),'clipped_area_mm2':float(areas[name].sum()*9.6**2/1e6),'components':comps,'component_count':len(comps),'boundary_faces':int(boundary.sum()),'connectivity':'Original parameter-mesh edges whose clipped segment has positive length; no Euclidean reconnection. Point-only contacts are not joined.','area_method':'Linear source triangles clipped to AABB via six-plane polygon clipping; polygon fan area, notbboxvolume or centroid-only estimate.'}
    out=root/'qualified-geometry.npz';tmp=out.with_suffix('.npz.partial')
    with tmp.open('wb') as f:np.savez(f,**payload)
    os.replace(tmp,out);result['geometry']={'path':str(out),'bytes':out.stat().st_size,'sha256':digest(out),'triangles_not_flattened_or_smoothed':True}
    result['limitations']=['Asset-local componentIDs are topological local labels,not windings.','No crossasset merging/deduplication or identity consensus was performed.','Reference geometry qualified independently of currentproductionoutput; noC0/C1scorecomputed.','Preserve originalfixed512reference results; this is a newforward expandedreference cohort.']
    write_json(root/'qualification.json',result)
    print(json.dumps({'event':'qualified','asset':root.name,'domains':{k:{'faces':v['triangles_with_positive_clipped_area'],'components':v['component_count'],'area_mm2':v['clipped_area_mm2']} for k,v in result['domains'].items()}}),flush=True)
    return result

def main(root,manifest):
    started=time.monotonic();root.mkdir(parents=True,exist_ok=True);tests=selftest();entries=json.loads(manifest.read_text());results=[]
    for entry in entries:
        sid=entry['segment_prefix'].strip('/').split('/')[-1];results.append(qualify(root/sid,entry))
    result={'schema':'iss-expanded-reference-qualification/1','state':'QUALIFIED_PROVISIONAL_REFERENCE_ASSETS','selection':'Frozen first2 newbboxintersection historicalsegments; noproductionerrorselection','assets':results,'tests':tests,'new_candidate_scores_computed':False,'previous_fixed_scores_modified':False,'elapsed_seconds':time.monotonic()-started,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'code_sha256':digest(Path(__file__))};write_json(root/'qualification.json',result);print(json.dumps({'event':'complete','seconds':result['elapsed_seconds'],'peak':result['self_peak_rss_bytes']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();main(a.root,a.manifest)
