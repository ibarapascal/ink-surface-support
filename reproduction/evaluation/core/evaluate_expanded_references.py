"""Exploratory A/B domains for new reference assets, keeping all old scores.

Each asset remains separate. Boundary-excluded denominators are explicit;
clipped area is not equated with independently decidable identity evidence.
"""
from __future__ import annotations
import argparse,json,os,resource,time
from pathlib import Path
import numpy as np
from expanded_references import clipped_area,components,polygon_area
from evaluate_patch import (TriangleSoup,surface_distance,sha,write_json,
                            write_npz,UM_PER_VOX,TOLERANCES,ORIGIN_XYZ)
from evaluate_patch_batch import read_candidate,restrict_summary
from evaluate_patch import all_quad,grid_triangles,face_stats

BOXES={
 'A_old_core384':(np.array([3904.,5568.,16320.]),np.array([4288.,5952.,16704.])),
 'B_input_core896':(np.array([3648.,5312.,16064.]),np.array([4544.,6208.,16960.]))}
ASSETS={
 '20230702185753':'cff7978966c2513791e90cf1ec8514cc3803665b4da3e7bf788c129952c6a1d8',
 '20230929220926':'3a1c07c4fb6d0db29527e819cdf0d8f9c1355e8c17611a5f30072dc84da49aed'}


def clipped_centroid(triangle,lo,hi):
    """Area-centroid of the same six-plane clipped convex triangle polygon.

    Areas are verified against the existing qualified clipper, which remains
    authoritative. This only supplies an in-domain quadrature point.
    """
    poly=list(np.asarray(triangle,float))
    for axis in range(3):
        for bound,lower in ((lo[axis],True),(hi[axis],False)):
            if not poly:return None,0.
            out=[];prev=poly[-1];pv=prev[axis]-bound;pin=pv>=0 if lower else pv<=0
            for cur in poly:
                cv=cur[axis]-bound;inside=cv>=0 if lower else cv<=0
                if pin!=inside:
                    t=(bound-prev[axis])/(cur[axis]-prev[axis]);out.append(prev+t*(cur-prev))
                if inside:out.append(cur)
                prev=cur;pin=inside
            poly=out
    if len(poly)<3:return None,0.
    poly=np.asarray(poly);a=poly[0];b=poly[1:-1];c=poly[2:]
    weights=np.linalg.norm(np.cross(b-a,c-a),axis=1)/2
    if weights.sum()<=1e-10:return None,0.
    return ((a+b+c)/3*weights[:,None]).sum(0)/weights.sum(),float(weights.sum())


def domain_geometry(vertices,faces,lo,hi):
    centers,fullarea=face_stats(vertices,faces)
    triangles=vertices[faces]
    inside=((triangles>=lo)&(triangles<=hi)).all((1,2))
    overlap=(triangles.max(1)>=lo).all(1)&(triangles.min(1)<=hi).all(1)
    areas=np.zeros(len(faces));areas[inside]=fullarea[inside]
    for i in np.flatnonzero(overlap&~inside):
        expected=clipped_area(triangles[i],lo,hi)
        if expected>1e-10:
            point,area=clipped_centroid(triangles[i],lo,hi)
            if point is None or not np.isclose(area,expected,rtol=1e-9,atol=1e-8):
                raise ValueError('Centroid polygon disagrees with qualified exact clip area')
            areas[i]=expected;centers[i]=point
    labels,comp,boundary=components(vertices,faces,areas,lo,hi)
    return centers,areas,labels,comp,boundary


def area_mm2(area):return float(area.sum()*UM_PER_VOX**2/1e6)


def run(asset,root,manifests,out):
    start=time.monotonic();path=Path(root)/asset
    geometry_path=path/'qualified-geometry.npz'
    if sha(geometry_path)!=ASSETS[asset]:raise ValueError('Qualified reference hash changed')
    qualification=json.loads((path/'qualification.json').read_text())
    d=np.load(geometry_path,allow_pickle=False)
    v=d['vertices_global_l2_xyz'];f=d['faces'];reference=TriangleSoup(v,f)
    halo_boundary=d['halo1024_boundary']
    candidates={}
    for condition,manifest_path in manifests.items():
        manifest=json.loads(Path(manifest_path).read_text());items=[]
        for record in manifest['patches']:
            xyz,valid=read_candidate(record)
            xyz=xyz.astype(np.float64)
            xyz[valid]+=ORIGIN_XYZ # old-local adapter -> globalL2, no extra float32 rounding
            q=all_quad(valid)
            if not q.any():
                items.append((record,None,None,None));continue
            cv,cf,_,_=grid_triangles(xyz,q);_,ca=face_stats(cv,cf)
            cf=cf[ca>1e-10]
            items.append((record,cv,cf,TriangleSoup(cv,cf) if len(cf) else None))
        candidates[condition]=(manifest,items)
    report={'schema':'iss-expanded-reference-exploratory-evaluation/1','asset':asset,
      'status':'COMPLETE_EXPLORATORY_ONLY','source_sha256':sha(__file__),
      'reference_geometry_sha256':ASSETS[asset],'qualification_sha256':sha(path/'qualification.json'),
      'reference_source':qualification['source'],'lineage':qualification.get('lineage'),
      'registration_accuracy_independently_verified':False,'sheet_identity_truth':'UNKNOWN',
      'domains':{},'old_scores_modified':False,'cross_asset_area_may_overlap_do_not_sum':True,
      'scope':'New references and B expansion examined after development outputs existed. Exploratory geometry, not heldout validation or method/seed tuning. A newasset arm retains oldcore coordinates; does not replace oldreference scores.'}
    for name,(lo,hi) in BOXES.items():
        centers,areas,labels,comps,boundary=domain_geometry(v,f,lo,hi)
        active=areas>1e-10;idx=np.flatnonzero(active);pts=centers[active]
        result={'bbox_global_l2_xyz':[lo.tolist(),hi.tolist()],
          'total_clipped_area_mm2':area_mm2(areas),
          'boundary_excluded_geometric_denominator_mm2':area_mm2(areas[~boundary]),
          'boundary_area_mm2':area_mm2(areas[boundary]),'reference_components':comps,
          'boundary_is_not_identity_truth':True,'conditions':{}}
        for condition,(manifest,items) in candidates.items():
            union=np.full(len(idx),np.inf);outputs=[]
            for record,cv,cf,soup in items:
                item={'attempt_id':record['attempt_id'],'path':record['path'],
                      'all_components_included':True}
                if soup is None:
                    item['status']='NO_VALID_SURFACE';outputs.append(item);continue
                if len(pts):
                    dist=surface_distance(pts,soup,chunk=128).dist
                    np.minimum(union,dist,out=union)
                cp,carea,_,_,cboundary=domain_geometry(cv,cf,lo,hi)
                cmask=carea>1e-10
                item.update(clipped_candidate_area_mm2=area_mm2(carea),
                            candidate_domain_boundary_area_mm2=area_mm2(carea[cboundary]))
                if cmask.any():
                    # Small batch keeps exact soup-query allocation bounded.
                    cr=surface_distance(cp[cmask],reference,chunk=8)
                    nearest_boundary=halo_boundary[cr.face_idx]|boundary[cr.face_idx]
                    supported=(cr.dist<=max(TOLERANCES))&~nearest_boundary&(labels[cr.face_idx]>0)
                    item['candidate_to_this_reference']=restrict_summary(cr.dist,carea[cmask])
                    item['area_supported_within8vox_mm2']=area_mm2(carea[cmask][supported])
                    item['unknown_area_mm2']=area_mm2(carea[cmask][~supported])
                    item['nearest_reference_boundary_area_mm2']=area_mm2(carea[cmask][nearest_boundary])
                    # No arbitrary single targetcomponent; all reference
                    # components listed separately with candidate correspondence.
                    near_components=labels[cr.face_idx]
                    item['nearest_reference_component_supported_area_mm2']={str(c):area_mm2(carea[cmask][supported&(near_components==c)]) for c in np.unique(near_components[supported])}
                    item['unknown_is_error']=False
                else:item['status']='OUTSIDE_THIS_DOMAIN_UNKNOWN'
                outputs.append(item)
            bycomponent=[]
            for comp in comps:
                cm=(labels[active]==comp['component']);interior=cm&~boundary[active]
                bycomponent.append({'component':comp['component'],
                    'all_clipped':restrict_summary(union[cm],areas[active][cm]),
                    'boundary_excluded':restrict_summary(union[interior],areas[active][interior])})
            interior=~boundary[active]
            result['conditions'][condition]={
                'status':'SCORED_GEOMETRY_ONLY' if bool((union[interior]<=max(TOLERANCES)).any()) else 'INCONCLUSIVE_NO_INTERIOR_REFERENCE_INTERSECTION',
                'manifest_sha256':sha(manifests[condition]),'attempts':manifest['attempts'],
                'all_clipped_union':restrict_summary(union,areas[active]),
                'boundary_excluded_union':restrict_summary(union[interior],areas[active][interior]),
                'components':bycomponent,'outputs':outputs}
            sample_path=Path(out).with_name(Path(out).stem+f'-{name}-{condition}-samples.npz')
            write_npz(sample_path,reference_face_indices=idx,clipped_centers_global_l2_xyz=pts,
                      clipped_area_vox2=areas[active],component=labels[active],boundary=boundary[active],union_distance_vox=union)
            result['conditions'][condition]['samples_sha256']=sha(sample_path)
        report['domains'][name]=result
        print(json.dumps({'event':'domain_evaluated','asset':asset,'domain':name,'area_mm2':result['total_clipped_area_mm2'],'geometric_denominator_mm2':result['boundary_excluded_geometric_denominator_mm2']}),flush=True)
    report['elapsed_seconds']=time.monotonic()-start
    report['peak_rss_darwin_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out,report);return report


def selftest():
    t=np.array([[0.,0,0],[2.,0,0],[0.,2,0]])
    lo=np.array([0.,0,-1]);hi=np.array([1.,1,1])
    center,area=clipped_centroid(t,lo,hi)
    assert np.allclose(center,[.5,.5,0]) and abs(area-1)<1e-12
    assert abs(area-clipped_area(t,lo,hi))<1e-12
    _,outside=clipped_centroid(t,[3.,3,-1],[4.,4,1]);assert outside==0
    # Fully inside and clipped components both tracked, never Euclidean joined.
    v=np.concatenate([t,t+[4.,0.,0.]])
    f=np.array([[0,1,2],[3,4,5]])
    pts,areas,labels,comps,boundary=domain_geometry(v,f,np.array([0.,0,-1]),np.array([5.,2,1]))
    assert len(comps)==2 and np.all(labels>0)
    assert np.all(pts[areas>0]>=np.array([0.,0,-1])) and np.all(pts[areas>0]<=np.array([5.,2,1]))
    assert np.all(boundary),'Isolated triangles retain topologicalboundary status'
    return {'status':'PASS_EXPANDED_DOMAIN_INTERFACE_ONLY','checks':['Clippedpolygon exactarea/centroid','Outsidearea zero','Existingcomponentclipper unchanged','Quadraturepoints insidebox','Topologicalboundarydenominator not silently ignored'],'scientific_claim':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['selftest','evaluate']);p.add_argument('--asset',choices=list(ASSETS));p.add_argument('--root');p.add_argument('--manifest-c0');p.add_argument('--manifest-c1');p.add_argument('--out',required=True);a=p.parse_args()
    if a.action=='selftest':r=selftest();write_json(a.out,r)
    else:r=run(a.asset,a.root,{'C0':a.manifest_c0,'C1':a.manifest_c1},a.out)
    print(json.dumps({'status':r['status']}),flush=True)


if __name__=='__main__':main()
