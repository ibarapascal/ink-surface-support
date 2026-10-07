#!/usr/bin/env python3
"""Render figures/*.png and assets/{overview,social-preview}.png from results/evidence.json.

--check renders into a temporary directory and compares every PNG hash with
figures/generated-figure-manifest.json. No scientific evaluation is run.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
T=(.5,1.,2.,4.,8.);COLORS=('#606B79','#007C83','#BB741D');MAIN=('existing10_native','support10_native','existing5_native');ASSETS=('20231210121321','20230702185753','20230929220926')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wrap(label):return label.replace(' (','\n(')
def main(root):
 e=json.loads((ROOT/'results/evidence.json').read_text());out=root/'figures';out.mkdir(parents=True,exist_ok=True);artifacts=[]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name,folder='figures',tight=True,dpi=180):
  p=root/folder/(name+'.png');p.parent.mkdir(parents=True,exist_ok=True)
  fig.savefig(p,dpi=dpi,**({'bbox_inches':'tight'} if tight else {}));artifacts.append({'file':folder+'/'+p.name,'sha256':sha(p)});plt.close(fig)
 def curve(ax,r):
  den=r['reference_area_mm2'];ys=[]
  for arm,c,m in zip(MAIN,COLORS,('s','o','^')):
   y=[r['arms'][arm]['reference']['covered_area_mm2'][str(t)] for t in T];ys.extend(y)
   if den:ax.plot(T,y,color=c,marker=m,linewidth=1.8,markersize=4)
  if den==0:
   ax.set_ylim(0,1);ax.set_yticks([]);ax.text(.5,.48,'NA: no reference area\nZero denominator',transform=ax.transAxes,ha='center',va='center')
  elif not any(ys):
   ax.set_ylim(0,1);ax.set_yticks([0]);ax.text(.5,.48,'All three configurations: zero intersection\nat all five tolerances\n(no answer either way)',transform=ax.transAxes,ha='center',va='center')
  else:ax.set_ylim(0,max(ys)*1.24)
  ax.set_xscale('log',base=2);ax.set_xticks(T,['0.5','1','2','4','8']);ax.set_xlabel('Distance tolerance (input voxels)');ax.set_ylabel('Covered reference area (mm²)');ax.grid(alpha=.16)
 def legend(fig,arms,y):
  handles=[Line2D([0],[0],color=c,marker='o',label=wrap(e['arm_definitions'][a]['label'])) for a,c in zip(arms,COLORS)];fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,y),ncol=3,frameon=False,fontsize=9)
 common='Working-grid interval: 5 voxels. Input CT grid: 9.6 µm/voxel. Output spacing is not scan resolution.\n32 fixed attempts per region and configuration; failures retained. Boundary-excluded reference union; no sum across assets.\nReference proximity does not establish sheet identity. Panels have independent vertical scales.'
 for domain,dimension in [('B1408','Primary 1408³-voxel evaluation boxes'),('A384','Secondary central 384³-voxel evaluation boxes')]:
  fig,axes=plt.subplots(2,3,figsize=(13.5,8.8))
  for i,region in enumerate(('V1','V2')):
   for k,asset in enumerate(ASSETS):
    r=next(r for r in e['coverage_rows'] if r['study']=='main10' and r['region']==region and r['asset']==asset and r['domain']==domain);ax=axes[i,k];curve(ax,r);ax.set_title(f"Sampled region {region} · reference {asset}\nReference denominator {r['reference_area_mm2']:.4f} mm²",fontsize=9)
  fig.suptitle('Registered-reference coverage after native flattening\n'+dimension+' · previously evaluated regions',fontsize=13,y=.985);legend(fig,MAIN,.915);fig.subplots_adjust(top=.79,bottom=.16,hspace=.5,wspace=.31);fig.text(.025,.025,common,fontsize=9,color='#404850',linespacing=1.4);save(fig,'main10-seen-'+domain)
 fig,axes=plt.subplots(3,2,figsize=(12.5,12.6))
 for i,asset in enumerate(ASSETS):
  for k,domain in enumerate(('B1408','A384')):
   r=next(r for r in e['coverage_rows'] if r['study']=='main10' and r['region']=='V3' and r['asset']==asset and r['domain']==domain);ax=axes[i,k];curve(ax,r);dimension='Primary 1408³-voxel box' if domain=='B1408' else 'Secondary central 384³-voxel box';ax.set_title(f"{dimension} · reference {asset}\nReference denominator {r['reference_area_mm2']:.4f} mm²",fontsize=9)
   if i==0 and k==1:ax.text(.025,.95,'Small absolute support only:\n0.00449 mm² at 4 voxels; 0.04273 mm² at 8 voxels',transform=ax.transAxes,va='top',fontsize=8)
 fig.suptitle('Registered-reference coverage after native flattening\nSampled region V3: not used for method development',fontsize=14,y=.99);legend(fig,MAIN,.944);fig.subplots_adjust(top=.86,bottom=.16,hspace=.48,wspace=.30);fig.text(.025,.028,common+'\nV3 shares the scan and registered reference assets: it is not independent ground truth or cross-scan validation.',fontsize=9,color='#404850',linespacing=1.4);save(fig,'V3-reference-curves')
 for regions,name,title in [(('V1','V2'),'main10-seen-cost','Previously evaluated regions V1/V2'),(('V3',),'V3-consumer-cost','Sampled region V3')]:
  cs=[r for r in e['costs'] if r['study']=='main10' and r['region'] in regions];fig,axes=plt.subplots(1,3,figsize=(13.2,5.7));metrics=[('CLI CPU time (minutes)','cpu_seconds',60,sum),('All consumer files (MiB)','all_files_bytes',2**20,sum),('Observed maximum process RSS (MiB)','max_process_rss_bytes',2**20,max)]
  for col,(ax,(label,key,div,fn)) in enumerate(zip(axes,metrics)):
   vals=[fn(r[key] for r in cs if r['arm']==a)/div for a in MAIN];bars=ax.barh(range(3),vals,color=COLORS);ax.bar_label(bars,fmt='%.2f',padding=4);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.24);ax.set_yticks(range(3),[wrap(e['arm_definitions'][a]['label']) for a in MAIN] if col==0 else ['']*3);ax.set_xlabel(label);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
  count=sum(r['arm']==MAIN[0] for r in cs);fig.suptitle(f'Observed native flattening cost — {title}\n{count} accepted source cases per configuration',fontsize=13,y=.98);fig.subplots_adjust(left=.26,right=.985,top=.79,bottom=.24,wspace=.24);fig.text(.025,.045,'Working-grid interval: 5 voxels. Input CT grid: 9.6 µm/voxel. CPU is user + system time, not wall time.\nFile totals include snapshots; RSS is not a full-process-tree bound. Timing observations are not causal speed benchmarks.\nFailed attempts are kept in results/native-noinput.csv. Each timing is a single run.',fontsize=9,color='#404850',linespacing=1.4);save(fig,name)
 rows=[r for r in e['coverage_rows'] if r['study']=='spacing5-development'];arms=('original5','support5','original2_5');fig,axes=plt.subplots(1,2,figsize=(12.8,6.7))
 for ax,r in zip(axes,rows):
  for a,c in zip(arms,COLORS):ax.plot(T,[r['arms'][a]['reference']['covered_area_mm2'][str(t)] for t in T],color=c,marker='o',linewidth=2)
  size=896 if r['case']=='C0-P05' else 384;ax.set_title(f"Fixed development patch {r['case']} · {size}³-voxel box\nReference {r['asset']} · denominator {r['reference_area_mm2']:.4f} mm²",fontsize=10);ax.set_ylim(bottom=0);ax.set_xscale('log',base=2);ax.set_xticks(T,['0.5','1','2','4','8']);ax.set_xlabel('Distance tolerance (input voxels)');ax.set_ylabel('Covered reference area (mm²)');ax.grid(alpha=.17)
 fig.suptitle('Registered-reference coverage after native flattening\nPreviously fixed development patches, not an unseen-region test',fontsize=14,y=.98);legend(fig,arms,.89);fig.subplots_adjust(top=.72,bottom=.22,wspace=.27);fig.text(.025,.035,'Working-grid interval: 5 voxels. Input CT grid: 9.6 µm/voxel. Denser output adds no scan information.\nAll five tolerances use the original reference boxes and boundary exclusions; do not add across assets or boxes.\nDistortion also increases at the same original source cells; see paired-distortion.png.',fontsize=9,color='#404850',linespacing=1.4);save(fig,'spacing5-development-coverage')
 groups=e['paired_distortion'];fig,axes=plt.subplots(2,3,figsize=(14,8.8));titles=['65 primary-study case pairs','2 fixed mechanism case pairs','2 fixed development case pairs']
 for col,g in enumerate(groups):
  step=10 if g['study']=='main10-common65' else 5;old=g['pooled']['original_common']['area_weighted_mean'];new=g['pooled']['support_common']['area_weighted_mean'];area=g['pooled']['original_common']['source_area_mm2'];ax=axes[0,col];bars=ax.bar([0,1],[old,new],color=COLORS[:2]);ax.bar_label(bars,labels=[f'{old:.7f}',f'{new:.7f}'],padding=4,fontsize=9);ax.set_xticks([0,1],['Uniform final\nerosion','Fine-support\nfinalization'],fontsize=8);ax.set_ylim(0,new*1.34);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3));ax.set_ylabel('Common-source weighted mean SDir');ax.set_title(titles[col]+f'\nOutput spacing: {step} voxels\nFixed common source area {area:.4f} mm²',fontsize=10);ax.text(.5,.92,f'+{(new/old-1)*100:.2f}% mean loss',ha='center',transform=ax.transAxes,color='#A23038',fontsize=11)
  ax=axes[1,col];ds=[c['paired_weighted_mean_delta'] for c in g['cases']];ax.scatter(range(1,len(ds)+1),ds,c='#A23038',s=19);ax.axhline(0,color='#606B79');ax.set_ylim(-.04*max(ds),max(ds)*1.15);ax.set_title(f'{sum(x>0 for x in ds)}/{len(ds)} case means increase',fontsize=10);ax.set_xlabel('Case order in evidence / CSV');ax.set_ylabel('Mean Δ: fine support − uniform erosion');ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3));ax.grid(alpha=.15)
  if len(ds)==2:ax.set_xticks([1,2],[c['case'] for c in g['cases']])
 fig.suptitle('Common-source distortion increases after replacing uniform final erosion',fontsize=14,y=.99);fig.subplots_adjust(top=.86,bottom=.19,hspace=.56,wspace=.41);fig.text(.025,.028,'Exactly matched source XYZ and original-cell area weights; no eligible common-source area was dropped.\nNative float32 SDir and epsilon are unchanged. Added regions are separate. Working-grid interval: 5 voxels; input CT: 9.6 µm/voxel.\nThe loss percentage is not a calibrated material-strain or legibility percentage. These cases are not independent acquisitions.\nPanels use different vertical scales. No confidence intervals or p-values are implied.',fontsize=9,color='#404850',linespacing=1.4);save(fig,'paired-distortion')
 for name,size in (('overview',(16,9)),('social-preview',(12.8,6.4))):
  fig=plt.figure(figsize=size,dpi=100);headline(fig,e,compact=name!='overview');save(fig,name,'assets',tight=False,dpi=100)
 (out/'generated-figure-manifest.json').write_text(json.dumps({'evidence_sha256':sha(ROOT/'results/evidence.json'),'renderer_sha256':sha(Path(__file__)),'matplotlib_version':matplotlib.__version__,'source':'Rendered from results/evidence.json by reproduction/render_figures.py.','artifacts':artifacts},indent=2)+'\n')

HEAD_ARMS=(('existing10_native','Spacing 10, no patch','#85847f'),('support10_native','Spacing 10 + this patch','#2a78d6'),('existing5_native','Spacing 5, no patch','#85847f'))
HEAD_REFS=(('20231210121321','Reference\n20231210121321'),('20230702185753','Reference\n20230702185753'))
def headline(fig,e,compact):
 """One-glance summary: V3 held-out region, 4-voxel tolerance, CPU summed over its 18 patches."""
 k=0.85 if compact else 1.0;ink,muted='#1f1f1d','#5c5b57'
 cpu=[sum(r['cpu_seconds'] for r in e['costs'] if r['study']=='main10' and r['region']=='V3' and r['arm']==a) for a,_,_ in HEAD_ARMS]
 n=sum(r['study']=='main10' and r['region']=='V3' and r['arm']==HEAD_ARMS[0][0] for r in e['costs'])
 cov={}
 for asset,_ in HEAD_REFS:
  row=next(r for r in e['coverage_rows'] if r['study']=='main10' and r['region']=='V3' and r['asset']==asset and r['domain']=='B1408')
  cov[asset]=[row['arms'][a]['reference']['covered_area_mm2']['4.0'] for a,_,_ in HEAD_ARMS]
 pct={a:[100*v/c[2] for v in c] for a,c in cov.items()}
 lo0,hi0=min(p[0] for p in pct.values()),max(p[0] for p in pct.values());lo1,hi1=min(p[1] for p in pct.values()),max(p[1] for p in pct.values())
 fig.patch.set_facecolor('white')
 fig.text(.04,.93,"Half the flattening CPU, ~98% of the denser output's coverage",fontsize=22.5 if compact else 30,weight='bold',color=ink,va='top')
 fig.text(.04,.835,f"Output spacing 10 without the patch keeps {lo0:.0f}–{hi0:.0f}% of the spacing-5 reference coverage;\nwith fine-support finalization it keeps {lo1:.1f}–{hi1:.1f}%.",fontsize=17*k,color=ink,va='top',linespacing=1.35)
 left,right=fig.add_axes([.24,.17,.24,.52]),fig.add_axes([.62,.17,.34,.52])
 ys=range(3)[::-1]
 for y,v,(a,label,c) in zip(ys,cpu,HEAD_ARMS):
  left.barh(y,v,color=c,height=.62);left.text(v+8,y,f'{v:.0f} s',va='center',fontsize=15*k,color=ink,weight='bold' if a=='support10_native' else 'normal')
  fig.text(.225,.17+.52*(y+.5)/3,label,ha='right',va='center',fontsize=14*k,color=ink,weight='bold' if a=='support10_native' else 'normal')
 left.set_ylim(-.5,2.5);left.set_xlim(0,max(cpu)*1.28);left.set_yticks([]);left.set_xticks([])
 left.set_title(f'Flattening CPU, {n} patches',fontsize=16*k,color=ink,loc='left',pad=12)
 h=.26
 for g,(asset,name) in enumerate(HEAD_REFS):
  for j,(a,label,c) in enumerate(HEAD_ARMS):
   y=g+(j-1)*h;v=pct[asset][j];right.barh(y,v,color=c,height=h*.86);right.text(2,y,label,va='center',fontsize=14*k,color='white')
   right.text(v+1.2,y,f'{v:.1f}%',va='center',fontsize=15*k,color=ink,weight='bold' if a=='support10_native' else 'normal')
 right.set_yticks([0,1],[name for _,name in HEAD_REFS],fontsize=14.5*k,color=muted);right.invert_yaxis();right.set_xlim(0,116);right.set_xticks([])
 right.set_title('Reference coverage, spacing 5 = 100%',fontsize=16*k,color=ink,loc='left',pad=12)
 for ax in (left,right):
  for side in ('top','right','bottom'):ax.spines[side].set_visible(False)
  ax.spines['left'].set_color('#c9c8c2');ax.tick_params(length=0)
 fig.text(.04,.085,'Trade-off: mean SDir distortion +11.7% vs unpatched spacing 10 (65 patches); not measured vs spacing 5. Single-run timings.',fontsize=14*k,color=muted)
 fig.text(.04,.045,f'PHercParis4, 9.6 µm voxels. Held-out region V3, {n} patches, 4-voxel tolerance. A third reference had no intersection.',fontsize=14*k,color=muted)

def check():
 manifest=json.loads((ROOT/'figures/generated-figure-manifest.json').read_text())
 with tempfile.TemporaryDirectory() as tmp:
  main(Path(tmp));fresh=json.loads((Path(tmp)/'figures/generated-figure-manifest.json').read_text())
 bad=[a['file'] for a in manifest['artifacts'] if sha(ROOT/a['file'])!=a['sha256']]
 if fresh!=manifest:bad.append('rendered output differs from manifest')
 if bad:sys.exit('FAIL: '+', '.join(bad))
 print(f"PASS: {len(manifest['artifacts'])} figures match the manifest")
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT,help='Root directory; figures/ and assets/ are written below it.');p.add_argument('--check',action='store_true');a=p.parse_args()
 check() if a.check else main(a.out)
