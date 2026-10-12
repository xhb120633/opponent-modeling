"""Existing-evidence supplementary figures, no inference or training."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
from matplotlib.lines import Line2D
from paper_style import apply,COLORS as C
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PDF=OUT/'pdf';FIG=OUT/'figures'
V='Hide_and_Seek/Visualization/';E='experiments_point_diagnostic/'
OP=['RB','QL','WSLS','ToM0','ToM1','ToM2']
sources={};values=[];checks=[]
def load(rel):
 p=ROOT/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
 return list(csv.DictReader(p.open(encoding='utf-8-sig'))) if p.suffix=='.csv' else json.loads(p.read_text(encoding='utf-8'))
def record(fig,role,model,op,value,source,status='observed',lo=None,hi=None):
 values.append(dict(figure=fig,role=role,model=model,opponent=op,value=value,status=status,source=source,ci_low=lo,ci_high=hi))
def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 texts=list(fig.texts)
 for ax in fig.axes:
  texts += [ax.title,ax._left_title,ax.xaxis.label,ax.yaxis.label,*ax.texts]
  for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
   texts += [t.label1 for t in axis.get_major_ticks() if min(limits)<=t.get_loc()<=max(limits)]
  if ax.get_legend():texts+=ax.get_legend().get_texts()
 for t in texts:
  if t.get_visible() and t.get_text():
   assert t.get_fontsize()>=7
   b=t.get_window_extent(renderer)
   assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(name,t.get_text(),b)
 fig.savefig(PDF/(name+'.pdf'));fig.savefig(FIG/(name+'.svg'));fig.savefig(FIG/(name+'.png'),dpi=220)
 checks.append(dict(figure=name,width=fig.get_figwidth(),height=fig.get_figheight(),minimum_font=7));plt.close(fig)
def heat(ax,arr,labels,limit,title,missing,show_y=True,groups=None):
 headers={}
 if groups:
  aa=[];ll=[];mm={}
  for i,label in enumerate(labels):
   if i in groups:headers[len(aa)]=groups[i];aa.append(np.full(6,np.nan));ll.append('')
   n=len(aa);aa.append(arr[i]);ll.append(label)
   for j in range(6):
    if (i,j) in missing:mm[n,j]=missing[i,j]
  arr=np.array(aa);labels=ll;missing=mm
 cmap=LinearSegmentedColormap.from_list('signed_payoff',['#80629B','#F8F7F2','#2D8A78']);cmap.set_bad('#eeeeee')
 mesh=ax.pcolormesh(np.arange(7)-.5,np.arange(len(labels)+1)-.5,np.ma.masked_invalid(arr),cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),edgecolors='white',linewidth=.5)
 ax.set_ylim(len(labels)-.5,-.5);ax.set_xticks(range(6),OP);ax.set_yticks(range(len(labels)),labels if show_y else ['']*len(labels));ax.set_xlabel('Opponent');ax.set_title(title,loc='left',fontweight='normal');ax.grid(False);ax.tick_params(length=0)
 for i in range(len(labels)):
  if i in headers:
   ax.axhspan(i-.5,i+.5,color='#d5dce2',zorder=2)
   ax.axhline(i-.5,color='#333333',lw=1.3)
   ax.text(-.43,i,headers[i],fontsize=7,fontweight='normal',ha='left',va='center',zorder=3)
   continue
  for j in range(6):
   v=arr[i,j];s=f'{v:+.1f}' if np.isfinite(v) else missing.get((i,j),'NA')
   ax.text(j,i,s,ha='center',va='center',fontsize=7,color='white' if np.isfinite(v) and abs(v)>limit*.65 else '#333333')
 return mesh
def build():
 apply();captions=[]
 final_path=E+'standard_final6_20260922/audit.json';final_cells=load(final_path);assert final_cells['audit_passed']
 c_hider_path=E+'fig2c_hider_task0_20260922/audit.json';c_hider=load(c_hider_path);assert c_hider['audit_passed']
 llama_hider_path=E+'standard_cells43_45_47_20260922/audit.json';llama_hider=load(llama_hider_path);assert llama_hider['audit_passed']
 qwen_hider_path=E+'standard_cells39_42_20260922/audit.json';qwen_hider=load(qwen_hider_path);assert qwen_hider['audit_passed']
 hider_next_path=E+'standard_cells35_38_20260922/audit.json';hider_next=load(hider_next_path);assert hider_next['audit_passed']
 hider_more_path=E+'standard_cells29_34_20260922/audit.json';hider_more=load(hider_more_path);assert hider_more['audit_passed']
 newest_path=E+'standard_cells24_28_20260922/audit.json';newest=load(newest_path);assert newest['audit_passed']
 latest_path=E+'standard_cells20_23_20260922/audit.json';latest=load(latest_path);assert latest['audit_passed']
 new_path=E+'standard_cells12_19_20260922/audit.json';new_cells=load(new_path);assert new_cells['audit_passed']
 extra_path=E+'gpt54_supplement_full_20260921/audit_merged.json';extra=load(extra_path)
 assert extra['audit_passed'] and extra['completed_matches_audited']==400
 small_path=E+'supplement_cachefix_results_20260922/audit_all_small.json';small=load(small_path)
 assert small['audit_passed']
 standard_path=E+'standard_first6_20260922/audit.json';standard=load(standard_path);assert standard['audit_passed']
 q7path=E+'qwen7b_hider_archive_20260922/audit.json';q7=load(q7path);assert q7['audit_passed'] and q7['n_unique_keys']==3000
 oss_low_path=E+'oss_hider_low_20260923/audit.json';oss_low=load(oss_low_path)
 assert oss_low['audit_passed'] and oss_low['n_matches']==20 and oss_low['n_rows']==1000
 oss_medium_path=E+'oss_hider_medium_20260925/audit.json';oss_medium=load(oss_medium_path)
 assert oss_medium['audit_passed'] and oss_medium['n_matches']==20 and oss_medium['n_rows']==1000 and oss_medium['fallback_random']==4
 oss_high_path=E+'oss_hider_high_20260924/audit.json';oss_high=load(oss_high_path)
 assert oss_high['audit_passed'] and oss_high['n_matches']==20 and oss_high['n_rows']==1000 and oss_high['fallback_random']==10
 gemma_path=E+'gemma_thinking_tom2_20260925/audit.json';gemma=load(gemma_path)
 assert gemma['audit_passed'] and gemma['n_matches']==20 and gemma['n_rows']==1000 and gemma['fallback_random']==30
 oss_rb_path=E+'reasoning_task12_audit_20260924/audit.json';oss_rb=load(oss_rb_path)
 assert oss_rb['audit_passed'] and oss_rb['n_matches']==20 and oss_rb['n_rows']==1000 and oss_rb['fallback_random']==1
 oss_tom1_path=E+'reasoning_task16_audit_20260924/audit.json';oss_tom1=load(oss_tom1_path)
 assert oss_tom1['audit_passed'] and oss_tom1['n_matches']==20 and oss_tom1['n_rows']==1000 and oss_tom1['fallback_random']==3
 structured32_path=E+'standard_cells6_8_20260922/audit.json';structured32=load(structured32_path);assert structured32['audit_passed']
 structured72_path=E+'standard_cells9_10_20260922/audit.json';structured72=load(structured72_path);assert structured72['audit_passed']
 structured72wsls_path=E+'standard_cell11_20260922/audit.json';structured72wsls=load(structured72wsls_path);assert structured72wsls['audit_passed']
 src=V+'figures_dynamic_tom_summary/unified_uncertainty_summary.csv';rows=load(src)
 # Figure S1 shows the baseline and output-format benchmarks. The separate
 # reasoning-effort comparison is restricted to ToM2 in main Figure 2B.
 names=[f'Qwen2.5-{s}B-Instruct' for s in ['0.5','1.5','3','7','14','32','72']]+['Qwen2.5-72B-Instruct','Qwen3.6-35B-A3B (no-thinking)','Llama-3.3-70B-Instruct','gemma-4-31B-it','GPT-5.4']
 labels=[f'Qwen2.5 {s}B' for s in ['0.5','1.5','3','7','14','32','72']]+['Qwen2.5 72B','Qwen3.6 35B-A3B','Llama 3.3 70B','Gemma 4 31B','GPT-5.4']
 def norm(s):return s.replace(' (no-thinking)','') if s.startswith('Qwen2.5') else s
 fig,axes=plt.subplots(1,2,figsize=(5.5,5.5),layout='constrained',gridspec_kw={'wspace':.06})
 for ax,role in zip(axes,['seeker','hider']):
  arr=np.full((len(names),6),np.nan);missing={}
  for i,name in enumerate(names):
   for j,op in enumerate(OP):
    rr=[r for r in rows if r['role']==role and norm(r['model_label'])==name and r['opponent']==op];assert len(rr)<=1
    status='not_in_archive'
    if rr:
     r=rr[0];status=r['inclusion_status']
     if status!='appendix_invalid':arr[i,j]=float(r['mean_final_payoff'])
     else:missing[i,j]='INV'
    elif role=='hider' and name.startswith('gpt-oss') and op=='ToM2':missing[i,j]='P';status='evaluation_incomplete'
    if name=='GPT-5.4' and op=='ToM2':
     gp=E+('gpt54_structured_live_20260921' if role=='seeker' else 'gpt54_hider_structured_20260921')+'/audit.json';g=load(gp);assert g['audit_passed'];arr[i,j]=g['metrics']['action_only']['mean'];status='audited_new_control'
    if name=='GPT-5.4' and op!='ToM2':
     gp=extra_path;z=extra['metrics'].get(role+'_'+op,{}).get('action_only',{})
     if z.get('n_matches')==20:arr[i,j]=z['mean'];status='audited_complete_cell'
     else:missing[i,j]='INC';status='infrastructure_incomplete'
    cell_source=gp if name=='GPT-5.4' else src;lo=hi=None
    if role=='hider' and name=='gemma-4-31B-it (thinking)' and op=='ToM2':
     assert not np.isfinite(arr[i,j])
     arr[i,j]=gemma['mean_final_payoff'];lo,hi=gemma['ci95_match_bootstrap_payoff']
     cell_source=gemma_path;status='audited_new_live_cell_30_fallbacks';missing.pop((i,j),None)
    if role=='hider' and name=='gpt-oss-20b (effort=low)' and op=='RB':
     assert not np.isfinite(arr[i,j])
     arr[i,j]=oss_rb['mean_final_payoff'];lo,hi=oss_rb['ci95_match_bootstrap']
     cell_source=oss_rb_path;status='audited_new_live_cell_one_fallback';missing.pop((i,j),None)
    if role=='hider' and name=='gpt-oss-20b (effort=low)' and op=='ToM1':
     assert not np.isfinite(arr[i,j])
     arr[i,j]=oss_tom1['mean_final_payoff'];lo,hi=oss_tom1['ci95_match_bootstrap']
     cell_source=oss_tom1_path;status='audited_new_live_cell_three_fallbacks';missing.pop((i,j),None)
    if role=='hider' and name=='gpt-oss-20b (effort=low)' and op=='ToM2':
     assert not np.isfinite(arr[i,j])
     arr[i,j]=oss_low['mean_final_payoff'];lo,hi=oss_low['ci95_match_bootstrap']
     cell_source=oss_low_path;status='audited_new_live_cell_three_fallbacks';missing.pop((i,j),None)
    if role=='hider' and name=='gpt-oss-20b (effort=medium)' and op=='ToM2':
     assert not np.isfinite(arr[i,j])
     arr[i,j]=oss_medium['mean_final_payoff'];lo,hi=oss_medium['ci95_match_bootstrap_payoff']
     cell_source=oss_medium_path;status='audited_new_live_cell_four_fallbacks';missing.pop((i,j),None)
    if role=='hider' and name=='gpt-oss-20b (effort=high)' and op=='ToM2':
     assert not np.isfinite(arr[i,j])
     arr[i,j]=oss_high['mean_final_payoff'];lo,hi=oss_high['ci95_match_bootstrap']
     cell_source=oss_high_path;status='audited_new_live_cell_ten_fallbacks';missing.pop((i,j),None)
    fresh=[z for z in small['cells'] if z['mode']=='baseline' and z['role']==role and z['model'].split('/')[-1]==name and z['opponent']==op]
    if fresh:
     assert len(fresh)==1 and not np.isfinite(arr[i,j]), 'Do not overwrite archived result'
     z=fresh[0];arr[i,j]=z['mean_final_payoff'];lo,hi=z['ci95'];cell_source=small_path;status='audited_new_live_cell';missing.pop((i,j),None)
    record('S1',role,name,op,float(arr[i,j]) if np.isfinite(arr[i,j]) else None,cell_source,status,lo,hi)
    fresh_standard=[z for z in standard['cells'] if z['role']==role and z['model'].split('/')[-1]==name and z['opponent']==op]
    if fresh_standard:
     assert len(fresh_standard)==1 and not np.isfinite(arr[i,j]);z=fresh_standard[0];arr[i,j]=z['mean_final_payoff'];missing.pop((i,j),None)
     values[-1].update(value=float(arr[i,j]),source=standard_path,status='audited_new_live_cell',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
    archived7=[z for z in q7['cells'] if role=='hider' and name=='Qwen2.5-7B-Instruct' and z['opponent']==op]
    if archived7:
     assert len(archived7)==1 and not np.isfinite(arr[i,j]);z=archived7[0];arr[i,j]=z['mean_final_payoff'];missing.pop((i,j),None)
     values[-1].update(value=float(arr[i,j]),source=q7path,status='audited_recovered_archive_cell',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  mesh=heat(ax,arr,labels,50,role.title(),missing,ax==axes[0],{0:'Scaling',7:'Flagship comparison'})
 cb=fig.colorbar(mesh,ax=axes,orientation='horizontal',fraction=.06,pad=.035);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.set_label('Mean payoff per 50-round match (higher = better)')
 name='figS1_full_benchmark_20260921';save(fig,name)
 captions.append((name,'S1','Baseline live-play benchmark across six opponent classes, augmented by audited live cells. Both roles share the same scale; cells are descriptive means, typically 20 matches of 50 rounds. Q2.5=Qwen2.5; Q3.6=Qwen3.6-35B-A3B; Gemma4=31B. RB=biased random, QL=Q-learning, WSLS=win-stay/lose-shift. NA means no usable score in the source package, not zero. Qwen2.5 hider cells were collected as a separate role supplement. GPT-5.4 uses action-only none/2048 controls; its five non-ToM2 opponents have audited 20-match cells per role after bounded recovery. Repeated model rows reuse the same source cells. Intervals and original inclusion flags remain in source tables; missing non-ToM hider scaling cells require an additional archive check.'))
 # Structured output: fixed behavior condition, not best prompt per cell.
 n,label,cap=captions[-1]
 cap=cap.replace('missing non-ToM hider scaling cells require an additional archive check.', 'Twelve missing hider cells now use audited new live runs (Q2.5 0.5/1.5/3/14B against RB/QL/WSLS), 20 matches each, TP1/H200. Match-bootstrap intervals are exported; historical runtime differences preclude a matched model-size intervention.')
 captions[-1]=(n,label,cap)
 n,label,cap=captions[-1];cap += ' Six further 32B/72B hider cells are audited TP4/H200 runs. Historical runtime differences preclude a matched model-size intervention.';captions[-1]=(n,label,cap)
 n,label,cap=captions[-1];cap += ' Qwen2.5-7B hider RB/QL/WSLS were recovered from one historical 3000-row CSV and semantically audited; source revision and historical runtime remain unverified.';captions[-1]=(n,label,cap)
 # Every cell in the current S1 rendering has a score; do not describe absent NA cells.
 n,label,cap=captions[-1]
 captions[-1]=(n,label,cap.replace('NA means no usable score in the source package, not zero. ',''))
 sp=V+'figures_structured_tom_all/structured_tom_all_intervention_summary.csv';st=load(sp)
 tp=V+'figures_structured_tom2/structured_tom2_intervention_summary.csv';t2=load(tp)
 mp=V+'figures_main_noreasoning_diagnostics/main_noreasoning_tom2_online_structured_summary_compact.csv';mm=load(mp)
 labs=['Q2.5 3B','Q2.5 32B','Q2.5 72B','Q2.5 72B','Q3.6 35B','Llama3.3 70B','Gemma4 31B','GPT-5.4']
 display_labs=['Qwen2.5 3B','Qwen2.5 32B','Qwen2.5 72B','Qwen2.5 72B','Qwen3.6 35B-A3B','Llama 3.3 70B','Gemma 4 31B','GPT-5.4']
 fig,axes=plt.subplots(1,2,figsize=(5.5,3.9),layout='constrained',gridspec_kw={'wspace':.06})
 for ax,role in zip(axes,['seeker','hider']):
  arr=np.full((8,6),np.nan);missing={}
  if role=='seeker':
   for i,size in enumerate(['3B','32B','72B']):
    for j,op in enumerate(OP):
     data=t2 if op=='ToM2' else st
     rr=[r for r in data if r['size_label']==size and (op=='ToM2' or r.get('opponent_name')==op)] if op.startswith('ToM') else []
     base=[r for r in rr if r['mode']=='baseline'];new=[r for r in rr if r['mode']=='structured_behavior']
     if base and new:arr[i,j]=float(new[0]['mean_final_payoff'])-float(base[0]['mean_final_payoff'])
   for i,size in enumerate(['3B','32B','72B']):
    files=list((ROOT/E/'supplement_archive_20260921').glob(f'*Qwen2.5-{size}-*ALL_MATCHES*.csv'))
    if files:
     assert len(files)==1;pp=str(files[0].relative_to(ROOT)).replace('\\','/');raw=load(pp)
     assert len(raw)==1000 and all(r['opponent_name']=='ToM1' and 'You are the SEEKER' in r['llm_prompt'] and r['llm_prompt_mode']=='structured_behavior' for r in raw)
     keys={(r['match_id'],r['round']) for r in raw};assert len(keys)==1000
     assert len({r['match_id'] for r in raw})==20 and all(r['llm_parse_mode']=='structured_json:final_action' for r in raw)
     assert all(float(r['payoff_agent0'])==(1 if r['choice_agent0']==r['choice_agent1'] else -1) for r in raw)
     base=[r for r in st if r['size_label']==size and r['opponent_name']=='ToM1' and r['mode']=='baseline'];assert len(base)==1
     arr[i,4]=sum(float(r['payoff_agent0']) for r in raw)/20-float(base[0]['mean_final_payoff'])
   arr[3]=arr[2].copy()
   for i,model in [(5,'Llama-3.3-70B'),(6,'Gemma-4-31B')]:
    r={x['condition']:x for x in mm if x['model']==model};arr[i,5]=float(r['structured_behavior']['mean_final_payoff'])-float(r['baseline']['mean_final_payoff'])
   missing[4,5]='P'
  else:
   for i in [2,3,4,5,6]:missing[i,5]='P'
  gp=E+('gpt54_structured_live_20260921' if role=='seeker' else 'gpt54_hider_structured_20260921')+'/audit.json';g=load(gp);assert g['audit_passed'];arr[7,5]=g['structured_minus_action_only']['mean']
  for j,op in enumerate(OP[:-1]):
   z=extra['structured_minus_action_only'].get(role+'_'+op,{})
   if z.get('complete_cell'):arr[7,j]=z['mean']
   else:missing[7,j]='INC'
  for z in small['contrasts']:
   if z['role']==role:
    j=OP.index(z['opponent']);assert not np.isfinite(arr[0,j]);arr[0,j]=z['difference'];missing.pop((0,j),None)
  for i,label in enumerate(labs):
    for j,op in enumerate(OP):record('S2',role,label,op,float(arr[i,j]) if np.isfinite(arr[i,j]) else None,(gp if op=='ToM2' else extra_path) if i==7 else ('experiments_point_diagnostic/supplement_archive_20260921/ + '+sp if op=='ToM1' and i<4 else sp if op!='ToM2' else tp if i<4 else mp),'observed' if np.isfinite(arr[i,j]) else missing.get((i,j),'not_in_archive'))
  for z in small['contrasts']:
   if z['role']==role:
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']=='Q2.5 3B' and v['opponent']==z['opponent'])
    row.update(source=small_path,status='audited_new_contrast_prompt_caveat',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in structured32['contrasts']:
   if z['role']==role:
    j=OP.index(z['opponent']);assert not np.isfinite(arr[1,j]);arr[1,j]=z['difference'];missing.pop((1,j),None)
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']=='Q2.5 32B' and v['opponent']==z['opponent'])
    row.update(value=z['difference'],source=structured32_path,status='audited_new_unpaired_contrast',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in structured72['contrasts']+structured72wsls['contrasts']:
   if z['role']==role:
    j=OP.index(z['opponent'])
    for i in [2,3]:
     assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    for row in values:
     if row['figure']=='S2' and row['role']==role and row['model']=='Q2.5 72B' and row['opponent']==z['opponent']:
      row.update(value=z['difference'],source=structured72wsls_path if z['opponent']=='WSLS' else structured72_path,status='audited_new_unpaired_contrast',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in new_cells['contrasts']:
   if z['role']==role:
    i=4 if z['model'].startswith('Qwen/') else 5;j=OP.index(z['opponent'])
    assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']==labs[i] and v['opponent']==z['opponent'])
    row.update(value=z['difference'],source=new_path,status='audited_new_unpaired_contrast',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in latest['contrasts']:
   if z['role']==role:
    i=5 if z['model'].startswith('meta-llama/') else 6;j=OP.index(z['opponent'])
    assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']==labs[i] and v['opponent']==z['opponent'])
    row.update(value=z['difference'],source=latest_path,status='audited_new_unpaired_contrast',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in newest['contrasts']:
   if z['role']==role:
    i=1 if z['model'].startswith('Qwen/') else 6;j=OP.index(z['opponent'])
    assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']==labs[i] and v['opponent']==z['opponent'])
    row.update(value=z['difference'],source=newest_path,status='audited_new_contrast_prompt_caveat' if role=='hider' else 'audited_new_unpaired_contrast',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in hider_more['contrasts']:
   if z['role']==role:
    indices=[1] if '32B' in z['model'] else [2,3];j=OP.index(z['opponent'])
    for i in indices:
     assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    for row in values:
     if row['figure']=='S2' and row['role']==role and row['model']==labs[indices[0]] and row['opponent']==z['opponent']:
      row.update(value=z['difference'],source=hider_more_path,status='audited_new_contrast_prompt_caveat',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in hider_next['contrasts']:
   if z['role']==role:
    indices=[2,3] if '72B' in z['model'] else [4];j=OP.index(z['opponent'])
    for i in indices:
     assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    for row in values:
     if row['figure']=='S2' and row['role']==role and row['model']==labs[indices[0]] and row['opponent']==z['opponent']:
      row.update(value=z['difference'],source=hider_next_path,status='audited_new_contrast_prompt_caveat',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in qwen_hider['contrasts']:
   if z['role']==role:
    i=4;j=OP.index(z['opponent'])
    assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']==labs[i] and v['opponent']==z['opponent'])
    row.update(value=z['difference'],source=qwen_hider_path,status='audited_new_contrast_prompt_caveat',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in llama_hider['contrasts']:
   if z['role']==role:
    i=5;j=OP.index(z['opponent'])
    assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']==labs[i] and v['opponent']==z['opponent'])
    row.update(value=z['difference'],source=llama_hider_path,status='audited_new_contrast_prompt_caveat',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  for z in final_cells['contrasts']+c_hider['contrasts']:
   if z['role']==role:
    indices=[2,3] if '72B' in z['model'] else [5] if 'Llama' in z['model'] else [6];j=OP.index(z['opponent'])
    for i in indices:
     assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    for row in values:
     if row['figure']=='S2' and row['role']==role and row['model']==labs[indices[0]] and row['opponent']==z['opponent']:
      row.update(value=z['difference'],source=c_hider_path if '72B' in z['model'] else final_path,status='audited_new_contrast_prompt_caveat',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  c12path=E+'fig2c_hider_tasks1_2_20260922/audit.json';c12=load(c12path);assert c12['audit_passed']
  for z in c12['contrasts']:
   if z['role']==role:
    i=4 if 'Qwen3.6' in z['model'] else 5;j=OP.index(z['opponent'])
    assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']==labs[i] and v['opponent']==z['opponent'])
    row.update(value=z['difference'],source=c12path,status='audited_new_contrast_prompt_caveat',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  c3path=E+'fig2c_hider_task3_20260922/audit.json';c3=load(c3path);assert c3['audit_passed']
  for z in c3['contrasts']:
   if z['role']==role:
    i=6;j=OP.index(z['opponent'])
    assert not np.isfinite(arr[i,j]);arr[i,j]=z['difference'];missing.pop((i,j),None)
    row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']==labs[i] and v['opponent']==z['opponent'])
    row.update(value=z['difference'],source=c3path,status='audited_system_payoff_one_fallback_prompt_caveat',ci_low=z['ci95'][0],ci_high=z['ci95'][1])
  if role=='seeker':
   qp=E+'qwen36_runtime_result_20260922/audit.json';q=load(qp)
   assert q['audit_passed'] and q['n_rows']==1000 and q['fallback']==0
   assert not np.isfinite(arr[4,5])
   arr[4,5]=q['contrast']['difference'];missing.pop((4,5),None)
   row=next(v for v in reversed(values) if v['figure']=='S2' and v['role']==role and v['model']==labs[4] and v['opponent']=='ToM2')
   row.update(value=q['contrast']['difference'],source=qp,status='audited_new_unpaired_contrast_brief_prompt',ci_low=q['contrast']['ci95'][0],ci_high=q['contrast']['ci95'][1])
  mesh=heat(ax,arr,display_labs,70,role.title(),missing,ax==axes[0],{0:'Scaling subset',3:'Flagship comparison'})
 cb=fig.colorbar(mesh,ax=axes,orientation='horizontal',fraction=.06,pad=.04);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.set_label('Structured minus action-only payoff / 50 rounds')
 name='figS2_structured_coverage_20260921';save(fig,name)
 captions.append((name,'S2','Structured-output changes in live payoff across roles and opponents. Positive cells indicate higher mean payoff with structured output; this is not a significance heatmap. Scaling and flagship sections reuse overlapping cells. Restored ToM1 cells come from archived raw outputs. Exactly the behavior prompt is used for archived open models, not whichever prompt performs best. Qwen3.6 seeker non-ToM2 cells are audited: 24/5000 actions repaired, zero fallback; ToM2 replacement also audited, 68/1000 action-only repairs, zero fallback, contrast +2.1[-1.0,5.4] with archived unpaired baseline. Llama seeker RB/QL/WSLS also added, with unpaired archived controls. GPT uses the fresh full5.4 none/2048 matched-glossary controls; other models have different historical prompts/budgets. GPT95%whole-match intervals: seeker+1.7[-1.2,4.5], hider+9.0[3.9,14.1]; no formal role-interaction claim. NA=no matching contrast in these sources. All GPT contrasts have 20 simulator-seed pairs per role/opponent; live histories differ and API randomness is unseeded. Benefits reverse by role/opponent, including ToM0. Recovery preserved original responses; no universal gain. Gray cells are evidence gaps, not null effects.'))
 n,label,cap=captions[-1];cap += ' New Q2.5 3B cells use 20 matches/arm and unpaired match-bootstrap contrasts (exported). Historical structured wording says opponent next hiding action even for hider; interpret role comparisons cautiously. Budgets are 64 baseline versus 384 structured tokens.';captions[-1]=(n,label,cap)
 src=V+'figures_sft_endpoint_formal_20260628/endpoint_online_condition_summary.csv';rr=load(src)
 n,label,cap=captions[-1];cap += ' New32B and72B seeker RB/QL/WSLS contrasts use archived unpaired baselines; runtime confounds remain. Gemma hider ToM2 includes 1/1000 random fallback after an incomplete reply; system payoff, not pure parsed-model performance.';captions[-1]=(n,label,cap)
 # The current S2 subset contains no unavailable cells; state the selected sizes instead.
 n,label,cap=captions[-1]
 cap=cap.replace('Scaling and flagship sections reuse overlapping cells.', 'The scaling subset contains Qwen2.5 3B, 32B and 72B; the 72B row is repeated in the flagship section.')
 cap=cap.replace('NA=no matching contrast in these sources. ','').replace('Gray cells are evidence gaps, not null effects. ','')
 captions[-1]=(n,label,cap)
 fig,axes=plt.subplots(2,2,figsize=(5.5,5.2),layout='constrained')
 for row,(base,new,title) in enumerate([('base_action_only','action_s2400','Action-only SFT'),('base_structured','structured_s0300','Structured-output SFT')]):
  for col,role in enumerate(['seeker','hider']):
   ax=axes[row,col]
   for k,cond in enumerate([base,new]):
    y=[];low=[];high=[]
    for op in OP:
     r=[r for r in rr if r['eval_set']=='main' and r['condition']==cond and r['llm_role']==role and r['opponent_name']==op];assert len(r)==1;r=r[0]
     v=float(r['mean_payoff']);lo=float(r['mean_payoff_ci_low']);hi=float(r['mean_payoff_ci_high']);y.append(v);low.append(v-lo);high.append(hi-v)
     record('S3',role,cond,op,v,src,lo=lo,hi=hi)
    ax.errorbar(np.arange(6)+(k-.5)*.18,y,yerr=[low,high],fmt=['o','s'][k],color=C['initial'] if role=='seeker' else C['intervention'],capsize=2)
   ax.axhline(0,color=C['constant'],ls='--',lw=.8);ax.set(xticks=range(6),xticklabels=OP,ylim=(-30,55),xlabel='Opponent',ylabel='Payoff / 50-round match')
   ax.set_title(title+'\n'+role.title(),loc='left',fontweight='normal')
   if row==0 and col==0:ax.legend(handles=[Line2D([],[],color='#5E6870',marker='o',linestyle='none',label='Before SFT'),Line2D([],[],color='#5E6870',marker='s',linestyle='none',label='After SFT')],frameon=False,fontsize=7,loc='upper left')
 name='figS3_sft_full_opponents_20260921';save(fig,name)
 captions.append((name,'S3','Behavioral SFT across all six original benchmark opponents. Qwen2.5-7B-Instruct, separate action-only and structured training packages, both roles. Each cell contains20live50-round matches; whiskers are archived95%whole-match bootstrap intervals, not training-seed uncertainty. Action-only endpoint is step2400; structured endpoint is step300. Compare each endpoint to its matching output-interface baseline; these packages are not an equal-update ablation. All roles/opponents use the same vertical scale. This broadens Figure2D without changing its data; higher-order and GPT-opponent transfer remain separate evaluations. Good task payoff is not direct evidence of belief recovery.'))
 with (OUT/'supplement_plotted_values.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 (OUT/'supplement_provenance.json').write_text(json.dumps({'inputs':sources,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':checks,'new_runs':0,'source_scope':'Archived summaries plus audited GPT controls; not a fresh raw-data audit of every historical cell.'},indent=2),encoding='utf-8')
 (OUT/'supplement_captions.json').write_text(json.dumps(captions,indent=2),encoding='utf-8')
 return captions
