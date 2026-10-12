"""Reproduce Figures 2-4 from archived data; no model calls or training."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz
from paper_style import apply, panel, legend, layout, COLORS as C, DIVERGING

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
PDF=OUT/'pdf'; PDF.mkdir(parents=True,exist_ok=True)
E='experiments_point_diagnostic/'
inputs={}; values=[]; figures=[]; checks=[]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path):
    p=ROOT/path; inputs[path]=sha(p)
    return list(csv.DictReader(p.read_text(encoding='utf-8-sig').splitlines())) if p.suffix=='.csv' else json.loads(p.read_text(encoding='utf-8'))
old='outputs/iclr_main_figures_20260917/'
prov=load(old+'provenance.json')
for n,h in prov['input_sha256'].items():
    assert sha(ROOT/n)==h, n
    inputs[n]=h
arch=load(old+'plotted_values.csv')
assert inputs[old+'plotted_values.csv']=={k.replace('\\','/'):v for k,v in prov['outputs'].items()}[old+'plotted_values.csv']
g03path=E+'g03_job13228623/audit_metrics.json'; g03=load(g03path)
r5path=E+'r5_results_20260918/audit_20260918.json'; r5=load(r5path)
cpath=E+'closure_retry_merged_20260917/supplementary_audit.json'; c=load(cpath)
dpath=E+'gpt_depth_retry_merged_20260917/audit_results.json'; depth=load(dpath)
assert g03['audit_passed'] and c['passed'] and depth['audit_passed']
assert r5['status']=='independently_audited'
g=c['groups']; M=r5['metrics']; D=r5['comparisons']
apply()
def record(f,p,label,v,metric,source,lo=None,hi=None):
    values.append(dict(figure=f,panel=p,label=label,value=float(v),metric=metric,source=source,ci_low=lo,ci_high=hi))
def archived(f,p,label):
    rr=[r for r in arch if r['figure']==str(f) and r['panel']==p and r['label']==label]
    assert len(rr)==1,(f,p,label)
    r=rr[0]
    return float(r['value']),float(r['ci_low']) if r['ci_low'] else None,float(r['ci_high']) if r['ci_high'] else None,r
def ap(ax,x,of,op,label,nf,npanel,col,marker='o',name=None):
    v,lo,hi,r=archived(of,op,label)
    ax.errorbar(x,v,yerr=None if lo is None else [[v-lo],[hi-v]],fmt=marker,color=col,capsize=2,label=name)
    record(nf,npanel,label,v,r['metric'],r['source'],lo,hi)
def grid(rows,height):
    f,a=plt.subplots(rows,2,figsize=(5.5,height),layout='constrained');layout(f);return f,a
def hpoints(ax,f,p,items,metric,src):
    for i,(label,v,col,marker) in enumerate(items):
        ax.plot(v,i,marker=marker,color=col,ls='none')
        record(f,p,label,v,metric,src)
    ax.set_yticks(range(len(items)),[x[0] for x in items]);ax.invert_yaxis()
    ax.set_xlim(left=0);ax.grid(axis='y',visible=False);ax.grid(axis='x')
def save(fig,name,caption):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); outside=[];fonts=[]
    texts=list(fig.texts)
    for ax in fig.axes:
        texts += [ax.title,ax._left_title,ax.xaxis.label,ax.yaxis.label,*ax.texts]
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            texts += [t.label1 for t in axis.get_major_ticks() if min(limits)<=t.get_loc()<=max(limits)]
        if ax.get_legend():texts+=ax.get_legend().get_texts()
    for t in texts:
        if not t.get_visible() or not t.get_text():continue
        fonts.append(t.get_fontsize());b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:outside.append(t.get_text())
    assert not outside,(name,outside)
    assert min(fonts)>=7
    for ext in ['svg','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=300)
    fig.savefig(PDF/f'{name}.pdf')
    p=fitz.open(PDF/f'{name}.pdf');assert abs(p[0].rect.width-396)<.1 and not p[0].get_images(full=True)
    p[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(FIG/f'{name}_pdf_render.png')
    checks.append(dict(figure=name,width_inches=5.5,height_inches=fig.get_figheight(),minimum_font_pt=min(fonts),outside_text=outside,raster_images=0))
    figures.append((name,caption));plt.close(fig)

# FIGURE 2: performance is elicitable, but neither scale nor transfer is automatic.
fig=plt.figure(figsize=(5.5,6.8),layout='constrained');layout(fig)
gs=fig.add_gridspec(3,2,height_ratios=[1,2.25,1.5])
cgrid=gs[1,:].subgridspec(2,1,hspace=.02)
axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(cgrid[0,0]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])]
hider_ax=fig.add_subplot(cgrid[1,0])
a=axes[0];panel(a,'a','Model size')
sizes=['0.5','1.5','3','7','14','32','72']
for j,role in enumerate(['seeker','hider']):
    for i,size in enumerate(sizes):ap(a,i+(j-.5)*.16,2,'a',f'{role}_{size}B',2,'a',[C['initial'],C['intervention']][j],'o',role.title() if i==0 else None)
a.set(xticks=range(7),xticklabels=sizes,xlabel='Qwen2.5 size (B parameters)',ylabel='Payoff / 50-round match');a.axhline(0,c=C['constant'],ls='--',lw=.8);legend(a,ncol=2,loc='lower right')
a=axes[1];panel(a,'b','Reasoning setting')
names=['gemma-4-31B-it','gemma-4-31B-it (thinking)','gpt-oss-20b (effort=low)','gpt-oss-20b (effort=medium)','gpt-oss-20b (effort=high)']
for i,n in enumerate(names):ap(a,i-.14,2,'b',n,2,'b',C['initial'],'o',name='Seeker' if i==0 else None)
gemma_standard_path=E+'fig2c_hider_20260921/baseline_audit.json';gemma_standard=load(gemma_standard_path)
assert gemma_standard['audit_passed']
standard=gemma_standard['metrics']['google/gemma-4-31B-it']
assert standard['matches']==20 and standard['rounds']==1000 and standard['strict_parse']==1000
assert sha(ROOT/standard['source'])==standard['sha256']
standard_frac=(1+standard['mean']/50)/2
standard_lo,standard_hi=[(1+v/50)/2 for v in standard['ci95']]
a.errorbar(.14,standard_frac,yerr=[[standard_frac-standard_lo],[standard_hi-standard_frac]],fmt='o',color=C['intervention'],capsize=2,label='Hider')
record(2,'b','gemma-4-31B-it standard hider',standard_frac,'online_action_accuracy',gemma_standard_path,standard_lo,standard_hi)
gemma_path=E+'gemma_thinking_tom2_20260925/audit.json';gemma=load(gemma_path)
assert gemma['audit_passed'] and gemma['n_matches']==20 and gemma['n_rows']==1000
assert gemma['fallback_random']==30 and gemma['prompt_exact']==1000
gemma_frac=gemma['mean_correct_action_fraction']
gemma_lo,gemma_hi=gemma['ci95_correct_action_fraction']
a.errorbar(1+.14,gemma_frac,yerr=[[gemma_frac-gemma_lo],[gemma_hi-gemma_frac]],fmt='o',color=C['intervention'],capsize=2)
record(2,'b','gemma-4-31B-it (thinking) hider',gemma_frac,'online_action_accuracy',gemma_path,gemma_lo,gemma_hi)
oss_low_path=E+'oss_hider_low_20260923/audit.json';oss_low=load(oss_low_path)
assert oss_low['audit_passed'] and oss_low['n_matches']==20 and oss_low['n_rows']==1000
assert oss_low['fallback_random']==3 and oss_low['prompt_exact']==1000
oss_frac=(1+oss_low['mean_final_payoff']/50)/2
oss_lo,oss_hi=[(1+v/50)/2 for v in oss_low['ci95_match_bootstrap']]
a.errorbar(2+.14,oss_frac,yerr=[[oss_frac-oss_lo],[oss_hi-oss_frac]],fmt='o',color=C['intervention'],capsize=2)
record(2,'b','gpt-oss-20b (effort=low) hider',oss_frac,'online_action_accuracy',oss_low_path,oss_lo,oss_hi)
oss_medium_path=E+'oss_hider_medium_20260925/audit.json';oss_medium=load(oss_medium_path)
assert oss_medium['audit_passed'] and oss_medium['n_matches']==20 and oss_medium['n_rows']==1000
assert oss_medium['fallback_random']==4 and oss_medium['prompt_exact']==1000
medium_frac=oss_medium['mean_correct_action_fraction']
medium_lo,medium_hi=oss_medium['ci95_correct_action_fraction']
a.errorbar(3+.14,medium_frac,yerr=[[medium_frac-medium_lo],[medium_hi-medium_frac]],fmt='o',color=C['intervention'],capsize=2)
record(2,'b','gpt-oss-20b (effort=medium) hider',medium_frac,'online_action_accuracy',oss_medium_path,medium_lo,medium_hi)
oss_high_path=E+'oss_hider_high_20260924/audit.json';oss_high=load(oss_high_path)
assert oss_high['audit_passed'] and oss_high['n_matches']==20 and oss_high['n_rows']==1000
assert oss_high['fallback_random']==10 and oss_high['prompt_exact']==1000
high_frac=(1+oss_high['mean_final_payoff']/50)/2
high_lo,high_hi=[(1+v/50)/2 for v in oss_high['ci95_match_bootstrap']]
a.errorbar(4+.14,high_frac,yerr=[[high_frac-high_lo],[high_hi-high_frac]],fmt='o',color=C['intervention'],capsize=2)
record(2,'b','gpt-oss-20b (effort=high) hider',high_frac,'online_action_accuracy',oss_high_path,high_lo,high_hi)
a.set(xticks=range(5),xticklabels=['Gemma 4 31B\nstandard','Gemma 4 31B\nthinking','GPT-OSS 20B\nlow','GPT-OSS 20B\nmedium','GPT-OSS 20B\nhigh'],xlabel='Model / reasoning setting',ylabel='Correct-action fraction',ylim=(.25,.72));a.axhline(.5,c=C['constant'],ls='--',lw=.8)
plt.setp(a.get_xticklabels(),rotation=35,ha='right')
a.legend(loc='upper center',ncol=2,frameon=False,fontsize=7)
a=axes[2];panel(a,'c','Structured output')
sp='Hide_and_Seek/Visualization/figures_structured_tom2/structured_tom2_intervention_summary.csv'
mp='Hide_and_Seek/Visualization/figures_main_noreasoning_diagnostics/main_noreasoning_tom2_online_structured_summary_compact.csv'
ss=load(sp);mm=load(mp)
models=[('Qwen2.5\n72B',ss,'size_label','72B',sp),('Qwen3.6\n35B-A3B',mm,'model','Qwen3.6-35B-A3B',mp),('Llama 3.3\n70B',mm,'model','Llama-3.3-70B',mp),('Gemma 4\n31B',mm,'model','Gemma-4-31B',mp),('GPT-5.4',None,None,'GPT-5.4',None)]
for i,(label,rows,key,val,src) in enumerate(models):
    if val=='Qwen3.6-35B-A3B':
        qp=E+'qwen36_runtime_result_20260922/audit.json';qa=load(qp)
        assert qa['audit_passed'] and qa['n_rows']==1000 and qa['n_matches']==20 and qa['fallback']==0
        vv=[]
        for j,(v,ci) in enumerate([(qa['contrast']['baseline_mean'],qa['contrast']['baseline_ci95']),(qa['mean_final_payoff'],qa['ci95'])]):
            lo,hi=ci;vv.append(v)
            a.plot(i+(j-.5)*.20,v,['o','s'][j],color=C['initial'])
            record(2,'c',val+('_baseline' if j==0 else '_structured_behavior_brief'),v,'payoff_per_50_round_match',qp,lo,hi)
        a.plot([i-.1,i+.1],vv,color='#bbbbbb',zorder=0)
        continue
    if val=='GPT-5.4':
        gp=E+'gpt54_structured_live_20260921/audit.json'
        if not (ROOT/gp).exists():
            a.text(i,-8,'PENDING\nlocal live control',ha='center',va='center',fontsize=7,color=C['constant'])
            continue
        audit=load(gp);assert audit['audit_passed'] and audit['unique_round_keys']==2000
        vv=[]
        for j,arm in enumerate(['action_only','structured']):
            z=audit['metrics'][arm];v=z['mean'];lo,hi=z['ci95'];vv.append(v)
            a.plot(i+(j-.5)*.20,v,['o','s'][j],color=C['initial'])
            record(2,'c','GPT-5.4_'+arm,v,'payoff_per_50_round_match',gp,lo,hi)
        a.plot([i-.1,i+.1],vv,color='#bbbbbb',zorder=0)
        continue
    vv=[]
    for j,cond in enumerate(['baseline','structured_behavior']):
        rr=[r for r in rows if r[key]==val and r.get('mode',r.get('condition'))==cond];assert len(rr)==1
        v=float(rr[0]['mean_final_payoff']);vv.append(v)
        a.plot(i+(j-.5)*.20,v,['o','s'][j],color=C['initial'])
        record(2,'c',val+'_'+cond,v,'payoff_per_50_round_match',src)
    a.plot([i-.1,i+.1],vv,color='#bbbbbb',zorder=0)
a.set(xticks=range(len(models)),xticklabels=[],ylim=(-26,15));a.axhline(0,c=C['constant'],ls='--',lw=.8)
output_handles=[Line2D([],[],marker='o',linestyle='None',color=C['constant'],label='Action-only'),Line2D([],[],marker='s',linestyle='None',color=C['constant'],label='Structured')]
a.legend(handles=output_handles,ncol=2,loc='upper left',frameon=False,fontsize=7)
a.text(.98,.92,'Seeker',transform=a.transAxes,ha='right',va='top',fontsize=7,fontweight='bold')
a=hider_ax
bp=E+'fig2c_hider_20260921/baseline_audit.json';ba=load(bp);assert ba['audit_passed']
ids=['Qwen/Qwen2.5-72B-Instruct','Qwen/Qwen3.6-35B-A3B','meta-llama/Llama-3.3-70B-Instruct','google/gemma-4-31B-it']
for i,mid in enumerate(ids):
    z=ba['metrics'][mid];v=z['mean'];lo,hi=z['ci95']
    assert sha(ROOT/z['source'])==z['sha256']
    a.plot(i-.1,v,'o',color=C['intervention'])
    record(2,'c',mid+'_hider_action_only',v,'payoff_per_50_round_match',bp,lo,hi)
    if i in [0,1,2,3]:
        hp=E+('fig2c_hider_task0_20260922/audit.json' if i==0 else 'fig2c_hider_task3_20260922/audit.json' if i==3 else 'fig2c_hider_tasks1_2_20260922/audit.json');hh=load(hp);assert hh['audit_passed'];zz=next(z for z in hh['cells'] if z['model']==mid)
        nv=zz['mean_final_payoff'];nl,nh=zz['ci95']
        a.plot(i+.1,nv,'s',color=C['intervention'])
        a.plot([i-.1,i+.1],[v,nv],color='#bbbbbb',zorder=0)
        record(2,'c',mid+'_hider_structured',nv,'payoff_per_50_round_match',hp,nl,nh)
    else:a.text(i+.12,4,'PENDING',ha='center',fontsize=7,color=C['constant'])
gp=E+'gpt54_hider_structured_20260921/audit.json'
if (ROOT/gp).exists():
    ga=load(gp);assert ga['audit_passed'] and ga['unique_round_keys']==2000
    vv=[]
    for j,arm in enumerate(['action_only','structured']):
        z=ga['metrics'][arm];v=z['mean'];lo,hi=z['ci95'];vv.append(v)
        a.plot(4+(j-.5)*.2,v,['o','s'][j],color=C['intervention'])
        record(2,'c','GPT-5.4_hider_'+arm,v,'payoff_per_50_round_match',gp,lo,hi)
    a.plot([3.9,4.1],vv,color='#bbbbbb',zorder=0)
else:a.text(4,-8,'PENDING',ha='center',fontsize=7,color=C['constant'])
a.set(xticks=range(len(models)),xticklabels=[m[0] for m in models],ylabel='Payoff / 50 rounds',ylim=(-26,15));a.axhline(0,c=C['constant'],ls='--',lw=.8)
a.text(.98,.92,'Hider',transform=a.transAxes,ha='right',va='top',fontsize=7,fontweight='bold')
a=axes[3];panel(a,'d','Structured-output SFT')
for j,(cond,name) in enumerate([('base_structured','Before SFT'),('structured_s0300','After SFT')]):
    for i,(tom,role) in enumerate([(t,r) for t in ['ToM2','ToM3'] for r in ['seeker','hider']]):ap(a,i+(j-.5)*.18,2,'c',f'{cond}_{tom}_{role}',2,'d',C['initial'] if role=='seeker' else C['intervention'],['o','s'][j])
a.set(xticks=range(4),xticklabels=['Seeker\nvs ToM2','Hider\nvs ToM2','Seeker\nvs ToM3','Hider\nvs ToM3'],xlabel='Qwen2.5-7B live play',ylabel='Payoff / 50-round match',ylim=(-19,12));a.axhline(0,c=C['constant'],ls='--',lw=.8);a.axvline(1.5,c=C['constant'],ls=':',lw=.8)
stage_handles=[Line2D([],[],marker='o',linestyle='None',color=C['constant'],label='Before SFT'),Line2D([],[],marker='s',linestyle='None',color=C['constant'],label='After SFT')]
a.legend(handles=stage_handles,ncol=2,loc='upper center',frameon=False,fontsize=7)
a=axes[4];panel(a,'e','Transfer to GPT-5.4')
# Within each training stage: action-only seeker/hider, then structured seeker/hider.
transfer_series=[
    ('action-only','seeker','base_action_only','action_s2400','-', 'o',C['initial'],-.09),
    ('action-only','hider','base_action_only','action_s2400','-', 'o',C['intervention'],-.03),
    ('structured','seeker','base_structured','structured_s0300','--','s',C['initial'],.03),
    ('structured','hider','base_structured','structured_s0300','--','s',C['intervention'],.09),
]
for fmt,role,base,trained,ls,mark,color,offset in transfer_series:
    vv=[]
    for x,cond in [(0,base),(1,trained)]:
        v,lo,hi,r=archived(2,'d',f'{cond}_{role}')
        a.errorbar(x+offset,v,yerr=[[v-lo],[hi-v]],fmt=mark,color=color,capsize=2)
        record(2,'e',f'{cond}_{role}',v,r['metric'],r['source'],lo,hi)
        vv.append(v)
    a.plot([offset,1+offset],vv,ls=ls,color=color,label=f'{fmt.title()}, {role}')
a.set(xticks=[0,1],xticklabels=['Before SFT','After SFT'],xlabel='Qwen2.5-7B live play',ylabel='Payoff / 50-round match',ylim=(-42,65),yticks=[-40,-20,0,20,40,60],xlim=(-.25,1.25));a.axhline(0,c=C['constant'],ls='--',lw=.8);legend(a,loc='upper right',ncol=1)
save(fig,'fig2_behavior_20260921',
"Performance across scale, reasoning and structured output. (a) Qwen2.5 scaling vs ToM2. (b) Archived seeker results, an audited historical Gemma standard hider cell, and audited Gemma thinking and OSS low-, medium- and high-effort hider cells (20 live matches of 50 rounds each). Gemma standard hider has 1,000 strict parses from an archived action-only run; its runtime differs from the new thinking run. Gemma thinking used 30 prespecified random fallbacks after truncated final-channel replies; OSS low/medium/high used 3/4/10 fallbacks after unparseable replies. All rounds remain in the deployed-system results. Intervals resample whole matches. (c) Separate seeker/hider comparisons, 20 live matches per condition, 50 rounds each. All four open-model hider arms and the Qwen3.6 seeker replacement are audited. Qwen3.6 seeker used brief structured output: 932 strict/68 action-only repairs, zero fallback; contrast +2.1, 95% CI [-1.0,5.4], inconclusive. Gemma hider includes 1/1000 random fallback (incomplete reply). Qwen3.6 hider: 1/1000 action-only repair, zero fallback. Historical hider prompt wording is role-ambiguous despite correct scoring instructions. Archived hider action-only baselines passed key/role/payoff/parse checks. GPT-5.4 uses fresh matched action-only/structured controls, identical candidate glossary, reasoning none/2048; not the medium-reasoning belief diagnostic. Prompt variants/budgets differ across families; Qwen3.6 uses brief output. Panel c uniformly omits error bars because verified match-level uncertainty is unavailable for all archived seeker conditions; audited intervals remain in the source table. (d) Qwen2.5-7B structured-output SFT against ToM2 and held-out ToM3. (e) Action-only and structured-output SFT transfer against GPT-5.4. Panels d-e use 20 matches/cell and archived intervals. ToM3 was absent from training and is a within-game higher-order transfer opponent. GPT in (e) is the opponent. Baselines and interventions are separate live runs, not common histories; historical comparisons retain runtime confounds. No universal structured-output benefit is established.")

# FIGURE 3: diagnose both targets, test supervision, then test its boundary.
# These figures pool both focal roles in several panels. Reserve Figure 2's
# blue/orange pair for seeker/hider; use a separate condition palette here.
DIAG=dict(start='#7C8790',focus='#2D8A78',extension='#8B6AA3',
          constant='#4B535B',frequency='#A28A45')
t04path=E+'t04_job13242123/audit_metrics.json';t04=load(t04path)
assert t04['audit_passed']
fig=plt.figure(figsize=(5.5,6.0))
ax=np.empty((3,2),dtype=object)
for row in range(3):
    for col in range(2):ax[row,col]=fig.add_axes([[.12,.64][col],[.72,.41,.10][row],.33,.18])
def diagnostic_panel(axis, letter, condition):
    # Keep headings to one line; the axes carry the outcome and units.
    axis.set_title(f'{letter}  {condition}',loc='left',fontweight='normal',pad=6)
    axis.set_axisbelow(True)
    axis.grid(axis='y')
for col,stage,letter,metric,ylabel,limit in [
    (0,'action','a','mse','Action-probability MSE',.23),
    (1,'belief','b','mse','Opponent-belief MSE',.035)]:
    a=ax[0,col];diagnostic_panel(a,letter,'Behavioral SFT');gm=t04['stages'][stage]['metrics']
    for i,k in enumerate(['original_qwen','original_parent']):
        v=gm[k][metric];a.bar(i,v,.5,color=[DIAG['start'],DIAG['focus']][i]);record(3,letter,k,v,stage+'_'+metric,t04path)
    for k,ls,color in [('constant','--',DIAG['constant']),('frequency',':',DIAG['frequency'])]:
        v=gm[k][metric];a.axhline(v,c=color,ls=ls,label=k.title());record(3,letter,k,v,stage+'_'+metric,t04path)
    a.set(xticks=[0,1],xticklabels=['Initial','Behavioral SFT'],ylabel=ylabel,ylim=(0,limit),
          xlabel='0/1 token-probability probe' if col==0 else 'Direct numeric belief report')
    legend(a,loc='upper left',ncol=2)
a=ax[1,0];diagnostic_panel(a,'c','Probability SFT')
items=[]
for label,name,col,mark in [('Raw / A format','Initial Qwen',DIAG['start'],'o'),('Trained A','Action-probability output SFT',DIAG['focus'],'s'),('Trained B','State + action-probability output SFT',DIAG['extension'],'^'),('Constant','Constant',DIAG['constant'],'D'),('Frequency','Frequency',DIAG['frequency'],'v')]:
    v,lo,hi,r=archived(5,'a',label);items.append((name,v,col,mark))
for i,(name,v,col,mark) in enumerate(items):
    if i<3:a.bar(i,v,.58,color=col)
    else:a.axhline(v,c=col,ls='--' if i==3 else ':',label=name)
    record(3,'c',name,v,'penalized_action_probability_mse',old+'plotted_values.csv')
a.set(xticks=range(3),xticklabels=['Initial\n(no SFT)','Probability\nonly','State +\nprobability'],xlabel='Supervised output',ylabel='Action-probability MSE',ylim=(0,.135));legend(a,loc='upper left',ncol=2)
a=ax[1,1];diagnostic_panel(a,'d','Five-field SFT')
gm=t04['stages']['belief']['metrics']
for start,end,name,color,marker in [('original_qwen','base_selected','Initial start',DIAG['start'],'o'),('original_parent','parent_selected','Behavioral start',DIAG['focus'],'s')]:
    vv=[gm[k]['mse'] for k in (start,end)]
    a.plot([0,1],vv,color=color,marker=marker,label=name)
    for k,v in zip((start,end),vv):record(3,'d',k,v,'belief_mse',t04path)
for k,ls,color in [('constant','--',DIAG['constant']),('frequency',':',DIAG['frequency'])]:
    a.axhline(gm[k]['mse'],color=color,ls=ls,label=k.title());record(3,'d',k,gm[k]['mse'],'belief_mse',t04path)
a.set(xticks=[0,1],xticklabels=['Initial','After SFT'],ylabel='Opponent-belief MSE',ylim=(0,.035),xlim=(-.15,1.15));legend(a,loc='upper left',ncol=2)
a=ax[2,0];diagnostic_panel(a,'e','Probability SFT')
for j,(model,name,col,mark) in enumerate([('original_A500','Before',DIAG['start'],'^'),('known','All specified',DIAG['focus'],'o'),('mixed','Half specified',DIAG['extension'],'s')]):
    vv=[]
    for info in ['known','unknown']:
        v=M[model]['fresh'][info]['all']['mse'];vv.append(v);record(3,'e',model+'_'+info,v,'action_probability_mse',r5path)
    a.plot([0,1],vv,marker=mark,color=col,label=name)
base=M['mixed']['fresh']['unknown']['all']['constant'];a.axhline(base,c=DIAG['constant'],ls='--');record(3,'e','constant',base,'action_probability_mse',r5path)
a.set(xticks=[0,1],xticklabels=['History +\nspecification','History\nonly'],xlabel='Test input',ylabel='Action-probability MSE',ylim=(0,.135));legend(a,loc='upper left')
a=ax[2,1];diagnostic_panel(a,'f','Probability SFT')
for j,(suffix,col,mark,name) in enumerate([('mixed_minus_known',DIAG['extension'],'o','Half - all'),('mixed_minus_constant',DIAG['constant'],'s','Half - constant')]):
    for i,(role,k) in enumerate([(r,k) for r in [0,1] for k in range(3)]):
        z=D[f'fresh_unknown_A{role}_ToM{k}_{suffix}'];v=z['delta_mse'];lo,hi=z['ci95']
        a.errorbar(v,i+(j-.5)*.22,xerr=[[v-lo],[hi-v]],fmt=mark,color=col,capsize=2,label=name if i==0 else None)
        record(3,'f',f'A{role}_ToM{k}_{suffix}',v,'action_probability_mse_difference',r5path,lo,hi)
a.set(yticks=range(6),yticklabels=['Seeker / ToM0','Seeker / ToM1','Seeker / ToM2','Hider / ToM0','Hider / ToM1','Hider / ToM2'],ylim=(5.5,-1.9),xlabel='MSE difference\n(negative = better)',ylabel='Role / opponent');a.axvline(0,c=DIAG['constant'],ls='--');a.grid(axis='y',visible=False);a.grid(axis='x');legend(a,loc='upper left')
a.set_ylim(5.5,-3.3)
a.set_xticks([-.10,-.05,0])
save(fig,'fig3_qwen_prediction_20260921',
"Fixed-history evaluation of Qwen2.5-7B. (a-b) Initial versus behavioral checkpoint300: 60 decision points from six matches. Action prediction uses 0/1 token confidence; belief reports estimate the opponent's prediction about us. Figure2 runtime tensor identity is not re-attested. (c) Both SFT conditions receive full history and opponent specification: ToM type, parameters (volatility=-2, b_temp=-1, bias=0), state-coordinate definitions and initial values. Neither receives the true current state or exact update equations. Outputs are action probability alone or state plus action probability. The latter receives true previous state during training and its own previous estimate during evaluation. Valid outputs:600/600 and582/600; invalid loss1. This is not a matched repair of (a). (d) Five-field supervision includes belief targets; not belief-only training. Initial marks each five-field run's own starting checkpoint, including the behavioral-SFT start. (e-f) Action-probability output throughout on the fresh R5 cohort. Extra training supplies specification on all examples or half (the remainder history only). Both start from (c)'s probability-output checkpoint. (e) Raw action-probability MSE by test information; (f) history-only MSE differences by opponent for the same continuations. Fresh60 matches,540 noninitial points/condition. History-only input removes type, parameters, state schema and initial values. (f) 95% match-bootstrap intervals, not training-seed uncertainty. Gains vary by opponent. These results do not establish mechanism understanding or live-transfer improvement.")

# FIGURE 4: frontier-model information use is separable from inference.
fig,ax=grid(3,5.8)
a=ax[0,0];panel(a,'a','Rule execution')
for j,cond in enumerate(['K','R']):
    vals=[]
    for k in range(3):
        v,_,_,r=archived(3,'a',f'{cond}_ToM{k}');vals.append(v);record(4,'a',f'{cond}_ToM{k}',v,'fraction_abs_error_le_0.001',r['source'])
    a.bar(np.arange(3)+(j-.5)*.34,vals,.31,color=DIAG['focus'] if j==0 else DIAG['extension'],hatch='' if j==0 else '//',label=['History + rules','Own state + rules'][j])
a.set(xticks=range(3),xticklabels=['ToM0','ToM1','ToM2'],xlabel='Opponent level',ylabel='Fraction with error ≤ 0.001',ylim=(0,1.40),yticks=[0,.5,1]);legend(a,loc='upper left')
a=ax[0,1];panel(a,'b','No-rule forecasts')
for j,cond in enumerate(['rep_L1_N','rep_L2_N']):
    for i,cell in enumerate([f'{r}_ToM{k}' for r in ['seeker','hider'] for k in range(3)]):
        z=g[cond][cell];v=z['q']['mse']-z['constant']['mse']
        a.plot(v,i+(j-.5)*.2,['o','s'][j],color=[DIAG['focus'],DIAG['extension']][j],label=['Action forecast only','Belief then forecast'][j] if i==0 else None);record(4,'b',cond+'_'+cell,v,'action_probability_mse_minus_constant',cpath)
a.set(yticks=range(6),yticklabels=['Seeker / ToM0','Seeker / ToM1','Seeker / ToM2','Hider / ToM0','Hider / ToM1','Hider / ToM2'],ylim=(5.5,-3),xlabel='MSE - constant\n(negative = better)',ylabel='GPT role / opponent');a.axvline(0,c=DIAG['constant'],ls='--');a.grid(axis='y',visible=False);a.grid(axis='x');legend(a,loc='upper left')
a=ax[1,0];panel(a,'c','Belief reports')
z=g['rep_L2_N']['all']
hpoints(a,4,'c',[(name,z[key]['mse'],col,mark) for name,key,col,mark in [('GPT belief report','belief',DIAG['extension'],'s'),('Constant','belief_constant',DIAG['constant'],'D'),('Frequency','belief_frequency',DIAG['frequency'],'v'),('Supervised control*','belief_supervised',DIAG['focus'],'^')]],'belief_mse',cpath)
a.set(xlabel='Opponent-belief MSE')
a=ax[1,1];panel(a,'d','Belief intervention')
for i,k in enumerate(['self','true','shuffled']):
    v=g['b_'+k]['all']['q']['mse'];a.bar(i,v,.6,color=[DIAG['start'],DIAG['focus'],DIAG['extension']][i],hatch=['','//','xx'][i]);record(4,'d',k,v,'action_probability_mse',cpath)
a.set(xticks=range(3),xticklabels=['Self','True','Shuffled'],xlabel='Supplied opponent belief',ylabel='Action-probability MSE',ylim=(0,.08))
a=ax[2,0];panel(a,'e','Choice regret')
for i,k in enumerate(['b_true-b_self','b_true-b_shuffled','q_true-q_self']):
    z=c['common_valid_contrasts'][k];v=z['match_mean_mse_regret_delta'][1];lo,hi=z['ci'][1]
    a.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=DIAG['focus'],capsize=2);record(4,'e',k,v,'expected_regret_difference',cpath,lo,hi)
a.set(yticks=range(3),yticklabels=['True b - self b','True b - shuffled b','True q - self q'],xlabel='Expected regret difference',ylabel='Supplied-information contrast',ylim=(2.6,-.6));a.axvline(0,c=DIAG['constant'],ls='--');a.grid(axis='y',visible=False);a.grid(axis='x')
a=ax[2,1];panel(a,'f','Report depth')
conds=['L1_N','L1_M','L2_N','L2_M','L3_N','L3_M']
arr=np.array([[depth['groups'][f'ToM{k}'][cond]['mse']-depth['groups'][f'ToM{k}']['constant']['mse'] for k in range(3)] for cond in conds])
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
diagnostic_diverging=LinearSegmentedColormap.from_list('diagnostic_mse',[DIAG['focus'],'#F8F7F2',DIAG['extension']])
im=a.pcolormesh(np.arange(4)-.5,np.arange(7)-.5,arr,cmap=diagnostic_diverging,norm=TwoSlopeNorm(vmin=-.06,vcenter=0,vmax=.06),rasterized=False)
a.set_ylim(5.5,-.5)
for i,cond in enumerate(conds):
    for k in range(3):
        a.text(k,i,f'{arr[i,k]:+.3f}',ha='center',va='center',fontsize=7,color='white' if abs(arr[i,k])>.04 else 'black');record(4,'f',cond+f'_ToM{k}',arr[i,k],'action_probability_mse_minus_constant',dpath)
a.set(xticks=range(3),xticklabels=['ToM0','ToM1','ToM2'],yticks=range(6),yticklabels=['L1','L1 + memory','L2','L2 + memory','L3','L3 + memory'],xlabel='Opponent level',ylabel='Requested report depth');a.grid(False)
a.set_xlabel('Opponent level\nCell: MSE - constant')
save(fig,'fig4_gpt_diagnosis_20260921',
"GPT-5.4 medium: using opponent information differs from inferring it. q is opponent action probability; b is its belief about the focal player's action. (a) Exact rules supplied, 18 scheduled updates/level; missing outputs remain in the denominator. Own-state execution completes 5/6 chains. (b-c) Separate 24-match no-rule replication, 216 noninitial predictions/arm; S/H denote GPT seeker/hider. *The supervised control has independent task-specific training, not equal zero-shot prior information. (d-e) Matched interventions on six matches, 54 decisions/arm: panel d shows forecast MSE and panel e expected-regret differences. The same audit records 52/54 reference-best actions with true b versus 36/54 with self b; agreement is not plotted. Panel e: 95% paired match-bootstrap intervals; negative means reduced regret. (f) Six-match pilot: L1 reports q, L2 adds b, L3 adds a nested belief; memory retains own outputs. Depth is not opponent ToM level. Cells show MSE minus constant without MSE uncertainty intervals. Panels are distinct experiments, not a matched rules/no-rules ablation. Coverage is retry-assisted; fixed-history decisions are not live performance.")

with (OUT/'plotted_values.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
(OUT/'captions.json').write_text(json.dumps(dict(figures),indent=2),encoding='utf-8')
from build_supplements import build as build_supplements
supplements=build_supplements()
proof_items=[(name,str(i),cap) for i,(name,cap) in enumerate(figures,2)]+supplements
doc=fitz.open()
for name,label,cap in proof_items:
    src=fitz.open(PDF/f'{name}.pdf');page=doc.new_page(width=612,height=792);h=src[0].rect.height
    setting='Live interaction' if label in ('2','S1','S2','S3') else 'Fixed-history evaluation'
    page.insert_text((108,30),f'Figure {label} | {setting} | draft',fontsize=8)
    page.show_pdf_page(fitz.Rect(108,45,504,45+h),src,0)
    room=page.insert_textbox(fitz.Rect(108,57+h,504,770),f'Figure {label}. '+cap,fontsize=8.2,fontname='helv',lineheight=1.12)
    assert room>=0,(name,room)
proof=PDF/'iclr_figures_2_4_20260921.pdf';doc.save(proof);doc.close()
check=fitz.open(proof)
for i,page in enumerate(check):
    assert not page.get_images(full=True)
    page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/f'proof_page_{i+2}.png')
(OUT/'qa.json').write_text(json.dumps(dict(figures=checks,proof_pages=len(check),new_model_runs=0,latex_manuscript_compiled=False),indent=2),encoding='utf-8')
tex=['% Insert after conceptual Figure 1. Requires graphicx. Paths relative to repository root.']
for name,cap in figures:
    cap=cap.replace('%',r'\%').replace('_',r'\_')
    tex += [r'\begin{figure}[t]',r'\centering',rf'\includegraphics[width=\linewidth]{{output/pdf/{name}.pdf}}',rf'\caption{{{cap}}}',rf'\label{{fig:{name}}}',r'\end{figure}','']
(OUT/'figures.tex').write_text('\n'.join(tex),encoding='utf-8')
output_files=[proof,*[PDF/f'{n}.pdf' for n,_ in figures],*[PDF/f'{n}.pdf' for n,_,_ in supplements],*FIG.glob('*'),OUT/'supplement_plotted_values.csv',OUT/'supplement_provenance.json',OUT/'supplement_captions.json',OUT/'plotted_values.csv',OUT/'captions.json',OUT/'qa.json',OUT/'figures.tex']
(OUT/'provenance.json').write_text(json.dumps(dict(input_sha256=inputs,output_sha256={str(p.relative_to(ROOT)):sha(p) for p in output_files},builder_sha256=sha(Path(__file__)),style_sha256=sha(OUT/'paper_style.py'),new_model_runs=0),indent=2),encoding='utf-8')
print(json.dumps(dict(proof=str(proof),figures=len(figures),plotted_values=len(values),checks=checks),indent=2))
