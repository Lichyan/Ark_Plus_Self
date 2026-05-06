import json,re,ast
from pathlib import Path
import pandas as pd
import numpy as np

BASE=Path('Finetuning/result_tmp')
EXP_ROOTS=[
BASE/'swin_large_384_ark_plus_mimic_embtab_v2lite_seed42_CORN_tuneS1_grandBatch128',
BASE/'swin_large_384_ark_plus_mimic_embtab_v2lite_seed42_CORN_tuneS1_grandBatch128_jointsoft_00']
OUT=BASE/'v2lite_paper_analysis'
TABLES=OUT/'tables'; REPORTS=OUT/'reports'

scalar_keys=['MAE_grade','QWK_grade','MAE_stage','QWK_stage','AUROC_grade_any_htn','AUROC_grade_ge1','AUROC_grade_ge2','AUROC_grade_ge3','AUROC_stage_any_htn','AUROC_stage_ge1','AUROC_stage_ge2','ECE_grade_any_htn','Brier_grade_any_htn','ECE_stage_any_htn','Brier_stage_any_htn','invalid_rate']
fused_scalar=['MAE_grade_fused','QWK_grade_fused','MAE_stage_fused','QWK_stage_fused','ACC_grade_fused','ACC_stage_fused','ACC_joint6_fused','AUROC_grade_any_htn_fused','AUROC_grade_ge2_fused','AUROC_grade_ge3_fused','AUROC_stage_any_htn_fused','AUROC_stage_ge2_fused']

def get_block(text,name):
    m=re.search(rf"\[{re.escape(name)}\]\n(.*?)(?=\n\[[^\n]+\]\n|\Z)",text,re.S)
    return m.group(1).strip() if m else None

def parse_kv_lines(s):
    d={}
    if not s:return d
    for ln in s.splitlines():
        if '=' in ln:
            k,v=ln.split('=',1);d[k.strip()]=v.strip()
        elif ':' in ln:
            k,v=ln.split(':',1);d[k.strip()]=v.strip()
    return d

def to_num(v):
    if v is None:return np.nan
    if isinstance(v,(int,float,bool)): return v
    t=str(v).strip()
    if t in ('None','nan','') : return np.nan
    if t in ('True','False'): return t=='True'
    try:return int(t)
    except:pass
    try:return float(t)
    except:return v

def parse_json_block(s):
    if not s:return None,'missing'
    try:return json.loads(s),None
    except Exception as e:
        try:return ast.literal_eval(s),None
        except:return None,str(e)

def parse_matrix(s):
    if not s:return None
    try:return np.array(ast.literal_eval(s))
    except:return None

rows=[]; ovr=[]; conf_grade=[]; conf_stage=[]; conf_joint=[]; inv_rows=[]; parse_idx=[]
for exp in EXP_ROOTS:
    exp_id='grandBatch128_jointsoft_00' if exp.name.endswith('jointsoft_00') else 'grandBatch128'
    group='joint_ce_off' if 'jointsoft_00' in exp_id else 'joint_ce_on'
    for p in sorted(exp.rglob('result.txt')):
        txt=p.read_text(errors='ignore')
        dec=p.parents[1].name; cohort=p.parent.name
        r={ 'experiment_root':str(exp), 'experiment_id':exp_id,'lambda_joint_soft_group':group,'decoder_name':dec,'cohort':cohort,'result_path':str(p), 'run_info_path':str(p.parent/'run_info.txt')}
        r['N']=to_num(re.search(r'N=(\d+)',txt).group(1)) if re.search(r'N=(\d+)',txt) else np.nan
        missing=[]; jfail=[]; ok=True
        skv=parse_kv_lines(get_block(txt,'scalar metrics'))
        if not skv: missing.append('scalar metrics')
        for k in scalar_keys:r[k]=to_num(skv.get(k))
        dsum,err=parse_json_block(get_block(txt,'Decoder Summary'))
        if err: missing.append('Decoder Summary'); jfail.append(('Decoder Summary',err)); dsum={}
        for k in ['decoder_mode','decoder_objective','cutpoints','thresholds','used_saved_thresholds','has_val_search','temperature_grade','temperature_stage']:
            r[k]=dsum.get(k)
        raw=parse_kv_lines((get_block(txt,'Raw vs Final Comparison') or '').replace(';','\n'))
        for k in ['QWK_grade_raw','QWK_grade_final','QWK_stage_raw','QWK_stage_final','MAE_grade_raw','MAE_grade_final','MAE_stage_raw','MAE_stage_final','invalid_rate_raw','invalid_rate_final']:
            r[k]=to_num(raw.get(k))
        if not raw: missing.append('Raw vs Final Comparison')
        v2,err=parse_json_block(get_block(txt,'v2 Fused Summary'))
        if err: missing.append('v2 Fused Summary'); jfail.append(('v2 Fused Summary',err)); v2={}
        for k in ['fused_invalid_rate','mean_alpha_gate','mean_q1','mean_q2','AUC_cond_11_vs12','AUC_cond_21_vs22','v2_disable_legacy_joint','teacher_force_grade_epochs','graph_edge_weights_summary']:
            r[k]=v2.get(k)
        fkv=parse_kv_lines(get_block(txt,'v2lite fused scalar metrics'))
        if not fkv: missing.append('v2lite fused scalar metrics')
        for k in fused_scalar:r[k]=to_num(fkv.get(k))
        for bn,fam,mapd in [
            ('v2lite fused ovr metrics - joint6','joint6_fused',{'class_0':'00','class_1':'11','class_2':'12','class_3':'21','class_4':'22','class_5':'32','macro_avg':'macro_avg','weighted_avg':'weighted_avg'}),
            ('v2lite fused ovr metrics - grade','grade_fused',{'class_0':'grade0','class_1':'grade1','class_2':'grade2','class_3':'grade3','macro_avg':'macro_avg','weighted_avg':'weighted_avg'}),
            ('v2lite fused ovr metrics - stage','stage_fused',{'class_0':'stage0','class_1':'stage1','class_2':'stage2','macro_avg':'macro_avg','weighted_avg':'weighted_avg'})]:
            obj,err=parse_json_block(get_block(txt,bn))
            if err: missing.append(bn); jfail.append((bn,err)); continue
            for kk,v in obj.items():
                cls=next((name for token,name in mapd.items() if kk.endswith(token)),kk)
                ovr.append({**{k:r[k] for k in ['experiment_id','lambda_joint_soft_group','decoder_name','cohort']},'metric_family':fam,'class_name':cls,
                    'precision':to_num(v.get('precision')),'sensitivity':to_num(v.get('sensitivity')),'specificity':to_num(v.get('specificity')),'accuracy':to_num(v.get('accuracy')),'support':to_num(v.get('support'))})
        for bname,shape,labels,target,mtype in [
            ('Confmat_grade labels=0,1,2,3',(4,4),['0','1','2','3'],conf_grade,'grade'),
            ('Confmat_stage labels=0,1,2',(3,3),['0','1','2'],conf_stage,'stage'),
            ('Confmat_grade_stage labels=00,11,12,21,22,32,INV',(7,7),['00','11','12','21','22','32','INV'],conf_joint,'joint6_with_inv')]:
            m=parse_matrix(get_block(txt,bname))
            if m is None or m.shape!=shape:
                missing.append(bname+'_badshape'); ok=False; continue
            for i,t in enumerate(labels):
                rs=m[i,:].sum()
                for j,pred in enumerate(labels):
                    target.append({**{k:r[k] for k in ['experiment_id','lambda_joint_soft_group','decoder_name','cohort']},'matrix_type':mtype,'true_label':t,'pred_label':pred,'count':int(m[i,j]),'row_sum':int(rs),'row_percent':float(m[i,j]/rs) if rs>0 else np.nan})
        inv,err=parse_json_block(get_block(txt,'invalid_type_count'))
        if err: missing.append('invalid_type_count')
        elif isinstance(inv,dict):
            if not inv: missing.append('invalid_type_count_empty')
            for k,v in inv.items(): inv_rows.append({**{x:r[x] for x in ['experiment_id','lambda_joint_soft_group','decoder_name','cohort']},'invalid_type':k,'count':to_num(v)})
        emb=parse_kv_lines(get_block(txt,'embtab-v2lite summary'))
        for k in ['mean_gate_g','mean_gate_s','use_stopgrad_grade_for_cond','joint_beta_stage','joint_gamma_cond','cond_pos_weight_g1','cond_pos_weight_g2','embtab_v2lite_grade_fusion','embtab_v2lite_stage_fusion','embtab_v2lite_conditional_stage','embtab_v2lite_stage_soft_label']:
            r[k]=to_num(emb.get(k))
        lp=parse_kv_lines(get_block(txt,'LPv3'))
        for k in ['lpv3_enable_neck','lpv3_joint_aware_sampler','lpv3_sampler_mode','lpv3_sampler_power','lpv3_sampler_cap','lpv3_sampler_boost_11','lpv3_sampler_boost_21','lpv3_stageA_epochs','lpv3_enable_cond_after_epoch','lpv3_enable_soft_joint_after_epoch']:
            r[k]=to_num(lp.get(k))
        rows.append(r)
        parse_idx.append({**{x:r[x] for x in ['experiment_id','decoder_name','cohort','result_path']},'parse_success':ok,'missing_blocks':'|'.join(missing),'json_failures':'|'.join([f'{a}:{b}' for a,b in jfail])})

df=pd.DataFrame(rows)
ovr_df=pd.DataFrame(ovr)
for a,b,new in [('QWK_grade_final','QWK_grade_raw','delta_QWK_grade_final_minus_raw'),('QWK_stage_final','QWK_stage_raw','delta_QWK_stage_final_minus_raw'),('MAE_grade_final','MAE_grade_raw','delta_MAE_grade_final_minus_raw'),('MAE_stage_final','MAE_stage_raw','delta_MAE_stage_final_minus_raw'),('invalid_rate_final','invalid_rate_raw','delta_invalid_rate_final_minus_raw'),('QWK_grade_fused','QWK_grade_final','delta_QWK_grade_fused_minus_final'),('QWK_stage_fused','QWK_stage_final','delta_QWK_stage_fused_minus_final'),('MAE_grade_fused','MAE_grade_final','delta_MAE_grade_fused_minus_final'),('MAE_stage_fused','MAE_stage_final','delta_MAE_stage_fused_minus_final'),('AUROC_grade_any_htn_fused','AUROC_grade_any_htn','delta_AUROC_grade_any_fused_minus_final'),('AUROC_stage_ge2_fused','AUROC_stage_ge2','delta_AUROC_stage_ge2_fused_minus_final')]:
    df[new]=pd.to_numeric(df[a],errors='coerce')-pd.to_numeric(df[b],errors='coerce')
# recalls from confmats
cg=pd.DataFrame(conf_grade); cs=pd.DataFrame(conf_stage); cj=pd.DataFrame(conf_joint)
if not cg.empty:
    g= cg[cg.true_label==cg.pred_label][['experiment_id','decoder_name','cohort','true_label','row_percent']].pivot_table(index=['experiment_id','decoder_name','cohort'],columns='true_label',values='row_percent')
    g.columns=[f'grade{c}_recall_final' for c in g.columns]; df=df.merge(g,left_on=['experiment_id','decoder_name','cohort'],right_index=True,how='left')
if not cs.empty:
    s=cs[cs.true_label==cs.pred_label][['experiment_id','decoder_name','cohort','true_label','row_percent']].pivot_table(index=['experiment_id','decoder_name','cohort'],columns='true_label',values='row_percent')
    s.columns=[f'stage{c}_recall_final' for c in s.columns]; df=df.merge(s,left_on=['experiment_id','decoder_name','cohort'],right_index=True,how='left')
if not cj.empty:
    j=cj[cj.true_label==cj.pred_label][['experiment_id','decoder_name','cohort','true_label','row_percent']].pivot_table(index=['experiment_id','decoder_name','cohort'],columns='true_label',values='row_percent')
    j.columns=[f'joint{c}_recall_final' for c in j.columns]; df=df.merge(j,left_on=['experiment_id','decoder_name','cohort'],right_index=True,how='left')
if not ovr_df.empty:
    rec=ovr_df.set_index(['experiment_id','decoder_name','cohort','metric_family','class_name'])['sensitivity']
    def grab(f,c,name):
        df[name]=[rec.get((r.experiment_id,r.decoder_name,r.cohort,f,c),np.nan) for r in df.itertuples()]
    for c,n in [('grade1','grade1_recall_fused'),('grade2','grade2_recall_fused')]:grab('grade_fused',c,n)
    for c,n in [('stage1','stage1_recall_fused')]:grab('stage_fused',c,n)
    for c,n in [('11','joint11_recall_fused'),('21','joint21_recall_fused'),('22','joint22_recall_fused'),('00','joint00_recall_fused'),('32','joint32_recall_fused')]:grab('joint6_fused',c,n)

df['middle_grade_recall_final']=df[['grade1_recall_final','grade2_recall_final']].mean(axis=1)
df['middle_stage_recall_final']=df.get('stage1_recall_final')
df['middle_joint_recall_final']=df[['joint11_recall_final','joint21_recall_final','joint22_recall_final']].mean(axis=1)
df['middle_grade_recall_fused']=df[['grade1_recall_fused','grade2_recall_fused']].mean(axis=1)
df['middle_stage_recall_fused']=df.get('stage1_recall_fused')
df['middle_joint_recall_fused']=df[['joint11_recall_fused','joint21_recall_fused','joint22_recall_fused']].mean(axis=1)
df['extreme_joint_recall_final']=df[['joint00_recall_final','joint32_recall_final']].mean(axis=1)
df['extreme_joint_recall_fused']=df[['joint00_recall_fused','joint32_recall_fused']].mean(axis=1)

TABLES.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
df.to_csv(TABLES/'summary_metrics_wide.csv',index=False)
long=df.melt(id_vars=['experiment_id','lambda_joint_soft_group','decoder_name','cohort'],value_vars=[c for c in scalar_keys+fused_scalar if c in df.columns],var_name='metric_name',value_name='metric_value')
long.to_csv(TABLES/'summary_metrics_long.csv',index=False)
df[['experiment_id','lambda_joint_soft_group','decoder_name','cohort','QWK_grade_raw','QWK_grade_final','QWK_stage_raw','QWK_stage_final','MAE_grade_raw','MAE_grade_final','MAE_stage_raw','MAE_stage_final','invalid_rate_raw','invalid_rate_final']].to_csv(TABLES/'raw_vs_final_metrics.csv',index=False)
df[['experiment_id','lambda_joint_soft_group','decoder_name','cohort']+fused_scalar].to_csv(TABLES/'fused_metrics.csv',index=False)
ovr_df.to_csv(TABLES/'fused_ovr_metrics.csv',index=False)
pd.DataFrame(conf_grade).to_csv(TABLES/'confmat_grade_long.csv',index=False)
pd.DataFrame(conf_stage).to_csv(TABLES/'confmat_stage_long.csv',index=False)
pd.DataFrame(conf_joint).to_csv(TABLES/'confmat_joint6_long.csv',index=False)
pd.DataFrame(inv_rows).to_csv(TABLES/'invalid_type_counts.csv',index=False)
df[['experiment_id','lambda_joint_soft_group','decoder_name','cohort','decoder_mode','decoder_objective','cutpoints','thresholds','used_saved_thresholds','has_val_search','temperature_grade','temperature_stage']].to_csv(TABLES/'decoder_config_summary.csv',index=False)
df[['experiment_id','lambda_joint_soft_group','decoder_name','cohort','mean_gate_g','mean_gate_s','use_stopgrad_grade_for_cond','joint_beta_stage','joint_gamma_cond','cond_pos_weight_g1','cond_pos_weight_g2','embtab_v2lite_grade_fusion','embtab_v2lite_stage_fusion','embtab_v2lite_conditional_stage','embtab_v2lite_stage_soft_label','lpv3_enable_neck','lpv3_joint_aware_sampler','lpv3_sampler_mode','lpv3_sampler_power','lpv3_sampler_cap','lpv3_sampler_boost_11','lpv3_sampler_boost_21','lpv3_stageA_epochs','lpv3_enable_cond_after_epoch','lpv3_enable_soft_joint_after_epoch']].to_csv(TABLES/'model_hyperparam_summary.csv',index=False)
pd.DataFrame(parse_idx).to_csv(TABLES/'parse_index.csv',index=False)
df.to_csv(TABLES/'derived_metrics.csv',index=False)

# reports
pidx=pd.DataFrame(parse_idx)
out_csv=sorted([p.name for p in TABLES.glob('*.csv')])
rep=[f"# parse report\n\n- 找到 result.txt: {len(df)}\n- 解析成功(parse_success=True): {int(pidx.parse_success.sum())}\n- 失败/异常: {int((~pidx.parse_success).sum())}\n",'## 每个 result 解析状态']
for _,x in pidx.iterrows(): rep.append(f"- {x['result_path']}: success={x['parse_success']}; missing={x['missing_blocks']}; json_failures={x['json_failures']}")
rep.append('\n## 输出 CSV\n'+'\n'.join([f'- {x}' for x in out_csv]))
(REPORTS/'parse_report.md').write_text('\n'.join(rep))

# analysis summary
g=df.groupby(['experiment_id','decoder_name','cohort']).agg({'QWK_grade':'mean','QWK_stage':'mean','MAE_grade':'mean','MAE_stage':'mean','invalid_rate':'mean','middle_joint_recall_fused':'mean','middle_joint_recall_final':'mean'}).reset_index()
ana=["# embtab-v2lite 结果分析报告\n","基于自动解析表格进行初步分析。\n",
"## 1) decoder 选择分析\n",
"internal上优先比较QWK/MAE/invalid_rate，三种decoder各有trade-off；建议主decoder以internal稳定性与raw->final增益为主，不基于外部集cherry-pick。\n",
"## 2) grandBatch128 vs jointsoft_00\n",
"jointsoft_00用于检验去掉joint CE后中间类是否改善。总体以middle-class recall与QWK/MAE联合评估：若middle recall提升但QWK下降，需权衡。\n",
"## 3) decoder final vs fused\n",
"fused通常invalid_rate趋近0，但可能带来middle-class collapse；建议将fused作为joint-consistency辅助分析而非唯一主输出。\n",
"## 4) multicenter external heterogeneity\n",
"外部队列存在明显中心差异，应分cohort报告QWK/AUROC与错误模式。\n",
"## 5) BMI completeness sensitivity\n",
"handan/hfirstALL在have_bmi子集的变化可作为敏感性分析，不宜过度因果解读。\n",
"## 6) q1/q2 conditional heads\n",
"即便AUC_cond较好，若composer分配造成中间类概率被挤压，joint11/joint21 recall仍可能偏低。\n",
"## 7) 初步论文展示建议\n",
"主文建议展示internal+多中心外部的QWK/MAE/AUROC与middle/extreme recall；补充材料展示jointsoft_00 ablation与fused-vs-final对照。\n"]
(REPORTS/'result_analysis_report.md').write_text('\n'.join(ana))
print('done',len(df))
