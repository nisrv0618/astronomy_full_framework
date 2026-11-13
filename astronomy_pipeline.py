"""Full-dataset astronomy triage framework (batch inference, traceable outputs).

Place processed.parquet beside this file. Run: python astronomy_pipeline.py
This is a research prototype on simulated astronomy data, not a production Rubin broker.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, sys, warnings, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'processed.parquet'
OUT=ROOT/'astronomy_full_outputs'
FIG=OUT/'figures'
BATCH=10_000
TRAIN_CAP=50_000
SEED=42
MISSING_LIMIT=.30
REVIEW_FRACTION=.25

PALETTE={'navy':'#253858','blue':'#3977C5','cyan':'#2CB1BC','teal':'#168C80','green':'#48A868','lime':'#A8C256','yellow':'#F2C14E','orange':'#F28E2B','coral':'#E45756','pink':'#D45087','purple':'#845EC2','violet':'#6C63FF','gray':'#8795A1','light':'#F3F6FA'}

def require():
    needed=[]
    for m,p in [('pyarrow','pyarrow'),('sklearn','scikit-learn'),('matplotlib','matplotlib'),('seaborn','seaborn')]:
        try: __import__(m)
        except ImportError: needed.append(p)
    if needed:
        print('Install missing packages: python -m pip install '+' '.join(needed)); sys.exit(1)

def main():
    require()
    import pyarrow.parquet as pq
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    if not DATA.exists():
        print(f'MISSING DATASET: {DATA}\nCopy processed.parquet beside astronomy_pipeline.py.'); sys.exit(2)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#FBFCFF','axes.titleweight':'bold','axes.titlesize':14,'axes.labelsize':11,'font.size':10,'savefig.dpi':220,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
    pf=pq.ParquetFile(DATA); total=int(pf.metadata.num_rows); columns=pf.schema.names
    features=[c for c in columns if c.startswith('feature_')]
    if not features: raise ValueError('No feature_* columns found. Inspect dataset schema.')
    if 'class' not in columns: raise ValueError("Dataset must contain 'class' for retrospective evaluation.")
    usecols=list(dict.fromkeys(['class','split']+[c for c in ['timestamps','mag','magerr'] if c in columns]+features))
    print(f'Source rows: {total:,}; feature columns found: {len(features)}')
    print('Pass 1/3: profiling all rows and collecting a training sample...')
    rng=np.random.default_rng(SEED); samples=[]; seen=0; batches=0; class_counts={}; missing_counts={c:0 for c in features}; nonfinite_counts={c:0 for c in features}; duplicate_hashes=set(); duplicate_count=0; missing_lightcurve=0
    # Reservoir-like per-batch sampling; entire dataset is scanned.
    per_batch=max(1,int(np.ceil(TRAIN_CAP/max(1,total/BATCH))))
    for batch in pf.iter_batches(batch_size=BATCH,columns=usecols):
        d=batch.to_pandas(); batches+=1; seen+=len(d)
        vc=d['class'].astype('string').fillna('__MISSING_CLASS__').value_counts()
        for k,v in vc.items(): class_counts[str(k)]=class_counts.get(str(k),0)+int(v)
        x=d[features].apply(pd.to_numeric,errors='coerce')
        for c in features:
            missing_counts[c]+=int(x[c].isna().sum())
            nonfinite_counts[c]+=int(np.isinf(x[c].to_numpy(dtype=float,na_value=np.nan)).sum())
        if 'mag' in d.columns:
            missing_lightcurve+=int(d['mag'].isna().sum())
        # duplicate feature vectors tracked by stable hashes; exact duplicates only
        arr=x.replace([np.inf,-np.inf],np.nan).round(8)
        hashes=pd.util.hash_pandas_object(arr, index=False).to_numpy(dtype=np.uint64)
        for h in hashes:
            ih=int(h)
            if ih in duplicate_hashes: duplicate_count+=1
            else: duplicate_hashes.add(ih)
        if len(d)>per_batch:
            take=rng.choice(len(d),size=per_batch,replace=False); d=d.iloc[take]
        samples.append(d[['class']+(['split'] if 'split' in d.columns else [])+features].copy())
        print(f'  scanned {seen:,}/{total:,} rows',end='\r')
    print(f'\nFull scan complete: {seen:,} rows in {batches} batches.')
    if seen!=total: raise RuntimeError(f'Row-accounting mismatch: metadata={total}, scanned={seen}')
    train_df=pd.concat(samples,ignore_index=True).drop_duplicates().reset_index(drop=True)
    # keep classes with enough sample examples; hold out most frequent eligible class for controlled OOD simulation
    train_df['class']=train_df['class'].astype('string').fillna('__MISSING_CLASS__').astype(str)
    sample_counts=train_df['class'].value_counts(); eligible=sample_counts[sample_counts>=30]
    if len(eligible)<2: raise ValueError('Insufficient classes in training sample. Increase TRAIN_CAP or inspect dataset.')
    heldout=str(eligible.idxmax())
    usable=set(eligible.index.tolist()); known_sample=train_df[(train_df['class']!=heldout)&train_df['class'].isin(usable)].copy()
    if len(known_sample)<100: raise ValueError('Not enough known-class training examples.')
    Xtrain=known_sample[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    # Remove all-null features based on training sample only
    feature_cols=[c for c in features if Xtrain[c].notna().any()]
    Xtrain=Xtrain[feature_cols]
    imputer=SimpleImputer(strategy='median'); scaler=StandardScaler()
    Xt=scaler.fit_transform(imputer.fit_transform(Xtrain))
    print(f'Model fit uses {len(known_sample):,} sampled known-class rows; held-out class for retrospective evaluation: {heldout}')
    ood=IsolationForest(n_estimators=200,contamination='auto',random_state=SEED,n_jobs=-1).fit(Xt)
    train_scores=ood.score_samples(Xt); threshold=float(np.quantile(train_scores,.05))
    clf=RandomForestClassifier(n_estimators=180,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1,min_samples_leaf=2).fit(Xt,known_sample['class'].to_numpy())
    imp=pd.DataFrame({'feature':feature_cols,'importance':clf.feature_importances_}).sort_values('importance',ascending=False)
    imp.to_csv(OUT/'feature_importance.csv',index=False)
    print('Pass 2/3: scoring EVERY source row in batches...')
    result_parts=[]; quality_parts=[]; hist_parts=[]; processed=0; duplicate_seen=set()
    # second scan all rows, including class only for retrospective evaluation, never model input
    for bi,batch in enumerate(pf.iter_batches(batch_size=BATCH,columns=usecols),1):
        d=batch.to_pandas(); n=len(d); processed+=n
        x=d[feature_cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        missing_frac=x.isna().mean(axis=1).to_numpy()
        # detect duplicate feature vectors consistently across batches using hashes
        hashes=pd.util.hash_pandas_object(x.round(8),index=False).to_numpy(dtype=np.uint64)
        dup=np.zeros(n,dtype=bool)
        for j,h in enumerate(hashes):
            ih=int(h)
            if ih in duplicate_seen: dup[j]=True
            else: duplicate_seen.add(ih)
        nonfinite=np.isinf(d[feature_cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)).any(axis=1)
        noise_reason=np.where(missing_frac>MISSING_LIMIT,'EXCESSIVE_MISSINGNESS','')
        noise_reason=np.where((noise_reason=='')&nonfinite,'NONFINITE_FEATURE_VALUE',noise_reason)
        noise_reason=np.where((noise_reason=='')&dup,'DUPLICATE_FEATURE_VECTOR',noise_reason)
        noise_flag=noise_reason!=''
        # score all rows, imputation preserves scoring even for flagged rows; flagged records remain in output/audit
        X=scaler.transform(imputer.transform(x))
        normality=ood.score_samples(X); nov_rank=pd.Series(-normality).rank(pct=True).to_numpy()
        probs=clf.predict_proba(X); entropy=-(probs*np.log(probs+1e-12)).sum(axis=1); unc=np.clip(entropy/np.log(max(2,probs.shape[1])),0,1)
        pred=clf.classes_[np.argmax(probs,axis=1)]
        # high novelty means lower IsolationForest normality; percentile score is relative to full incoming dataset
        nov=nov_rank
        priority=.60*nov+.40*unc
        labels=d['class'].astype('string').fillna('__MISSING_CLASS__').astype(str).to_numpy()
        split=d['split'].astype('string').fillna('UNSPECIFIED').astype(str).to_numpy() if 'split' in d.columns else np.repeat('UNSPECIFIED',n)
        start=processed-n
        ids=np.array([f'ASTRO_{i+1:09d}' for i in range(start,processed)])
        # operational status: quality flags are routed to quality review, otherwise threshold OOD candidates ranked
        ood_flag=normality<threshold
        decision=np.where(noise_flag,'QUALITY_REVIEW',np.where(ood_flag,'OOD_DEFERRED','LOWER_PRIORITY'))
        out=pd.DataFrame({'observation_id':ids,'source_row_index':np.arange(start,processed),'class_for_evaluation_only':labels,'split':split,'missing_feature_fraction':missing_frac,'duplicate_feature_vector':dup,'nonfinite_feature_value':nonfinite,'noise_flag':noise_flag,'noise_reason':noise_reason,'isolation_forest_normality':normality,'ood_threshold':threshold,'ood_flag':ood_flag,'novelty_score':nov,'uncertainty_score':unc,'priority_score':priority,'predicted_known_class':pred,'initial_route':decision})
        result_parts.append(out)
        quality_parts.append(out[['observation_id','source_row_index','missing_feature_fraction','duplicate_feature_vector','nonfinite_feature_value','noise_flag','noise_reason']])
        if bi%5==0 or processed==total: print(f'  scored {processed:,}/{total:,} rows',end='\r')
    print(f'\nScoring complete: {processed:,} rows.')
    results=pd.concat(result_parts,ignore_index=True)
    if len(results)!=total: raise RuntimeError(f'Output row mismatch: {len(results)} outputs for {total} source rows')
    # Ranking and queue: exclude quality-flagged records from automatic review selection; preserve them for human quality review
    clean=results[~results.noise_flag & results.ood_flag].sort_values('priority_score',ascending=False)
    queue_n=int(np.ceil(len(clean)*REVIEW_FRACTION)) if len(clean) else 0
    immediate=set(clean.head(queue_n).observation_id)
    results['route']=np.where(results.noise_flag,'QUALITY_REVIEW',np.where(results.observation_id.isin(immediate),'IMMEDIATE_HUMAN_REVIEW',np.where(results.ood_flag,'DEFERRED_OOD_RETRIEVABLE','LOWER_PRIORITY_RETRIEVABLE')))
    results['priority_rank']=results['priority_score'].rank(method='first',ascending=False).astype(int)
    results['review_status']=np.where(results.route.isin(['IMMEDIATE_HUMAN_REVIEW','QUALITY_REVIEW']),'PENDING','NOT_YET_REVIEWED')
    results['review_decision']='PENDING'; results['reviewer_notes']=''
    results.to_csv(OUT/'all_observation_results.csv.gz',index=False,compression='gzip')
    results[results.route=='IMMEDIATE_HUMAN_REVIEW'].to_csv(OUT/'human_review_queue.csv',index=False)
    results[results.route=='QUALITY_REVIEW'].to_csv(OUT/'noise_assessment_results.csv',index=False)
    results[results.route.str.contains('RETRIEVABLE')].to_csv(OUT/'deferred_observations.csv.gz',index=False,compression='gzip')
    # stage-wise event log for every row
    hist=[]
    for stage,status_col,reason_col,score_col in [
      ('0_INPUT',None,None,None),('1_QUALITY',None,'noise_reason','missing_feature_fraction'),('2_OOD','ood_flag',None,'novelty_score'),('3_UNCERTAINTY',None,None,'uncertainty_score'),('4_PRIORITY',None,None,'priority_score'),('5_ROUTING','route',None,'priority_score')]:
        h=pd.DataFrame({'observation_id':results.observation_id,'source_row_index':results.source_row_index,'stage':stage,'status':'RECEIVED' if stage=='0_INPUT' else 'ASSESSED','score':np.nan if score_col is None else results[score_col],'reason':'' if reason_col is None else results[reason_col].fillna('')})
        if status_col:
            if stage=='2_OOD': h['status']=np.where(results[status_col],'OOD_CANDIDATE','BELOW_THRESHOLD')
            else: h['status']=results[status_col]
        if stage=='1_QUALITY': h['status']=np.where(results.noise_flag,'FLAGGED_FOR_QUALITY_REVIEW','PASSED_RULES')
        hist.append(h)
    history=pd.concat(hist,ignore_index=True); history.to_csv(OUT/'pruning_history.csv.gz',index=False,compression='gzip')
    with open(OUT/'run_metadata.json','w',encoding='utf-8') as f: json.dump({'run_utc':datetime.now(timezone.utc).isoformat(),'source_file':DATA.name,'source_rows_metadata':total,'rows_scanned_pass1':seen,'rows_scored_pass2':processed,'batches_pass1':batches,'batch_size':BATCH,'feature_count_available':len(features),'feature_count_used':len(feature_cols),'training_sample_rows':len(known_sample),'heldout_class_for_evaluation_only':heldout,'ood_threshold':threshold,'model_inference_rows':len(results),'complete_source_accounting':seen==total==processed,'note':'Simulated dataset; training sample used, inference and row-level outputs cover all source rows.'},f,indent=2)
    # evaluation only: held-out class labels retrospective, no label used in operational scoring
    y=(results.class_for_evaluation_only==heldout).astype(int).to_numpy(); metrics={'source_rows':total,'rows_scanned_for_quality_profile':seen,'rows_scored':processed,'row_accounting_passed':seen==total==processed,'batches_scanned':batches,'features_used':len(feature_cols),'training_rows':len(known_sample),'heldout_class_for_evaluation_only':heldout,'quality_flagged_rows':int(results.noise_flag.sum()),'excessive_missingness_rows':int((results.noise_reason=='EXCESSIVE_MISSINGNESS').sum()),'nonfinite_rows':int(results.nonfinite_feature_value.sum()),'duplicate_vector_rows':int(results.duplicate_feature_vector.sum()),'ood_candidates':int(results.ood_flag.sum()),'immediate_review_queue':int((results.route=='IMMEDIATE_HUMAN_REVIEW').sum()),'quality_review_queue':int((results.route=='QUALITY_REVIEW').sum()),'deferred_ood_retrievable':int((results.route=='DEFERRED_OOD_RETRIEVABLE').sum()),'lower_priority_retrievable':int((results.route=='LOWER_PRIORITY_RETRIEVABLE').sum()),'immediate_queue_reduction_percent':float(100*(1-(results.route=='IMMEDIATE_HUMAN_REVIEW').sum()/max(1,total))),'ood_roc_auc_heldout_class':float(roc_auc_score(y,results.novelty_score)) if len(np.unique(y))==2 else None,'ood_average_precision_heldout_class':float(average_precision_score(y,results.novelty_score)) if len(np.unique(y))==2 else None,'top20_unknown_precision':float(results.sort_values('priority_score',ascending=False).head(20).class_for_evaluation_only.eq(heldout).mean()),'human_reviews_completed':0,'human_decisions_fabricated':False}
    with open(OUT/'metrics.json','w',encoding='utf-8') as f: json.dump(metrics,f,indent=2)
    # CSV template for decisions, preserving pending states
    review=results[results.route.isin(['IMMEDIATE_HUMAN_REVIEW','QUALITY_REVIEW'])][['observation_id','source_row_index','route','priority_rank','priority_score','novelty_score','uncertainty_score','noise_reason','review_status','review_decision','reviewer_notes']].copy()
    review.to_csv(OUT/'human_review_template.csv',index=False)
    # figures
    print('Generating research figures...')
    sns.set_theme(style='whitegrid',context='notebook')
    def savefig(name): plt.tight_layout(); plt.savefig(FIG/name,dpi=240); plt.close()
    # 1 class distribution
    fig,ax=plt.subplots(figsize=(12,6)); pd.Series(class_counts).sort_values().plot.barh(ax=ax,color=sns.color_palette('viridis',len(class_counts))); ax.set_title(f'Full dataset class distribution (n={total:,})'); ax.set_xlabel('Observations'); ax.set_ylabel('Ground-truth class (evaluation only)'); savefig('01_class_distribution.png')
    # 2 missingness heatmap top features
    miss=pd.Series({c:missing_counts[c]/max(1,total) for c in features}).sort_values(ascending=False).head(35)
    fig,ax=plt.subplots(figsize=(13,5)); sns.heatmap(miss.to_frame('missing fraction').T,cmap='magma',vmin=0,vmax=max(.01,float(miss.max())),annot=False,ax=ax,cbar_kws={'label':'Fraction missing'}); ax.set_title('Full-dataset missingness across the 35 most affected features'); ax.set_xlabel('Feature'); ax.set_ylabel(''); ax.tick_params(axis='x',rotation=70); savefig('02_missingness_heatmap.png')
    # 3 quality reasons
    q=results.noise_reason.replace('', 'No rule triggered').value_counts()
    fig,ax=plt.subplots(figsize=(10,6)); q.plot.bar(ax=ax,color=[PALETTE['green'],PALETTE['orange'],PALETTE['coral'],PALETTE['purple']][:len(q)]); ax.set_title('Quality-screening outcomes across all observations'); ax.set_ylabel('Observations'); ax.set_xlabel('First applicable quality rule'); ax.tick_params(axis='x',rotation=20); savefig('03_quality_outcomes.png')
    # 4 novelty score distributions
    fig,ax=plt.subplots(figsize=(10,6)); sns.histplot(data=results,x='novelty_score',hue='ood_flag',bins=45,element='step',stat='density',common_norm=False,palette={False:PALETTE['blue'],True:PALETTE['coral']},ax=ax); ax.set_title('Novelty score distribution across the complete dataset'); ax.set_xlabel('Relative novelty percentile score'); savefig('04_novelty_distribution.png')
    # 5 novelty uncertainty hexbin
    plotdf=results.sample(min(100000,len(results)),random_state=SEED)
    fig,ax=plt.subplots(figsize=(10,7)); hb=ax.hexbin(plotdf.novelty_score,plotdf.uncertainty_score,gridsize=45,cmap='turbo',mincnt=1,bins='log'); fig.colorbar(hb,ax=ax,label='Log observation density'); ax.scatter(plotdf.loc[plotdf.route=='IMMEDIATE_HUMAN_REVIEW','novelty_score'],plotdf.loc[plotdf.route=='IMMEDIATE_HUMAN_REVIEW','uncertainty_score'],s=12,c=PALETTE['pink'],alpha=.7,label='Immediate review'); ax.set_title('Novelty × uncertainty landscape (hexbin density; review items overlaid)'); ax.set_xlabel('Novelty score'); ax.set_ylabel('Normalized predictive entropy'); ax.legend(); savefig('05_novelty_uncertainty_landscape.png')
    # 6 funnel
    stages=['Source rows','OOD candidates','Immediate review','Quality review','Deferred OOD','Lower priority']
    vals=[total,int(results.ood_flag.sum()),int((results.route=='IMMEDIATE_HUMAN_REVIEW').sum()),int((results.route=='QUALITY_REVIEW').sum()),int((results.route=='DEFERRED_OOD_RETRIEVABLE').sum()),int((results.route=='LOWER_PRIORITY_RETRIEVABLE').sum())]
    fig,ax=plt.subplots(figsize=(12,6)); bars=ax.bar(stages,vals,color=[PALETTE['navy'],PALETTE['purple'],PALETTE['pink'],PALETTE['orange'],PALETTE['cyan'],PALETTE['blue']]); ax.set_yscale('log'); ax.set_title('Full-dataset triage outcomes (log scale; deferred records retained)'); ax.set_ylabel('Observations (log scale)'); ax.tick_params(axis='x',rotation=20)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()*1.06,f'{v:,}',ha='center',fontsize=9)
    savefig('06_triage_outcomes.png')
    # 7 priority distributions
    fig,ax=plt.subplots(figsize=(11,6)); sns.violinplot(data=results.sample(min(100000,len(results)),random_state=SEED),x='route',y='priority_score',hue='route',palette=[PALETTE['pink'],PALETTE['orange'],PALETTE['purple'],PALETTE['blue']],legend=False,inner='quartile',cut=0,ax=ax); ax.set_title('Priority score by routing outcome'); ax.set_xlabel('Routing outcome'); ax.tick_params(axis='x',rotation=18); savefig('07_priority_by_route.png')
    # 8 human review status
    counts=review.route.value_counts(); fig,ax=plt.subplots(figsize=(8,7)); ax.pie(counts.values,labels=counts.index,autopct='%1.1f%%',startangle=90,colors=[PALETTE['pink'],PALETTE['orange']],wedgeprops={'width':.42,'edgecolor':'white'},textprops={'fontsize':10}); ax.set_title(f'Human review workload (all decisions pending; n={len(review):,})'); savefig('08_human_review_workload.png')
    # 9 feature importance
    top=imp.head(25).sort_values('importance'); fig,ax=plt.subplots(figsize=(11,8)); ax.barh(top.feature,top.importance,color=sns.color_palette('mako',len(top))); ax.set_title('Top 25 Random Forest feature importances'); ax.set_xlabel('Impurity-based importance (not causal attribution)'); savefig('09_feature_importance.png')
    # 10 top20 composition
    top20=results.sort_values('priority_score',ascending=False).head(20); comp=top20.class_for_evaluation_only.eq(heldout).value_counts().reindex([True,False],fill_value=0)
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(['Held-out class','Other classes'],[int(comp[True]),int(comp[False])],color=[PALETTE['purple'],PALETTE['cyan']]); ax.set_title('Top-20 priority observations: retrospective held-out-class composition'); ax.set_ylabel('Observations');
    for i,v in enumerate([int(comp[True]),int(comp[False])]):
        ax.text(i,v+.2,str(v),ha='center')
    savefig('10_top20_unknown_composition.png')
    # 11 route by class stacked normalized using evaluation labels
    ctab=pd.crosstab(results.class_for_evaluation_only,results.route,normalize='index')
    fig,ax=plt.subplots(figsize=(13,7)); ctab.plot(kind='bar',stacked=True,ax=ax,colormap='Spectral'); ax.set_title('Routing outcomes by ground-truth class (retrospective analysis only)'); ax.set_ylabel('Fraction within class'); ax.set_xlabel('Class'); ax.legend(title='Route',bbox_to_anchor=(1.02,1),loc='upper left'); ax.tick_params(axis='x',rotation=45); savefig('11_route_by_class.png')
    # 12 pruning trace Sankey-like flow via stacked stage counts
    fig,ax=plt.subplots(figsize=(11,6)); route_counts=results.route.value_counts(); route_counts=route_counts.reindex(['IMMEDIATE_HUMAN_REVIEW','QUALITY_REVIEW','DEFERRED_OOD_RETRIEVABLE','LOWER_PRIORITY_RETRIEVABLE'],fill_value=0); ax.barh(route_counts.index,route_counts.values,color=[PALETTE['pink'],PALETTE['orange'],PALETTE['cyan'],PALETTE['blue']]); ax.set_title('Auditable terminal routing: every source row assigned a retrievable route'); ax.set_xlabel('Observations');
    for i,v in enumerate(route_counts.values):
        ax.text(v+max(total*.002,1),i,f'{v:,}',va='center')
    savefig('12_retrievable_routes.png')
    # Dashboard 3x3 selected figures using existing PNGs
    from PIL import Image, ImageOps, ImageDraw
    files=['01_class_distribution.png','02_missingness_heatmap.png','03_quality_outcomes.png','04_novelty_distribution.png','05_novelty_uncertainty_landscape.png','06_triage_outcomes.png','08_human_review_workload.png','09_feature_importance.png','12_retrievable_routes.png']
    thumbs=[]
    for fn in files:
        im=Image.open(FIG/fn).convert('RGB'); im.thumbnail((800,460)); canvas=Image.new('RGB',(820,500),'white'); canvas.paste(im,((820-im.width)//2,20)); ImageDraw.Draw(canvas).text((18,475),fn,fill=PALETTE['navy']); thumbs.append(canvas)
    dashboard=Image.new('RGB',(2460,1500),'#EAF0F7')
    for i,im in enumerate(thumbs): dashboard.paste(im,((i%3)*820,(i//3)*500))
    dashboard.save(OUT/'astronomy_dashboard.png',quality=95)
    report=f"""ASTRONOMY FULL-DATASET TRIAGE REPORT\nRun UTC: {datetime.now(timezone.utc).isoformat()}\n\nDATA ACCOUNTING\nSource rows: {total:,}\nRows scanned during profiling: {seen:,}\nRows scored: {processed:,}\nRow accounting passed: {seen==total==processed}\nFeatures used: {len(feature_cols)}\nTraining sample rows (known classes): {len(known_sample):,}\nHeld-out evaluation class: {heldout}\n\nRESULTS\nQuality flags: {metrics['quality_flagged_rows']:,}\nOOD candidates: {metrics['ood_candidates']:,}\nImmediate human-review queue: {metrics['immediate_review_queue']:,}\nQuality-review queue: {metrics['quality_review_queue']:,}\nDeferred OOD retained: {metrics['deferred_ood_retrievable']:,}\nLower-priority retained: {metrics['lower_priority_retrievable']:,}\nTop-20 held-out-class precision: {metrics['top20_unknown_precision']:.4f}\nOOD ROC-AUC (held-out class retrospective evaluation): {metrics['ood_roc_auc_heldout_class']}\n\nINTERPRETATION AND LIMITATIONS\nAll source rows were profiled and scored in batches. The model was fitted on a capped sample of known-class records for memory/runtime practicality. The held-out class is used only for retrospective evaluation, not as an operational input. Isolation Forest scores are not calibrated probabilities. Feature importance is impurity-based and is not causal explanation. The noise rules are conservative data-quality flags, not proof of astrophysical noise; flagged rows are retained for review. No human decisions have been fabricated; review records begin as PENDING. This is simulated-data proof-of-concept, not validation on live Rubin alerts.\n\nFILES\nall_observation_results.csv.gz contains a row-level result for every source record. pruning_history.csv.gz contains stage events for every record. Use retrieve_history.py OBSERVATION_ID to export an observation history. Edit human_review_template.csv and run import_human_decisions.py to validate and store reviewer decisions.\n"""
    (OUT/'results_report.txt').write_text(report,encoding='utf-8')
    print('\nDONE. Full row-level outputs and figures saved to:',OUT)
    print(json.dumps(metrics,indent=2))
    print('Dashboard:',OUT/'astronomy_dashboard.png')

if __name__=='__main__':
    with warnings.catch_warnings(): warnings.simplefilter('ignore'); main()
