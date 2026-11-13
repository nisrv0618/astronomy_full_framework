from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
root=Path(__file__).resolve().parent/'astronomy_full_outputs'
template=root/'human_review_template.csv'; dest=root/'human_decisions.csv'
if not template.exists(): raise SystemExit('Run astronomy_pipeline.py first; human_review_template.csv missing.')
df=pd.read_csv(template)
if 'review_decision' not in df: df['review_decision']='PENDING'
if 'reviewer_notes' not in df: df['reviewer_notes']=''
allowed={'PENDING','RELEVANT','NOISE','UNCERTAIN','REVISIT'}
df['review_decision']=df.review_decision.fillna('PENDING').astype(str).str.strip().str.upper()
bad=sorted(set(df.review_decision)-allowed)
if bad: raise SystemExit(f'Invalid decisions {bad}. Allowed: {sorted(allowed)}')
df['decision_imported_utc']=datetime.now(timezone.utc).isoformat()
if dest.exists():
    old=pd.read_csv(dest)
    # retain history of previous decisions before writing the latest state
    archive=root/'human_decisions_archive.csv'
    old.to_csv(archive,index=False)
df.to_csv(dest,index=False)
print(f'Validated {len(df):,} review rows. Decisions saved to {dest}')
print(df.review_decision.value_counts(dropna=False).to_string())
