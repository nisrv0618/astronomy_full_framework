from pathlib import Path
import sys, pandas as pd
root=Path(__file__).resolve().parent/'astronomy_full_outputs'
if len(sys.argv)<2: print('Usage: python retrieve_history.py ASTRO_000000001'); raise SystemExit(1)
needle=sys.argv[1]
path=root/'pruning_history.csv.gz'
if not path.exists(): print('Run astronomy_pipeline.py first.'); raise SystemExit(2)
found=[]
for chunk in pd.read_csv(path,chunksize=100_000):
    hit=chunk[chunk.observation_id.astype(str)==needle]
    if not hit.empty: found.append(hit)
if not found: print('Observation ID not found:',needle); raise SystemExit(3)
out=pd.concat(found,ignore_index=True); dest=root/f'history_{needle}.csv'; out.to_csv(dest,index=False); print(out.to_string(index=False)); print('\nSaved:',dest)
