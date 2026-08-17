import sys, time, gc
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))

from scripts.dump_bin import DumpDataAll

FIELDS = 'open,close,high,low,volume,adjclose,change,factor,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv'
QLIB_DIR = r'C:\Users\pp\.qlib\qlib_data\cn_data_fwd'
DATA_PATH = r'E:\kaggle_code\qlib\scripts\data_collector\tushare\normalize_fwd'

# Remove old data
import shutil, os
for sub in ['features', 'calendars', 'instruments']:
    p = os.path.join(QLIB_DIR, sub)
    if os.path.exists(p):
        shutil.rmtree(p)
        print(f'Removed {sub}')

t0 = time.time()
dumper = DumpDataAll(
    data_path=DATA_PATH,
    qlib_dir=QLIB_DIR,
    freq='day',
    max_workers=1,
    include_fields=FIELDS,
)
dumper.dump()
print(f'Total: {time.time()-t0:.0f}s')
