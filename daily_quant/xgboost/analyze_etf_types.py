"""Analyze which ETF types the model selects most frequently."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import os
os.environ['NO_PROXY'] = '*'

def main():
    import tushare as ts
    import pandas as pd
    from collections import Counter
    import qlib
    from qlib.constant import REG_CN
    from qlib.workflow import R
    from qlib.data.dataset import DatasetH
    from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

    _CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]
    qlib.init(provider_uri='C:/Users/pp/.qlib/qlib_data/etf_data', region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=1)
    pro = ts.pro_api('a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202')

    fund_df = pro.fund_basic(market='E')
    stock_etf = fund_df[fund_df['fund_type'].str.contains('股票|跨境', na=False, regex=True)]

    def to_qlib(ts_code):
        code, exchange = ts_code.split('.')
        return f'{exchange.lower()}{code}'.upper()

    type_map = {}
    name_map = {}
    for _, row in stock_etf.iterrows():
        sym = to_qlib(row['ts_code'])
        type_map[sym] = row['fund_type']
        name_map[sym] = row['name']

    exp_name = 'XGB_Alpha158_stock'
    recs_dict = R.list_recorders(experiment_name=exp_name)
    recs = [r for r in recs_dict.values() if 'trained_model' in r.list_artifacts()]
    recs.sort(key=lambda r: r.info.get('end_time') or '', reverse=True)
    model = R.get_recorder(recorder_id=recs[0].id, experiment_name=exp_name).load_object('trained_model')

    ds = DatasetH(
        handler={
            'class': 'Alpha158Date',
            'module_path': 'daily_quant.handler.alpha158_date',
            'kwargs': {
                'start_time': '2022-01-01', 'end_time': '2026-06-29',
                'fit_start_time': '2022-01-01', 'fit_end_time': '2024-12-31',
                'instruments': 'stock',
                'use_alpha_factors': False,
                'infer_processors': [
                    {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
                    {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}},
                ],
                'learn_processors': [
                    {'class': 'DropnaLabel'},
                    {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}},
                ],
            },
        },
        segments={'test': ('2025-07-01', '2026-06-29')},
    )

    pred_all = model.predict(ds)
    df = pred_all.reset_index()
    score_col = df.columns[2]

    top_picks = []
    for week, group in df.groupby(pd.Grouper(key='datetime', freq='W')):
        if group.empty:
            continue
        top10 = group.nlargest(10, score_col)
        for _, row in top10.iterrows():
            top_picks.append(row['instrument'])

    print(f'Top picks collected: {len(top_picks)}')
    print(f'First 5: {top_picks[:5]}')

    type_counts = Counter()
    name_counts = Counter()
    for sym in top_picks:
        t = type_map.get(sym.upper(), 'UNKNOWN_TYPE')
        n = name_map.get(sym.upper(), 'UNKNOWN_NAME')
        type_counts[t] += 1
        name_counts[n] += 1

    print('\n=== Most Picked ETF Names (Top 15) ===')
    for name, c in name_counts.most_common(15):
        print(f'  {name}: {c}')

    print('\n=== By Fund Type ===')
    total = sum(type_counts.values())
    for t, c in type_counts.most_common():
        print(f'  {t}: {c} ({c/total:.1%})')

if __name__ == '__main__':
    main()
