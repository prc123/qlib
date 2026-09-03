import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

base = Path(r'C:\Users\Administrator\.qlib\qlib_data\etf_data')

def read_qlib_bin(fpath):
    """Read qlib binary file format: [date_index, value1, value2, ...]."""
    data = np.fromfile(fpath, dtype=np.float32)
    if len(data) == 0:
        return 0, np.array([])
    date_idx = int(data[0])
    values = data[1:]
    return date_idx, values
# 读取日历
with open(base / 'calendars' / 'day.txt', 'r') as f:
    dates = [line.strip() for line in f if line.strip()]
dates = pd.to_datetime(dates)
print('=== 日历信息 ===')
print(f'交易日总数: {len(dates)}')
print(f'时间范围: {dates[0].date()} -> {dates[-1].date()}')
print(f'时间跨度: {(dates[-1] - dates[0]).days} 天')

# 读取 instruments
with open(base / 'instruments' / 'all.txt', 'r') as f:
    insts = [line.strip().split('\t') for line in f if line.strip()]
print('\n=== ETF 标的 ===')
print(f'ETF 总数: {len(insts)}')
sh = sum(1 for i in insts if i[0].startswith('SH'))
sz = sum(1 for i in insts if i[0].startswith('SZ'))
print(f'上交所(SH): {sh} ({sh/len(insts)*100:.1f}%)')
print(f'深交所(SZ): {sz} ({sz/len(insts)*100:.1f}%)')

# 上市/退市日期统计
list_dates = pd.to_datetime([i[1] for i in insts])
delist_dates = pd.to_datetime([i[2] for i in insts])
print(f'最早上市: {list_dates.min().date()}')
print(f'最晚上市: {list_dates.max().date()}')
print(f'已退市数量: {(delist_dates < dates[-1]).sum()}')
print(f'仍在交易数量: {(delist_dates >= dates[-1]).sum()}')

# 读取价格和成交量数据并统计
features_dir = base / 'features'

stats = {
    'symbol': [],
    'exchange': [],
    'list_date': [],
    'delist_date': [],
    'data_start_idx': [],
    'data_start_date': [],
    'data_end_idx': [],
    'data_end_date': [],
    'trading_days': [],
    'last_close': [],
    'max_close': [],
    'min_close': [],
    'mean_close': [],
    'total_volume': [],
    'mean_volume': [],
    'max_volume': [],
}

print('\n=== 开始读取价格/成交量数据 ===')
for idx, (symbol, ld, dd) in enumerate(insts):
    sym_dir = features_dir / symbol.lower()
    if not sym_dir.exists():
        continue
    
    start_idx, close = read_qlib_bin(sym_dir / 'close.day.bin')
    _, volume = read_qlib_bin(sym_dir / 'volume.day.bin')
    
    valid_mask = ~np.isnan(close) & (close > 0)
    valid_close = close[valid_mask]
    valid_volume = volume[valid_mask]
    
    stats['symbol'].append(symbol)
    stats['exchange'].append('SH' if symbol.startswith('SH') else 'SZ')
    stats['list_date'].append(ld)
    stats['delist_date'].append(dd)
    stats['data_start_idx'].append(start_idx)
    stats['data_start_date'].append(dates[start_idx].strftime('%Y-%m-%d') if start_idx < len(dates) else '')
    end_idx = start_idx + len(close) - 1
    stats['data_end_idx'].append(end_idx)
    stats['data_end_date'].append(dates[end_idx].strftime('%Y-%m-%d') if end_idx < len(dates) else '')
    stats['trading_days'].append(int(valid_mask.sum()))
    stats['last_close'].append(float(valid_close[-1]) if len(valid_close) > 0 else np.nan)
    stats['max_close'].append(float(valid_close.max()) if len(valid_close) > 0 else np.nan)
    stats['min_close'].append(float(valid_close.min()) if len(valid_close) > 0 else np.nan)
    stats['mean_close'].append(float(valid_close.mean()) if len(valid_close) > 0 else np.nan)
    stats['total_volume'].append(float(valid_volume.sum()) if len(valid_volume) > 0 else np.nan)
    stats['mean_volume'].append(float(valid_volume.mean()) if len(valid_volume) > 0 else np.nan)
    stats['max_volume'].append(float(valid_volume.max()) if len(valid_volume) > 0 else np.nan)
    
    if (idx + 1) % 500 == 0:
        print(f'  已处理 {idx+1}/{len(insts)} ...')

df_stats = pd.DataFrame(stats)
print(f'\n成功读取 {len(df_stats)} 只 ETF 的数据')

print('\n=== 收盘价统计 ===')
print(df_stats[['last_close', 'max_close', 'min_close', 'mean_close']].describe())

print('\n=== 成交量统计 ===')
print(df_stats[['total_volume', 'mean_volume', 'max_volume']].describe())

print('\n=== 交易天数统计 ===')
print(df_stats['trading_days'].describe())

print('\n=== 交易所对比 ===')
print(df_stats.groupby('exchange')[['last_close', 'total_volume', 'trading_days']].mean())

print('\n=== 成交量最大的 10 只 ETF ===')
print(df_stats.nlargest(10, 'total_volume')[['symbol', 'exchange', 'trading_days', 'last_close', 'total_volume']])

print('\n=== 最新收盘价最高的 10 只 ETF ===')
print(df_stats.nlargest(10, 'last_close')[['symbol', 'exchange', 'trading_days', 'last_close', 'total_volume']])

# 保存到 CSV
output_csv = base / 'etf_basic_stats.csv'
df_stats.to_csv(output_csv, index=False)
print(f'\n详细统计已保存: {output_csv}')