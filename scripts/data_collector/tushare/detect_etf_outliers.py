import pandas as pd
import numpy as np
from pathlib import Path

def detect_outliers(stats_csv, output_dir=None):
    """Detect outliers in ETF basic statistics and suggest cleaning rules."""
    df = pd.read_csv(stats_csv)
    if output_dir is None:
        output_dir = Path(stats_csv).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} ETFs from {stats_csv}\n")

    # 1. 最新收盘价异常高：last_close > 5（通常是分级基金B/杠杆产品）
    high_last = df[df['last_close'] > 5].copy()
    print(f"=== 1. 最新收盘价异常高（last_close > 5）: {len(high_last)} 只 ===")
    if len(high_last) > 0:
        print(high_last[['symbol', 'exchange', 'trading_days', 'last_close', 'total_volume', 'data_start_date']]
              .sort_values('last_close', ascending=False).head(20))
    print()

    # 2. 最新收盘价异常低：last_close < 0.2（可能下折/退市风险）
    low_last = df[df['last_close'] < 0.2].copy()
    print(f"=== 2. 最新收盘价异常低（last_close < 0.2）: {len(low_last)} 只 ===")
    if len(low_last) > 0:
        print(low_last[['symbol', 'exchange', 'trading_days', 'last_close', 'total_volume', 'data_start_date']]
              .sort_values('last_close').head(20))
    print()

    # 3. 历史最低价接近 0：min_close < 0.05
    low_min = df[df['min_close'] < 0.05].copy()
    print(f"=== 3. 历史最低价接近 0（min_close < 0.05）: {len(low_min)} 只 ===")
    if len(low_min) > 0:
        print(low_min[['symbol', 'exchange', 'trading_days', 'min_close', 'max_close', 'last_close']]
              .sort_values('min_close').head(20))
    print()

    # 4. 价格振幅过大：max_close / min_close > 10
    df['price_range_ratio'] = df['max_close'] / df['min_close'].replace(0, np.nan)
    huge_range = df[df['price_range_ratio'] > 10].copy()
    print(f"=== 4. 价格振幅过大（max/min > 10）: {len(huge_range)} 只 ===")
    if len(huge_range) > 0:
        print(huge_range[['symbol', 'exchange', 'trading_days', 'min_close', 'max_close', 'price_range_ratio', 'last_close']]
              .sort_values('price_range_ratio', ascending=False).head(20))
    print()

    # 5. 交易天数过少：trading_days < 60（样本不足）
    few_days = df[df['trading_days'] < 60].copy()
    print(f"=== 5. 交易天数过少（< 60 天）: {len(few_days)} 只 ===")
    if len(few_days) > 0:
        print(few_days[['symbol', 'exchange', 'list_date', 'delist_date', 'trading_days', 'last_close']]
              .sort_values('trading_days').head(20))
    print()

    # 6. 成交量异常低：mean_volume < 1000（流动性极差或数据问题）
    low_volume = df[df['mean_volume'] < 1000].copy()
    print(f"=== 6. 日均成交量异常低（< 1000）: {len(low_volume)} 只 ===")
    if len(low_volume) > 0:
        print(low_volume[['symbol', 'exchange', 'trading_days', 'mean_volume', 'total_volume', 'last_close']]
              .sort_values('mean_volume').head(20))
    print()

    # 7. 单日成交量异常高：max_volume > 1e8（可能是巨量换购或异常交易）
    high_one_day = df[df['max_volume'] > 1e8].copy()
    print(f"=== 7. 单日成交量异常高（> 1 亿）: {len(high_one_day)} 只 ===")
    if len(high_one_day) > 0:
        print(high_one_day[['symbol', 'exchange', 'trading_days', 'max_volume', 'mean_volume', 'last_close']]
              .sort_values('max_volume', ascending=False).head(20))
    print()

    # 8. 数据起始日期异常晚：data_start_date 在 2022 年之后（上市太晚，历史短）
    df['data_start_date'] = pd.to_datetime(df['data_start_date'])
    late_start = df[df['data_start_date'] >= '2022-01-01'].copy()
    print(f"=== 8. 数据起始日期较晚（2022年后上市）: {len(late_start)} 只 ===")
    if len(late_start) > 0:
        print(late_start[['symbol', 'exchange', 'data_start_date', 'trading_days', 'last_close', 'total_volume']]
              .sort_values('data_start_date', ascending=False).head(20))
    print()

    # 汇总建议剔除的 ETF
    exclude_symbols = set()
    # 规则 1：最新收盘价异常低，可能即将退市或数据异常
    exclude_symbols.update(low_last['symbol'].tolist())
    # 规则 2：历史最低价接近 0，风险极高
    exclude_symbols.update(low_min['symbol'].tolist())
    # 规则 3：交易天数过少，样本不足
    exclude_symbols.update(few_days['symbol'].tolist())
    # 规则 4：流动性极差
    exclude_symbols.update(low_volume['symbol'].tolist())

    print(f"=== 建议剔除/单独处理的 ETF 总数: {len(exclude_symbols)} 只 ===")
    print(f"  - 最新收盘价过低（< 0.2）: {len(low_last)}")
    print(f"  - 历史最低价接近 0（< 0.05）: {len(low_min)}")
    print(f"  - 交易天数过少（< 60）: {len(few_days)}")
    print(f"  - 日均成交量过低（< 1000）: {len(low_volume)}")
    print(f"\n剩余可用 ETF 数量: {len(df) - len(exclude_symbols)} / {len(df)}")

    # 保存剔除清单
    exclude_df = df[df['symbol'].isin(exclude_symbols)].copy()
    exclude_df['exclude_reason'] = ''
    for sym in exclude_symbols:
        reasons = []
        if sym in low_last['symbol'].values:
            reasons.append('last_close<0.2')
        if sym in low_min['symbol'].values:
            reasons.append('min_close<0.05')
        if sym in few_days['symbol'].values:
            reasons.append('trading_days<60')
        if sym in low_volume['symbol'].values:
            reasons.append('mean_volume<1000')
        exclude_df.loc[exclude_df['symbol'] == sym, 'exclude_reason'] = ';'.join(reasons)

    exclude_path = output_dir / 'etf_exclude_list.csv'
    exclude_df.to_csv(exclude_path, index=False)
    print(f"\n剔除清单已保存: {exclude_path}")

    # 保存清洗后可用清单
    usable_df = df[~df['symbol'].isin(exclude_symbols)].copy()
    usable_path = output_dir / 'etf_usable_list.csv'
    usable_df.to_csv(usable_path, index=False)
    print(f"可用清单已保存: {usable_path}")

    return exclude_df, usable_df

if __name__ == '__main__':
    stats_csv = r'C:\Users\Administrator\.qlib\qlib_data\etf_data\etf_basic_stats.csv'
    detect_outliers(stats_csv)