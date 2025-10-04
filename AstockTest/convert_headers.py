import pandas as pd
import os

# 定义中英文表头映射
header_mapping = {
    '日期': 'date',
    '股票代码': 'stock_code', 
    '开盘': 'open',
    '收盘': 'close',
    '最高': 'high',
    '最低': 'low',
    '成交量': 'volume',
    '成交额': 'amount',
    '振幅': 'amplitude',
    '涨跌幅': 'change_percent',
    '涨跌额': 'change_amount',
    '换手率': 'turnover_rate',
    '代码': 'code'
}

def convert_csv_headers(input_file, output_file=None):
    """
    将CSV文件的中文表头转换为英文表头
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = input_file
    
    print(f"正在读取文件: {input_file}")
    
    # 使用chunksize分批读取大文件
    chunk_size = 100000
    first_chunk = True
    
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        chunk = chunk.dropna()
        if first_chunk:
            # 第一块：转换表头
            print("原始表头:", list(chunk.columns))
            chunk.columns = [header_mapping.get(col, col) for col in chunk.columns]
            print("转换后表头:", list(chunk.columns))
            first_chunk = False
            
            # 写入模式：创建新文件
            chunk.to_csv(output_file, index=False, encoding='utf-8-sig', mode='w')
            print(f"已创建新文件: {output_file}")
        else:
            # 后续块：追加数据
            chunk.to_csv(output_file, index=False, encoding='utf-8-sig', mode='a', header=False)
        
        print(f"已处理第 {i+1} 批数据，共 {len(chunk)} 行")
    
    print(f"表头转换完成！文件已保存到: {output_file}")

def main():
    # 输入文件路径
    input_file = r'c:\Users\Administrator\Documents\qlib\AstockTest\data\df_base_daily_20250930.csv'
    
    # 输出文件路径（覆盖原文件）
    #output_file = input_file  # 覆盖原文件
    
    # 或者指定新文件路径
    output_file = r'c:\Users\Administrator\Documents\qlib\AstockTest\data\df_base_daily_20250930_en.csv'
    
    convert_csv_headers(input_file, output_file)

if __name__ == "__main__":
    main()