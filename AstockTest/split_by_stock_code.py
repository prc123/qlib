import pandas as pd
import os
from pathlib import Path

def split_csv_by_stock_code(input_file, output_dir):
    """
    根据股票代码拆分CSV文件（优化版：只读取一次文件）
    
    Args:
        input_file (str): 输入CSV文件路径
        output_dir (str): 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在读取文件: {input_file}")
    
    # 使用chunksize分批读取大文件
    chunk_size = 100000
    stock_data = {}
    total_rows = 0
    
    print("正在处理数据...")
    
    # 单次读取：同时收集股票代码和数据
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        total_rows += len(chunk)
        print(f"处理第 {i+1} 批数据，当前累计 {total_rows} 行...")
        
        # 按股票代码分组并保存到字典
        for code, group in chunk.groupby('stock_code'):
            if code not in stock_data:
                stock_data[code] = []
            stock_data[code].append(group)
    
    print(f"文件总行数: {total_rows}")
    print(f"发现 {len(stock_data)} 个不同的股票代码")
    
    # 保存每个股票的数据到单独的CSV文件
    print("正在保存文件...")
    for code, data_chunks in stock_data.items():
        if data_chunks:  # 确保有数据
            # 合并所有chunk
            combined_data = pd.concat(data_chunks, ignore_index=True)
            
            # 按日期排序
            combined_data = combined_data.sort_values('date')
            
            # 生成输出文件名
            output_file = os.path.join(output_dir, f'{code}.csv')
            
            # 保存到CSV
            combined_data.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"已保存: {output_file} (共 {len(combined_data)} 行)")
    
    print("拆分完成！")

def split_csv_simple(input_file, output_dir):
    """
    简单的拆分方法（适用于较小文件）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在读取文件: {input_file}")
    
    # 读取整个文件
    df = pd.read_csv(input_file)
    print(f"文件总行数: {len(df)}")
    
    # 按股票代码分组
    grouped = df.groupby('stock_code')
    
    # 保存每个组到单独的CSV文件
    for code, group in grouped:
        # 按日期排序
        group = group.sort_values('date')
        
        # 生成输出文件名
        output_file = os.path.join(output_dir, f'{code}.csv')
        
        # 保存到CSV
        group.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"已保存: {output_file} (共 {len(group)} 行)")
    
    print("拆分完成！")

def main():
    # 输入文件路径
    input_file = r'c:\Users\Administrator\Documents\qlib\AstockTest\data\df_base_daily_20250930_en.csv'
    
    # 输出目录
    output_dir = r'c:\Users\Administrator\Documents\qlib\AstockTest\split_stocks'
    
    # 检查文件大小决定使用哪种方法
    file_size = os.path.getsize(input_file)
    
    if file_size > 100 * 1024 * 1024:  # 大于100MB使用分批处理
        print("文件较大，使用分批处理方法...")
        split_csv_by_stock_code(input_file, output_dir)
    else:
        print("文件较小，使用简单处理方法...")
        split_csv_simple(input_file, output_dir)

if __name__ == "__main__":
    main()