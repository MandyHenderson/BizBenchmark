import pandas as pd
import json

def test_cikm18_reading():
    """
    测试CIKM18数据集的读取
    """
    # 定义数据集路径
    splits = {
        'train': 'data/train-00000-of-00001-f71a7dda3fae0889.parquet', 
        'test': 'data/test-00000-of-00001-e1663a0932037903.parquet', 
        'valid': 'data/valid-00000-of-00001-b105ab56855808e4.parquet'
    }
    
    base_url = "hf://datasets/TheFinAI/flare-sm-cikm/"
    
    # 只测试train数据集
    split_name = 'train'
    split_path = splits[split_name]
    full_path = base_url + split_path
    
    print(f"正在测试读取: {full_path}")
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(full_path)
        print(f"成功读取 {split_name} 数据集，共 {len(df)} 行")
        print(f"列名: {list(df.columns)}")
        
        # 显示前几行数据
        print("\n前3行数据:")
        print(df.head(3))
        
        # 显示数据结构
        print(f"\n数据类型:")
        print(df.dtypes)
        
        return df
        
    except Exception as e:
        print(f"读取失败: {e}")
        return None

if __name__ == "__main__":
    df = test_cikm18_reading()
