import os
import pandas as pd
from tqdm import tqdm  # 用于显示进度条


def load_reference_ids(reference_tsv_path):
    """加载参考TSV文件中的基因ID集合"""
    ref_df = pd.read_csv(reference_tsv_path, sep='\t', header=None, usecols=[0])
    return set(ref_df[0].astype(str))


def process_tsv_file(input_path, ref_ids, output_path):
    """处理单个TSV文件"""
    try:
        df = pd.read_csv(input_path, sep='\t', header=None)
        filtered_df = df[df[0].astype(str).isin(ref_ids)]
        filtered_df.to_csv(output_path, sep='\t', index=False, header=False)
        return True
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {str(e)}")
        return False


def process_directory(data_dir, ref_ids, output_base_dir, mode='train'):
    """处理整个目录结构"""
    class_dirs = ['0', '1', '2']
    processed_files = 0

    for class_dir in class_dirs:
        input_dir = os.path.join(data_dir, mode, class_dir)
        output_dir = os.path.join(output_base_dir, mode, class_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n处理 {mode}/{class_dir} 目录...")
        tsv_files = [f for f in os.listdir(input_dir) if f.endswith('.tsv')]

        for filename in tqdm(tsv_files, desc=f"Class {class_dir}"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            if process_tsv_file(input_path, ref_ids, output_path):
                processed_files += 1

    return processed_files


def main():
    # 配置路径
    reference_tsv = "D:\deeplearning\Gene_fanxiu\不同特征选择方法\cross\processed_genes.csv"  # 替换为您的参考文件路径
    input_data_dir = "D:\deeplearning\Gene_fanxiu\data_waibu"  # 包含train/val的根目录
    output_base_dir = "D:\deeplearning\Gene_fanxiu/filtered_data"  # 输出目录

    # 加载参考基因ID
    print("加载参考基因ID...")
    ref_ids = load_reference_ids(reference_tsv)
    print(f"已加载 {len(ref_ids)} 个参考基因ID")

    # 处理train和val数据
    total_processed = 0
    for mode in ['train', 'val']:
        print(f"\n开始处理 {mode} 数据...")
        processed = process_directory(input_data_dir, ref_ids, output_base_dir, mode)
        total_processed += processed
        print(f"完成 {mode} 数据处理，共处理 {processed} 个文件")

    print(f"\n全部完成！共处理 {total_processed} 个文件")


if __name__ == "__main__":
    main()