import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest


def load_gene_list(gene_list_file):
    """加载基因ID列表文件"""
    with open(gene_list_file, 'r') as f:
        gene_ids = set(line.strip() for line in f if line.strip())
    return gene_ids


def load_all_samples(fold_dir, subset, gene_ids):
    """
    加载指定fold和subset的所有样本数据
    每个样本是一个TSV文件，只保留基因ID列表中的行
    返回: 样本矩阵 (n_samples, n_features), 标签数组, 特征索引
    """
    all_samples = []
    labels = []
    feature_indices = []

    # 首先读取第一个文件确定特征
    sample_dir = os.path.join(fold_dir, subset, '0')
    sample_file = os.listdir(sample_dir)[0]

    # 获取匹配的行索引
    with open(os.path.join(sample_dir, sample_file), 'r') as f:
        lines = f.readlines()
        feature_indices = [i for i, line in enumerate(lines)
                           if line.split('\t')[0] in gene_ids]

    print(f"从每个样本中保留 {len(feature_indices)} 个特征")

    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                filepath = os.path.join(class_dir, filename)

                # 读取匹配的行数据
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    selected_lines = [lines[i] for i in feature_indices]

                # 提取第三列数据
                data = [float(line.split('\t')[2]) for line in selected_lines]
                all_samples.append(data)
                labels.append(int(class_label))

    return np.array(all_samples), np.array(labels), feature_indices


def save_selected_features(fold_dir, subset, selected_indices, output_dir):
    """保存筛选后的数据"""
    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
        output_class_dir = os.path.join(output_dir, f'fold_{os.path.basename(fold_dir)}', subset, class_label)
        os.makedirs(output_class_dir, exist_ok=True)

        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                filepath = os.path.join(class_dir, filename)
                output_filepath = os.path.join(output_class_dir, filename)

                with open(filepath, 'r') as f:
                    all_lines = f.readlines()

                selected_lines = [all_lines[i] for i in selected_indices]

                with open(output_filepath, 'w') as f:
                    f.writelines(selected_lines)


def perform_gene_selection(data_dir, gene_list_file, n_splits=5):
    """
    根据基因列表筛选数据

    参数:
    data_dir: 数据目录路径
    gene_list_file: 基因ID列表文件路径
    n_splits: 交叉验证折数
    """
    # 加载基因列表
    gene_ids = load_gene_list(gene_list_file)

    # 创建输出目录
    output_dir = os.path.join(data_dir, 'selected_by_gene_list')
    os.makedirs(output_dir, exist_ok=True)

    for fold_idx in range(n_splits):
        print(f"\n=== 处理第 {fold_idx} 折 ===")
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')

        # 加载数据并获取匹配的特征索引
        _, _, feature_indices = load_all_samples(fold_dir, 'train', gene_ids)

        # 保存筛选后的数据
        save_selected_features(fold_dir, 'train', feature_indices, output_dir)
        save_selected_features(fold_dir, 'val', feature_indices, output_dir)

        print(f"第 {fold_idx} 折的数据筛选完成")


# 使用示例
data_directory = r'D:\deeplearning\Gene_fanxiu\不同特征选择方法\cross'
gene_list_file = r'D:\deeplearning\Gene_fanxiu\不同特征选择方法\cross\4.csv'  # 替换为实际路径
perform_gene_selection(data_directory, gene_list_file, n_splits=5)