import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample


def load_all_data(data_dir, n_splits=5):
    """
    加载所有fold的训练和验证数据
    """
    all_X = []
    all_y = []

    for fold_idx in range(n_splits):
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')

        # 加载训练集
        X_train, y_train = load_and_process_data(fold_dir, 'train')
        # 加载验证集
        X_val, y_val = load_and_process_data(fold_dir, 'val')

        # 合并数据
        all_X.append(X_train)
        all_X.append(X_val)
        all_y.append(y_train)
        all_y.append(y_val)

    # 合并所有数据
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    return X, y


def load_and_process_data(fold_dir, subset):
    """
    加载单个fold的训练或验证数据
    """
    X = []
    y = []

    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue

        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                filepath = os.path.join(class_dir, filename)
                # 读取TSV文件，只取第三列
                data = pd.read_csv(filepath, sep='\t', header=None, usecols=[2])
                # 将数据转换为浮点数并保留5位小数
                features = np.round(data.values.astype(float), 5).flatten()
                X.append(features)
                y.append(int(class_label))

    return np.array(X), np.array(y)


def plot_sampled_heatmap(X, y, n_samples_per_class=20, n_genes=100, random_state=42):
    """
    绘制采样后的热图(每类n_samples_per_class个样本，前n_genes个基因)
    """
    # 随机采样样本
    sampled_indices = []
    for class_label in [0, 1, 2]:
        class_indices = np.where(y == class_label)[0]
        if len(class_indices) > n_samples_per_class:
            sampled = resample(class_indices,
                               replace=False,
                               n_samples=n_samples_per_class,
                               random_state=random_state)
        else:
            sampled = class_indices  # 如果样本不足则取全部
        sampled_indices.extend(sampled)

    # 选取前n_genes个基因
    X_sampled = X[np.array(sampled_indices), :n_genes]
    y_sampled = y[np.array(sampled_indices)]

    # 按类别排序样本
    sort_idx = np.argsort(y_sampled)
    X_sorted = X_sampled[sort_idx]
    y_sorted = y_sampled[sort_idx]

    # 创建绘图
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(X_sorted, cmap='viridis',
                     yticklabels=True,  # 显示样本标签
                     xticklabels=np.arange(1, n_genes + 1),  # 显示基因编号
                     cbar_kws={'label': 'Expression Level'})

    # 添加类别分隔线
    class_changes = np.where(np.diff(y_sorted) != 0)[0] + 1
    for change in class_changes:
        ax.axhline(change, color='white', linewidth=1)

    # 添加类别标签
    class_boundaries = np.concatenate([[0], class_changes, [len(y_sorted)]])
    for i in range(len(class_boundaries) - 1):
        y_pos = (class_boundaries[i] + class_boundaries[i + 1]) / 2
        ax.text(-0.5, y_pos, f'Class {y_sorted[class_boundaries[i]]}',
                ha='right', va='center', color='black', fontsize=12)

    plt.title(f'Gene Expression Heatmap ({n_samples_per_class * 3} samples, {n_genes} genes)', fontsize=14)
    plt.xlabel('Gene Index', fontsize=12)
    plt.ylabel('Samples', fontsize=12)

    plt.tight_layout()
    plt.show()

    return X_sampled, y_sampled


# 使用示例
if __name__ == "__main__":
    data_directory = r'D:\deeplearning\Gene_fanxiu\data\cross'  # 使用原始字符串

    # 加载所有数据
    X_all, y_all = load_all_data(data_directory, n_splits=5)

    print(f"Loaded {len(y_all)} samples in total")
    print(f"Class distribution: {np.unique(y_all, return_counts=True)}")

    # 绘制采样热图(每类20个样本，前100个基因)
    X_sampled, y_sampled = plot_sampled_heatmap(X_all, y_all,
                                                n_samples_per_class=20,
                                                n_genes=100)

    print(f"\nSampled data shape: {X_sampled.shape}")
    print(f"Sampled class distribution: {np.unique(y_sampled, return_counts=True)}")