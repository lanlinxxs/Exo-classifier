import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_process_data(fold_dir, subset):
    """
    加载并处理单个fold的训练或验证数据
    返回原始数据和标签
    """
    X = []
    y = []
    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                filepath = os.path.join(class_dir, filename)
                data = pd.read_csv(filepath, sep='\t', header=None, usecols=[2])
                features = np.round(data.values.astype(float), 5).flatten()
                X.append(features)
                y.append(int(class_label))
    return np.array(X), np.array(y)


def augment_with_adasyn(X, y, target_samples=1000):
    """
    使用ADASYN方法进行数据扩充
    参数:
        X: 原始特征矩阵
        y: 原始标签
        target_samples: 每类目标样本数
    """
    # 计算当前各类样本数
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # 设置ADASYN采样策略
    sampling_strategy = {cls: max(target_samples, count)
                         for cls, count in class_counts.items()}

    # 应用ADASYN
    adasyn = ADASYN(sampling_strategy=sampling_strategy,
                    n_neighbors=min(5, min(class_counts.values()) - 1),
                    random_state=42)
    X_aug, y_aug = adasyn.fit_resample(X, y)

    return X_aug, y_aug


def save_augmented_data(fold_dir, subset, X_aug, y_aug, output_dir):
    """
    保存扩充后的数据，保持原始TSV文件结构
    """
    for class_label in ['0', '1', '2']:
        class_mask = (y_aug == int(class_label))
        X_class = X_aug[class_mask]

        output_class_dir = os.path.join(output_dir, f'fold_{os.path.basename(fold_dir)}', subset, class_label)
        os.makedirs(output_class_dir, exist_ok=True)

        # 清空目录（如果已有文件）
        for f in os.listdir(output_class_dir):
            os.remove(os.path.join(output_class_dir, f))

        # 保存为单独的TSV文件
        for i in range(len(X_class)):
            output_path = os.path.join(output_class_dir, f'sample_{i}.tsv')
            # 创建与原始文件相同格式的TSV（假设有3列）
            with open(output_path, 'w') as f:
                for val in X_class[i]:
                    f.write(f"gene\tinfo\t{val:.5f}\n")


def augment_and_save_with_adasyn(data_dir, n_splits=5, target_samples=1000):
    """
    使用ADASYN进行数据扩充并保存结果
    保持五折交叉验证结构
    """
    # 创建输出目录
    output_dir = os.path.join(data_dir, 'augmented_adasyn_data')
    os.makedirs(output_dir, exist_ok=True)

    for fold_idx in range(n_splits):
        print(f"\n=== Processing Fold {fold_idx} ===")
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')

        # 加载原始训练数据
        X_train, y_train = load_and_process_data(fold_dir, 'train')

        # 打印原始数据分布
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Fold {fold_idx} 原始训练数据分布: {dict(zip(unique, counts))}")

        # 数据扩充
        X_aug, y_aug = augment_with_adasyn(X_train, y_train, target_samples)

        # 打印扩充后分布
        unique_aug, counts_aug = np.unique(y_aug, return_counts=True)
        print(f"Fold {fold_idx} 扩充后训练数据分布: {dict(zip(unique_aug, counts_aug))}")

        # 保存扩充后的训练集
        save_augmented_data(fold_dir, 'train', X_aug, y_aug, output_dir)

        # 复制原始验证集（不扩充）
        X_val, y_val = load_and_process_data(fold_dir, 'val')
        save_augmented_data(fold_dir, 'val', X_val, y_val, output_dir)


# 使用示例
data_directory = r'D:\deeplearning\Gene_fanxiu\不同数据扩充方法\cross'
augment_and_save_with_adasyn(data_directory,
                             n_splits=5,
                             target_samples=1000)