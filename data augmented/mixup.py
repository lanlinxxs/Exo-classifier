import os
import numpy as np
import pandas as pd
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


def mixup_augmentation(X, y, target_samples=1000, alpha=0.4):
    """
    使用Mixup方法进行数据扩充
    参数:
        X: 原始特征矩阵
        y: 原始标签
        target_samples: 每类目标样本数
        alpha: Mixup参数，控制插值强度
    """
    # 获取各类样本索引
    class_indices = {0: np.where(y == 0)[0],
                     1: np.where(y == 1)[0],
                     2: np.where(y == 2)[0]}

    # 计算需要生成的样本数
    n_to_generate = {cls: target_samples - len(indices)
                     for cls, indices in class_indices.items()}

    X_aug = []
    y_aug = []

    for cls in [0, 1, 2]:
        # 添加原始样本
        X_aug.extend(X[class_indices[cls]])
        y_aug.extend(y[class_indices[cls]])

        # 生成新样本
        for _ in range(n_to_generate[cls]):
            # 随机选择两个样本（可以是同类或不同类）
            idx1, idx2 = np.random.choice(len(X), 2, replace=True)

            # 生成混合系数
            lam = np.random.beta(alpha, alpha)

            # 混合特征和标签
            mixed_x = lam * X[idx1] + (1 - lam) * X[idx2]
            mixed_y = lam * y[idx1] + (1 - lam) * y[idx2]

            X_aug.append(mixed_x)
            y_aug.append(mixed_y)

    return np.array(X_aug), np.array(y_aug)


def save_augmented_data(fold_dir, subset, X_aug, y_aug, output_dir):
    """
    保存扩充后的数据，保持原始TSV文件结构
    """
    # 将连续标签转换为离散标签（Mixup生成的是软标签）
    y_discrete = np.argmax(np.vstack([
        (y_aug == 0).astype(float),
        (y_aug == 1).astype(float),
        (y_aug == 2).astype(float)
    ]).T, axis=1)

    for class_label in ['0', '1', '2']:
        # 选择最可能属于该类的样本
        class_prob = (y_aug == int(class_label)).astype(float)
        selected = np.where(class_prob > 0.5)[0]  # 概率大于0.5的样本

        output_class_dir = os.path.join(output_dir, f'fold_{os.path.basename(fold_dir)}', subset, class_label)
        os.makedirs(output_class_dir, exist_ok=True)

        # 清空目录
        for f in os.listdir(output_class_dir):
            os.remove(os.path.join(output_class_dir, f))

        # 保存样本
        for i, idx in enumerate(selected):
            output_path = os.path.join(output_class_dir, f'sample_{i}.tsv')
            with open(output_path, 'w') as f:
                for val in X_aug[idx]:
                    f.write(f"gene\tinfo\t{val:.5f}\n")


def augment_and_save_with_mixup(data_dir, n_splits=5, target_samples=1000, alpha=0.4):
    """
    使用Mixup进行数据扩充并保存结果
    保持五折交叉验证结构
    """
    # 创建输出目录
    output_dir = os.path.join(data_dir, 'augmented_mixup_data')
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
        X_aug, y_aug = mixup_augmentation(X_train, y_train, target_samples, alpha)

        # 打印扩充后分布（基于硬标签）
        y_hard = np.argmax(np.vstack([
            (y_aug == 0).astype(float),
            (y_aug == 1).astype(float),
            (y_aug == 2).astype(float)
        ]).T, axis=1)
        unique_aug, counts_aug = np.unique(y_hard, return_counts=True)
        print(f"Fold {fold_idx} 扩充后训练数据分布: {dict(zip(unique_aug, counts_aug))}")

        # 保存扩充后的训练集
        save_augmented_data(fold_dir, 'train', X_aug, y_aug, output_dir)

        # 复制原始验证集（不扩充）
        X_val, y_val = load_and_process_data(fold_dir, 'val')
        save_augmented_data(fold_dir, 'val', X_val, y_val, output_dir)


# 使用示例
data_directory = r'D:\deeplearning\Gene_fanxiu\不同数据扩充方法\cross'
augment_and_save_with_mixup(data_directory,
                            n_splits=5,
                            target_samples=1000,
                            alpha=0.4)  # alpha控制混合强度