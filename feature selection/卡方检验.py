import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest


def load_all_samples(fold_dir, subset):
    """
    加载指定fold和subset的所有样本数据
    每个样本是一个TSV文件，第三列是特征数据，每行一个特征值
    返回: 样本矩阵 (n_samples, n_features), 标签数组
    """
    all_samples = []
    labels = []

    # 首先确定特征数量（通过检查第一个文件）
    sample_dir = os.path.join(fold_dir, subset, '0')  # 检查class 0的第一个文件
    sample_file = os.listdir(sample_dir)[0]
    with open(os.path.join(sample_dir, sample_file), 'r') as f:
        n_features = sum(1 for line in f if line.strip())

    print(f"确定每个样本有 {n_features} 个特征")

    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                filepath = os.path.join(class_dir, filename)
                # 读取第三列数据，每行一个特征值
                data = pd.read_csv(filepath, sep='\t', header=None, usecols=[2])
                features = data[2].values.astype(float)

                # 确保特征数量一致
                if len(features) != n_features:
                    raise ValueError(f"文件 {filename} 有 {len(features)} 个特征，但预期 {n_features} 个")

                all_samples.append(features)
                labels.append(int(class_label))

    return np.array(all_samples), np.array(labels)


def save_selected_features(fold_dir, subset, selector, output_dir):
    """
    保存经过特征选择后的数据
    保持原始文件格式：每个样本一个TSV文件，只保留被选中的特征行
    """
    # 获取被选中的特征索引
    selected_indices = selector.get_support(indices=True)

    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
        output_class_dir = os.path.join(output_dir, f'fold_{os.path.basename(fold_dir)}', subset, class_label)
        os.makedirs(output_class_dir, exist_ok=True)

        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                filepath = os.path.join(class_dir, filename)
                output_filepath = os.path.join(output_class_dir, filename)

                # 读取原始文件所有行
                with open(filepath, 'r') as f:
                    all_lines = f.readlines()

                # 只保留被选中的特征行（注意：行号从0开始）
                selected_lines = [all_lines[i] for i in selected_indices if i < len(all_lines)]

                # 保存到新文件
                with open(output_filepath, 'w') as f:
                    f.writelines(selected_lines)


def perform_feature_selection(data_dir, n_splits=5, k_features=100):
    """
    执行基于卡方检验的特征选择并保存处理后的数据
    特征选择基于训练集进行，然后统一应用于训练集和验证集

    参数:
    data_dir: 数据目录路径
    n_splits: 交叉验证折数
    k_features: 要选择的最佳特征数量
    """
    # 创建输出目录
    output_dir = os.path.join(data_dir, f'selected_features_chi2_k{k_features}')
    os.makedirs(output_dir, exist_ok=True)

    for fold_idx in range(n_splits):
        print(f"\n=== 处理第 {fold_idx} 折 ===")
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')

        # 1. 加载训练数据用于特征选择
        X_train, y_train = load_all_samples(fold_dir, 'train')

        # 2. 创建卡方检验特征选择器并拟合训练数据
        selector = SelectKBest(score_func=chi2, k=k_features)

        # 卡方检验要求特征值为非负数，因此需要确保数据符合要求
        X_train_non_negative = X_train - np.min(X_train, axis=0)  # 平移使最小值为0

        selector.fit(X_train_non_negative, y_train)  # 卡方检验需要标签信息

        print(f"原始特征数量: {X_train.shape[1]}")
        print(f"选择后的特征数量: {selector.get_support().sum()}")
        print(f"被选中的特征索引: {selector.get_support(indices=True)}")

        # 打印特征得分（卡方统计量）
        print(f"卡方统计量: {selector.scores_[selector.get_support(indices=True)]}")

        # 3. 保存特征选择后的训练集和验证集
        save_selected_features(fold_dir, 'train', selector, output_dir)
        save_selected_features(fold_dir, 'val', selector, output_dir)

        print(f"第 {fold_idx} 折的特征选择结果已保存")


# 使用示例
data_directory = r'D:\deeplearning\Gene_fanxiu\不同特征选择方法\cross'  # 使用原始字符串
perform_feature_selection(data_directory, n_splits=5, k_features=1000)