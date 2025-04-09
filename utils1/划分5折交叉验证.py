import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import shutil


def prepare_cross_validation(input_dir, output_dir, n_splits=5):
    """
    对三类数据集进行分层交叉验证划分

    参数:
        input_dir: 输入目录结构应为:
            input_dir/
                0/  # 类别0的样本
                1/  # 类别1的样本
                2/  # 类别2的样本
        output_dir: 输出目录
        n_splits: 交叉验证折数(默认为5)
    """
    # 创建输出目录结构
    for split in range(n_splits):
        for subset in ['train', 'val']:
            for class_label in ['0', '1', '2']:
                os.makedirs(os.path.join(output_dir, f'fold_{split}', subset, class_label), exist_ok=True)

    # 收集所有文件路径和对应的标签
    file_paths = []
    labels = []

    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(input_dir, class_label)
        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                file_paths.append(os.path.join(class_dir, filename))
                labels.append(int(class_label))

    # 转换为numpy数组
    file_paths = np.array(file_paths)
    labels = np.array(labels)

    # 使用分层K折交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        # 创建训练集和验证集的符号链接(或复制文件)
        for idx, subset in zip([train_idx, val_idx], ['train', 'val']):
            for i in idx:
                src_path = file_paths[i]
                class_label = str(labels[i])
                filename = os.path.basename(src_path)
                dest_path = os.path.join(output_dir, f'fold_{fold_idx}', subset, class_label, filename)

                # 使用复制文件(如需符号链接可替换为os.symlink)
                shutil.copy2(src_path, dest_path)

        print(f'Fold {fold_idx} created: {len(train_idx)} train samples, {len(val_idx)} val samples')


# 使用示例
input_directory = 'D:\deeplearning\Gene_fanxiu\data\cross5'  # 替换为你的输入目录
output_directory = 'D:\deeplearning\Gene_fanxiu\data\cross'  # 替换为你的输出目录
prepare_cross_validation(input_directory, output_directory, n_splits=5)