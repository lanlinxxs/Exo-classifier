import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import PoissonRegressor
from sklearn.decomposition import FactorAnalysis  # 用于潜在变量建模
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_data(fold_dir, subset):
    """加载并处理数据（与原始代码相同）"""
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

def calculate_statistics(metrics):
    """计算统计信息（与原始代码相同）"""
    mean = np.mean(metrics)
    std = np.std(metrics)
    min_val = np.min(metrics)
    max_val = np.max(metrics)
    return f"{mean:.5f} ± {std:.5f} [{min_val:.5f}, {max_val:.5f}]"

def run_splr_cross_validation(data_dir, n_splits=5, n_latent=3):
    """使用SPLR（近似实现）进行交叉验证"""
    all_reports = []
    confusion_matrices = []
    accuracies = []
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for fold_idx in range(n_splits):
        print(f"\n=== Processing Fold {fold_idx} ===")
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')

        # 加载数据
        X_train, y_train = load_and_process_data(fold_dir, 'train')
        X_val, y_val = load_and_process_data(fold_dir, 'val')

        # SPLR 近似实现分三步：
        # 1. 潜在变量提取（模拟γ_i）
        fa = FactorAnalysis(n_components=n_latent, random_state=42)
        gamma_train = fa.fit_transform(X_train)
        gamma_val = fa.transform(X_val)

        # 2. 稀疏泊松回归（模拟β_jk，需对每个类别独立拟合）
        # 注意：这里简化处理，实际SPLR应对每个基因拟合回归
        # 此处改为多分类的泊松回归（通过One-vs-Rest策略）
        pipe = make_pipeline(
            StandardScaler(),
            PoissonRegressor(alpha=0.1, max_iter=1000)  # L1正则化
        )

        # 训练和预测（需处理多分类问题）
        y_pred = []
        for class_label in [0, 1, 2]:
            # 二分类标签（当前类别 vs 其他）
            y_binary = (y_train == class_label).astype(int)
            pipe.fit(np.hstack([X_train, gamma_train]), y_binary)  # 拼接特征和潜在变量
            pred = pipe.predict(np.hstack([X_val, gamma_val]))
            y_pred.append(pred)

        # 选择预测概率最高的类别
        y_pred = np.argmax(np.column_stack(y_pred), axis=1)

        # 评估
        report = classification_report(y_val, y_pred, output_dict=True)
        cm = confusion_matrix(y_val, y_pred)

        all_reports.append(report)
        confusion_matrices.append(cm)
        accuracies.append(report['accuracy'])
        macro_precisions.append(report['macro avg']['precision'])
        macro_recalls.append(report['macro avg']['recall'])
        macro_f1s.append(report['macro avg']['f1-score'])

        print(f"Fold {fold_idx} - Validation Results:")
        print(classification_report(y_val, y_pred, digits=5))

    # 汇总结果
    print("\n=== Final Summary ===")
    print(f"Accuracy: {calculate_statistics(accuracies)}")
    print(f"Macro Precision: {calculate_statistics(macro_precisions)}")
    print(f"Macro Recall: {calculate_statistics(macro_recalls)}")
    print(f"Macro F1-score: {calculate_statistics(macro_f1s)}")

    # 绘制混淆矩阵
    avg_cm = np.mean(confusion_matrices, axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
    plt.title("Average Confusion Matrix (SPLR Approximate)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# 运行
data_directory = r'D:\deeplearning\Gene_fanxiu\data\cross'
run_splr_cross_validation(data_directory, n_splits=5)