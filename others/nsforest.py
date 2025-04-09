import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_data(fold_dir, subset):
    """加载并处理单个fold的训练或验证数据（与原始代码相同）"""
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
    """计算指标的统计信息（与原始代码相同）"""
    mean = np.mean(metrics)
    std = np.std(metrics)
    min_val = np.min(metrics)
    max_val = np.max(metrics)
    return f"{mean:.5f} ± {std:.5f} [{min_val:.5f}, {max_val:.5f}]"

def run_nsforest_cross_validation(data_dir, n_splits=5):
    """使用NS-Forest（模拟实现）进行交叉验证"""
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

        # NS-Forest模拟实现（分两步）：
        # 1. 非负稀疏特征选择（用Lasso模拟）
        selector = SelectFromModel(
            Lasso(alpha=0.01, positive=True, max_iter=10000),
            threshold="median"
        ).fit(X_train, y_train)
        X_train_selected = selector.transform(X_train)
        X_val_selected = selector.transform(X_val)

        # 2. 随机森林分类
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_val_selected)

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

    # 汇总结果（与原始代码相同）
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
    plt.title("Average Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# 运行
data_directory = r'D:\deeplearning\Gene_fanxiu\data\cross'  # 使用原始字符串
run_nsforest_cross_validation(data_directory, n_splits=5)