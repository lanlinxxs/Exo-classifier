import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_process_data(fold_dir, subset):
    """
    加载并处理单个fold的训练或验证数据
    修改：只读取每个TSV文件的第三列作为特征，并保留5位小数
    """
    X = []
    y = []

    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
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


def calculate_statistics(metrics):
    """
    计算指标的统计信息：平均值±标准差 [最小值, 最大值]
    """
    mean = np.mean(metrics)
    std = np.std(metrics)
    min_val = np.min(metrics)
    max_val = np.max(metrics)
    return f"{mean:.5f} ± {std:.5f} [{min_val:.5f}, {max_val:.5f}]"


def run_random_forest_cross_validation(data_dir, n_splits=5):
    """
    执行随机森林交叉验证分类
    修改：使用随机森林代替LDA
    """
    # 存储各折结果
    all_reports = []
    confusion_matrices = []

    # 存储各项指标的列表
    accuracies = []
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    weighted_precisions = []
    weighted_recalls = []
    weighted_f1s = []

    for fold_idx in range(n_splits):
        print(f"\n=== Processing Fold {fold_idx} ===")
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')

        # 加载训练和验证数据
        X_train, y_train = load_and_process_data(fold_dir, 'train')
        X_val, y_val = load_and_process_data(fold_dir, 'val')

        # 检查数据形状
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")

        # 创建随机森林分类器
        rf_classifier = RandomForestClassifier(
            n_estimators=6,      # 树的数量
            criterion='gini',      # 分裂标准
            max_depth=2,        # 树的最大深度
            min_samples_split=2,   # 分裂内部节点所需的最小样本数
            min_samples_leaf=1,    # 叶节点所需的最小样本数
            max_features='sqrt',   # 寻找最佳分割时考虑的特征数
            bootstrap=True,        # 是否使用bootstrap采样
            random_state=42,
            n_jobs=-1             # 使用所有CPU核心
        )

        # 训练模型
        rf_classifier.fit(X_train, y_train)

        # 验证集预测
        y_pred = rf_classifier.predict(X_val)

        # 评估结果
        report = classification_report(y_val, y_pred, output_dict=True)
        cm = confusion_matrix(y_val, y_pred)

        all_reports.append(report)
        confusion_matrices.append(cm)

        # 收集各项指标
        accuracies.append(report['accuracy'])
        macro_precisions.append(report['macro avg']['precision'])
        macro_recalls.append(report['macro avg']['recall'])
        macro_f1s.append(report['macro avg']['f1-score'])
        weighted_precisions.append(report['weighted avg']['precision'])
        weighted_recalls.append(report['weighted avg']['recall'])
        weighted_f1s.append(report['weighted avg']['f1-score'])

        # 打印当前折结果，数值保留5位小数
        print(f"Fold {fold_idx} - Validation Results:")
        report_df = pd.DataFrame(report).transpose()
        float_cols = report_df.select_dtypes(include=['float64']).columns
        report_df[float_cols] = report_df[float_cols].apply(lambda x: round(x, 5))
        print(report_df.to_string(float_format="%.5f"))

        print("\nConfusion Matrix:")
        print(cm)

    # 汇总所有折的结果
    print("\n=== Final Summary ===")
    print("\n=== Detailed Statistics Across All Folds ===")
    print(f"Accuracy: {calculate_statistics(accuracies)}")
    print(f"Macro Precision: {calculate_statistics(macro_precisions)}")
    print(f"Macro Recall: {calculate_statistics(macro_recalls)}")
    print(f"Macro F1-score: {calculate_statistics(macro_f1s)}")
    print(f"Weighted Precision: {calculate_statistics(weighted_precisions)}")
    print(f"Weighted Recall: {calculate_statistics(weighted_recalls)}")
    print(f"Weighted F1-score: {calculate_statistics(weighted_f1s)}")

    # 按类别统计
    print("\n=== Class-wise Statistics ===")
    for class_label in ['0', '1', '2']:
        precisions = [r[class_label]['precision'] for r in all_reports if class_label in r]
        recalls = [r[class_label]['recall'] for r in all_reports if class_label in r]
        f1s = [r[class_label]['f1-score'] for r in all_reports if class_label in r]

        print(f"\nClass {class_label}:")
        print(f"Precision: {calculate_statistics(precisions)}")
        print(f"Recall: {calculate_statistics(recalls)}")
        print(f"F1-score: {calculate_statistics(f1s)}")

    # 绘制平均混淆矩阵
    avg_cm = np.mean(confusion_matrices, axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.round(avg_cm, 5), annot=True, fmt='.5f', cmap='Blues',
                xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
    plt.title('Random Forest - Average Confusion Matrix (values rounded to 5 decimal places)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# 使用示例
data_directory = r'D:\deeplearning\Gene_fanxiu\data\cross'  # 使用原始字符串
run_random_forest_cross_validation(data_directory, n_splits=5)