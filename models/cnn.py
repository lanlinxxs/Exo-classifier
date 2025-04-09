import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 定义精简版1D CNN模型
class LightCNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LightCNN1D, self).__init__()
        self.conv_block = nn.Sequential(
            # 第一层卷积，使用较大的步长进行下采样
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),  # 输出长度减半
            nn.BatchNorm1d(8),
            nn.ReLU(),

            # 第二层卷积，继续下采样
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),  # 输出长度再减半
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # 第三层卷积，保持尺寸
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 再次下采样
        )

        # 计算经过卷积块后的特征维度
        self.feature_dim = 32 * (input_dim // 8)  # 总共下采样8倍

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


# 自定义数据集类（保持不变）
class GeneDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])
        return feature, label


# 数据加载函数（保持不变）
def load_and_process_data(fold_dir, subset):
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


# 统计计算函数（保持不变）
def calculate_statistics(metrics):
    mean = np.mean(metrics)
    std = np.std(metrics)
    min_val = np.min(metrics)
    max_val = np.max(metrics)
    return f"{mean:.5f} ± {std:.5f} [{min_val:.5f}, {max_val:.5f}]"


# 训练函数（保持不变）
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc


# 评估函数（保持不变）
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = running_loss / len(data_loader)
    val_acc = correct / total
    return val_loss, val_acc, all_preds, all_labels


def run_light_cnn_cross_validation(data_dir, n_splits=5, epochs=30, batch_size=64):
    """
    执行精简版1D CNN交叉验证分类
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

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for fold_idx in range(n_splits):
        print(f"\n=== Processing Fold {fold_idx + 1}/{n_splits} ===")
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')

        # 加载训练和验证数据
        X_train, y_train = load_and_process_data(fold_dir, 'train')
        X_val, y_val = load_and_process_data(fold_dir, 'val')

        # 检查数据形状
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")

        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 创建数据集和数据加载器
        train_dataset = GeneDataset(X_train, y_train)
        val_dataset = GeneDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model = LightCNN1D(input_dim, num_classes).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        best_val_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_fold{fold_idx}.pth')

            if (epoch + 1) % 5 == 0:  # 每5个epoch打印一次
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 加载最佳模型进行评估
        model.load_state_dict(torch.load(f'best_model_fold{fold_idx}.pth'))
        _, _, y_pred, y_true = evaluate_model(model, val_loader, criterion, device)

        # 评估结果
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

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

        # 打印当前折结果
        print(f"\nFold {fold_idx} - Validation Results:")
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
    plt.title('Light 1D CNN - Average Confusion Matrix (values rounded to 5 decimal places)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# 使用示例
data_directory = r'D:\deeplearning\Gene_fanxiu\data\cross'  # 使用原始字符串
run_light_cnn_cross_validation(data_directory, n_splits=5, epochs=20, batch_size=64)