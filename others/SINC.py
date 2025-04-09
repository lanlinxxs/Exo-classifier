import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子（增强可复现性）
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 改进版SINC模型（添加BatchNorm和更稳健的初始化）
class EnhancedSINC(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)  # 输入层归一化

        # 动态构建隐藏层
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 尺度不变处理
        x = torch.log1p(x)  # 更安全的log(1+x)
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.input_norm(x)
        return self.classifier(self.features(x))


# 增强数据预处理
class GeneDataset(Dataset):
    def __init__(self, features, labels):
        # 外部预处理：log1p + 标准化
        self.features = np.log1p(features)
        self.features = (self.features - np.mean(self.features, axis=0)) / (np.std(self.features, axis=0) + 1e-6)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]).unsqueeze(0),  # 保持[B,1,L]形状
            torch.LongTensor([self.labels[idx]])
        )


# 数据加载函数（增加数据校验）
def load_and_process_data(fold_dir, subset):
    X, y = [], []
    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                data = pd.read_csv(os.path.join(class_dir, filename),
                                   sep='\t', header=None, usecols=[2])
                features = data.values.astype(np.float32)
                if np.isnan(features).any():
                    raise ValueError(f"NaN values detected in {filename}")
                X.append(features.flatten())
                y.append(int(class_label))

    X = np.array(X)
    y = np.array(y)

    # 检查类别平衡
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    return X, y


# 改进训练流程（添加学习率调度和早停）
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (total_loss / len(loader),
            correct / len(loader.dataset),
            all_preds, all_labels)


def run_enhanced_sinc(data_dir, n_splits=5, epochs=30, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 结果跟踪
    metrics = {
        'fold': [], 'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for fold in range(n_splits):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        fold_dir = os.path.join(data_dir, f'fold_{fold}')

        # 加载数据
        X_train, y_train = load_and_process_data(fold_dir, 'train')
        X_val, y_val = load_and_process_data(fold_dir, 'val')

        # 创建数据加载器
        train_loader = DataLoader(
            GeneDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            GeneDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        # 初始化模型
        model = EnhancedSINC(
            input_dim=X_train.shape[1],
            num_classes=len(np.unique(y_train)),
            hidden_dims=[256, 128]
        ).to(device)

        # 优化设置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

        best_val_acc = 0.0
        patience = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device
            )

            # 记录指标
            metrics['fold'].append(fold)
            metrics['epoch'].append(epoch)
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_acc'].append(val_acc)

            # 学习率调度
            scheduler.step(val_acc)

            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                torch.save(model.state_dict(), f'sinc_fold{fold}_best.pth')
            else:
                patience += 1
                if patience >= 5:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 加载最佳模型评估
        model.load_state_dict(torch.load(f'sinc_fold{fold}_best.pth', map_location=device))
        _, val_acc, y_pred, y_true = evaluate(model, val_loader, criterion, device)

        # 保存结果
        print(f"\nFold {fold} Best Val Acc: {val_acc:.4f}")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    # 可视化训练过程
    df_metrics = pd.DataFrame(metrics)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_metrics, x='epoch', y='train_loss', hue='fold', legend='full')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_metrics, x='epoch', y='val_acc', hue='fold', legend='full')
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    data_directory = "D:\deeplearning\Gene_fanxiu\data\cross"
    run_enhanced_sinc(
        data_dir=data_directory,
        n_splits=5,
        epochs=30,
        batch_size=16
    )