import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取 CSV 文件（替换为你实际的文件名）
df = pd.read_csv("D:/deeplearning/Gene_fanxiu/utils/pheatmap_example_data2.csv")  # 你可以直接拖进来替换文件名

# 拆分基因名和表达数据
gene_names = df.iloc[:, 0]
expression_data = df.iloc[:, 1:]

# 步骤 1：对每一行进行归一化（Min-Max 归一化）
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(expression_data.T).T  # 按行归一化

normalized_df = pd.DataFrame(normalized_values, columns=expression_data.columns)

# 步骤 2：提取中间7个样本（第8到14列，对应索引7到13）
middle_7_samples = normalized_df.iloc[:, 7:14]

# 步骤 3：计算中间7个样本的平均值
mid_avg = middle_7_samples.mean(axis=1)

# 步骤 4：根据中间值排序，从高到低
sorted_indices = mid_avg.sort_values(ascending=False).index
sorted_df = df.loc[sorted_indices].reset_index(drop=True)

# 步骤 5：添加调控方向（高表达视为下调基因）
threshold = mid_avg.median()
regulation_labels = ["Downregulated" if val > threshold else "Upregulated" for val in mid_avg[sorted_indices]]
sorted_df["Regulation"] = regulation_labels

# 保存结果
sorted_df.to_csv("D:/deeplearning/Gene_fanxiu/utils/processed_output.csv", index=False)

print("处理完成，结果已保存为 processed_output.csv")
