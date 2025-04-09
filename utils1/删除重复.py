import pandas as pd

# 读取CSV文件（假设文件名为 genes.csv，没有表头）
df = pd.read_csv('D:\deeplearning\Gene_fanxiu\不同特征选择方法\cross/4.csv', header=None, names=['gene_id'])

# 删除小数点及其后面的内容
df['gene_id'] = df['gene_id'].str.split('.').str[0]

# 删除重复值
df = df.drop_duplicates()

# 保存处理后的结果到新文件（可选）
df.to_csv('D:\deeplearning\Gene_fanxiu\不同特征选择方法\cross\processed_genes.csv', index=False, header=False)

# 显示处理后的结果
print("处理后的基因ID（无重复）：")
print(df)