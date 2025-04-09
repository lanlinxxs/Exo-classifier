import os
import pandas as pd


def split_tsv(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹
    df = pd.read_csv(input_file, sep='\t')  # 读取TSV文件
    first_col = df.iloc[:, 0]  # 获取第一列

    for i in range(1, df.shape[1]):  # 遍历第二列到最后一列
        output_file = os.path.join(output_folder, f"output_column_{i}.tsv")
        split_df = pd.concat([first_col, df.iloc[:, i]], axis=1)  # 组合第一列和当前列
        split_df.to_csv(output_file, sep='\t', index=False)  # 保存为TSV
        print(f"Saved: {output_file}")


# 示例调用
split_tsv("D:\deeplearning\Gene_fanxiu\data\FPKM_cufflinks.tsv", "D:\deeplearning\Gene_fanxiu\data/3")
