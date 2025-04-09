import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置字体样式
rcParams['font.style'] = 'italic'
rcParams['font.size'] = 10
plt.rcParams['axes.facecolor'] = 'white'  # 设置背景为纯白

# 从CSV文件读取数据
df = pd.read_csv('D:/deeplearning/Gene_fanxiu/utils/pheatmap_example_data3.csv', index_col=0)

# 定义样本分组
sample_types = ['LUSC']*7 + ['Normal']*7 + ['LUAD']*7
colors = {'LUSC':'#FF4500', 'Normal':'#808080', 'LUAD':'#4169E1'}  # 更鲜艳的颜色
sample_colors = [colors[typ] for typ in sample_types]

# 数据预处理
log_df = np.log10(df.replace(0, 1e-4))

# 创建更方正的图形 (宽度增大，高度减小)
fig, ax = plt.subplots(figsize=(16, 12))  # 调整为16:12的比例
sns.set_style("whitegrid", {'grid.color': '.95'})  # 极浅的网格线

# 自定义更鲜艳的蓝-白-红颜色方案
custom_cmap = sns.diverging_palette(20, 220, s=90, l=50, as_cmap=True)

# 绘制热图
heatmap = sns.heatmap(
    log_df,
    cmap=custom_cmap,
    center=0,
    yticklabels=True,
    xticklabels=True,
    ax=ax,
    cbar_kws={
        'label': 'log10(Expr+1e-4)',
        'location': 'right',
        'shrink': 0.3,  # 缩小颜色条
        'aspect': 10    # 使颜色条更细长
    },
    linewidths=0.2,
    linecolor='whitesmoke'
)

# 添加样本类型颜色条 (更细的标记条)
for i, color in enumerate(sample_colors):
    ax.add_patch(
        plt.Rectangle(
            (i, -0.02), 1, 0.02,  # 更细的标记条
            facecolor=color,
            edgecolor='none',
            transform=ax.get_xaxis_transform(),
            clip_on=False
        )
    )

# 设置基因名称斜体
for ylabel in heatmap.get_yticklabels():
    ylabel.set_style('italic')
    ylabel.set_fontsize(10)

# 调整x轴标签
for xlabel in heatmap.get_xticklabels():
    xlabel.set_rotation(90)
    xlabel.set_horizontalalignment('center')
    xlabel.set_fontsize(10)

# 创建图例 (放在右侧，与颜色条平行)
legend_patches = [plt.Rectangle((0,0),0.5,0.5, color=color, label=label)
                 for label, color in colors.items()]

# 精心调整图例位置 (使用gridspec)
plt.subplots_adjust(right=0.8)  # 为图例留出空间
leg = plt.legend(
    handles=legend_patches,
    title='Sample Type',
    bbox_to_anchor=(1.25, 0.5),  # 向右移动
    loc='center left',
    frameon=True,
    framealpha=0.9,
    edgecolor='lightgray',
    borderaxespad=0.5,
    handlelength=1,
    handleheight=1
)

# 调整热图颜色条位置
cbar = heatmap.collections[0].colorbar
cbar.ax.set_position([0.82, 0.15, 0.02, 0.7])  # [左, 下, 宽, 高]

# 移除标题和调整边界
ax.set_title('')
plt.tight_layout(rect=[0, 0, 0.8, 1])  # 右边留20%空间

# 保存图像
plt.savefig(
    'heatmap.png',
    dpi=800,
    bbox_inches='tight',
    facecolor='white',  # 保存为白色背景
    edgecolor='none'
)

plt.show()