import matplotlib.pyplot as plt
import numpy as np

# 数据
constitutions = ['平和质', '气虚质', '气郁质', '湿热质', '特禀质', '痰湿质', '血瘀质', '阳虚质', '阴虚质']
consistency = [75.2, 70.8, 87.6, 73.5, 82.5, 78.0, 83.5, 94.0, 83.5]

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
plt.rcParams['axes.unicode_minus'] = False

# 绘图
plt.figure(figsize=(10, 6))
bars = plt.bar(constitutions, consistency, color='#4caf50', edgecolor='#2e7d32', linewidth=1.5)
plt.ylim(0, 100)
plt.ylabel('重合度 (%)', fontsize=12)
plt.title('各体质SHAP特征与中医理论核心特征重合度', fontsize=14)
plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, consistency):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('consistency_bar_chart.png', dpi=300)
plt.show()
print("柱状图已保存为 consistency_bar_chart.png")