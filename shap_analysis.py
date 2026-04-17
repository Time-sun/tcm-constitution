import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 忽略警告（不影响结果）
import warnings
warnings.filterwarnings('ignore')

# 1. 加载模型
model = joblib.load('xgb_model.pkl')

# 2. 读取原始数据并重新划分测试集（因为test_data.csv中label是编码后的数字，不方便显示特征名）
df = pd.read_csv('constitution_data.csv', encoding='utf-8-sig')
X = df.drop('label', axis=1)
y = df['label']

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 划分训练集和测试集（只用测试集做SHAP，节省时间）
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
print(f"测试集大小：{len(X_test)}")

# 3. 创建SHAP解释器（TreeExplainer）
print("正在计算SHAP值，请稍等...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 4. 绘制全局特征重要性图（蜂群图）
print("生成蜂群图...")
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 绘制条形图（平均绝对SHAP值）
print("生成条形图...")
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ SHAP分析完成！")
print("生成的图片：shap_summary_plot.png 和 shap_bar_plot.png")
print("请在你的文件夹中查看这两张图片。")