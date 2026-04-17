import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

# 1. 读取数据
df = pd.read_csv('constitution_data.csv', encoding='utf-8-sig')
print("数据集形状：", df.shape)
print("标签分布：\\n", df['label'].value_counts())

# 2. 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 3. 将标签编码为数字（XGBoost要求）
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# 保存标签编码器，后续预测时需要解码
joblib.dump(le, 'label_encoder.pkl')
print("标签编码映射：", dict(zip(le.classes_, le.transform(le.classes_))))

# 4. 划分训练集、验证集、测试集（7:1.5:1.5）
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"训练集大小：{len(X_train)}，验证集大小：{len(X_val)}，测试集大小：{len(X_test)}")

# 5. 基线模型（默认参数）
baseline_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
print(f"基线模型测试准确率：{acc_baseline:.4f}")

# 6. 超参数优化（网格搜索，可选，如果时间有限可跳过但建议运行）
print("\\n开始网格搜索...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0]
}
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), random_state=42, use_label_encoder=False)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"最佳参数：{grid_search.best_params_}")
best_model = grid_search.best_estimator_

# 7. 在测试集上评估最佳模型
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\\n优化后模型测试准确率：{acc:.4f}")
print("\\n分类报告：")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 混淆矩阵热力图
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_pred_test = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('预测体质')
plt.ylabel('真实体质')
plt.title('混淆矩阵 - XGBoost体质分类')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ 混淆矩阵已保存为 confusion_matrix.png")

# 8. 保存模型
joblib.dump(best_model, 'xgb_model.pkl')
print("模型已保存为 xgb_model.pkl")

# 9. （可选）保存测试集供后续SHAP分析使用
test_data = X_test.copy()
test_data['label_encoded'] = y_test
test_data.to_csv('test_data.csv', index=False)
print("测试集已保存为 test_data.csv")