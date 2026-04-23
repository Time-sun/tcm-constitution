# ablation.py - 消融实验：依次去掉最重要的特征，观察准确率变化
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. 加载原始数据
df = pd.read_csv('constitution_data.csv', encoding='utf-8-sig')
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

# 2. 加载已训练好的 XGBoost 模型（获取特征重要性）
model = joblib.load('xgb_model.pkl')
importance = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importance)[::-1]  # 按重要性降序排列的索引

# 3. 消融实验：依次去掉最重要的 1,2,3 个特征
results = []
for k in [0, 1, 2, 3]:
    if k == 0:
        X_tr = X_train
        X_te = X_test
        desc = "原始特征（无删除）"
    else:
        drop_feats = feature_names[sorted_idx[:k]]
        X_tr = X_train.drop(columns=drop_feats)
        X_te = X_test.drop(columns=drop_feats)
        desc = f"去掉前{k}个重要特征"

    clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.3, random_state=42)
    clf.fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    results.append((desc, acc))
    print(f"{desc} 准确率: {acc:.4f}")

# 4. 保存结果
df_results = pd.DataFrame(results, columns=['特征集', '准确率'])
df_results.to_csv('ablation_results.csv', index=False)
print("\\\\n消融实验结果已保存至 ablation_results.csv")