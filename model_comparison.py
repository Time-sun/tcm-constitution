import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import time

# 读取数据
df = pd.read_csv('constitution_data.csv', encoding='utf-8-sig')
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

models = {
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.3, subsample=0.8, random_state=42)
}

results = []
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({'模型': name, '准确率': f'{acc:.4f}', '加权F1': f'{f1:.4f}', '训练时间(秒)': f'{train_time:.2f}'})

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))
result_df.to_csv('model_comparison.csv', index=False, encoding='utf-8-sig')
print("\\n✅ 对比结果已保存为 model_comparison.csv")