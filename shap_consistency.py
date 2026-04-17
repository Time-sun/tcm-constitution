import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# 加载模型和数据
model = joblib.load('xgb_model.pkl')
le = joblib.load('label_encoder.pkl')
df = pd.read_csv('constitution_data.csv', encoding='utf-8-sig')
X = df.drop('label', axis=1)
y = df['label']

y_encoded = le.transform(y)
_, X_test, _, y_test_encoded = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
y_test = le.inverse_transform(y_test_encoded)

# 金标准特征（根据你的实际列名）
gold_features = {
    '平和质': ['精力充沛', '情绪稳定', '睡眠良好'],
    '气虚质': ['神疲乏力', '气短懒言', '自汗'],
    '阳虚质': ['畏寒怕冷', '四肢不温', '喜热饮食'],
    '阴虚质': ['口干咽燥', '五心烦热', '潮热盗汗'],
    '痰湿质': ['形体肥胖', '肢体困重', '胸闷痰多'],
    '湿热质': ['面垢油光', '口苦口臭', '大便黏滞'],
    '血瘀质': ['肤色晦暗', '刺痛部位', '唇色紫暗'],
    '气郁质': ['情绪低落', '胸胁胀满', '咽部异物感'],
    '特禀质': ['过敏史', '喷嚏流涕', '皮肤瘙痒']
}

print("创建SHAP解释器...")
explainer = shap.TreeExplainer(model)

n_samples = 100
np.random.seed(42)
indices = np.random.choice(len(X_test), n_samples, replace=False)
X_sample = X_test.iloc[indices]
y_sample = y_test[indices]

print(f"抽取 {n_samples} 个测试样本，计算SHAP值...")
shap_values = explainer.shap_values(X_sample)  # 形状: (n_samples, n_features, n_classes)

consistency_list = []
details = []

for i in range(n_samples):
    true_label = y_sample[i]
    pred_encoded = model.predict(X_sample.iloc[[i]])[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    if pred_label != true_label:
        continue

    # 正确提取SHAP值：第i个样本，所有特征，预测类别
    if isinstance(shap_values, list):
        # 多分类返回list，每个元素形状 (n_samples, n_features)
        shap_sample = shap_values[pred_encoded][i]
    else:
        # 三维数组
        shap_sample = shap_values[i, :, pred_encoded]

    shap_sample = np.array(shap_sample).flatten()
    # 取绝对值最大的3个特征
    top_indices = np.argsort(np.abs(shap_sample))[-3:][::-1]
    top_features = [X_sample.columns[j] for j in top_indices]

    gold = gold_features.get(pred_label, [])
    if gold:
        overlap = len(set(top_features) & set(gold))
        consistency = overlap / len(gold)
        consistency_list.append(consistency)
        details.append({
            '真实体质': true_label,
            '预测体质': pred_label,
            'Top3特征': ', '.join(top_features),
            '重合度': f"{consistency:.0%}"
        })

avg_consistency = np.mean(consistency_list) if consistency_list else 0
print(f"\\n有效样本数: {len(consistency_list)}")
print(f"平均重合度: {avg_consistency:.2%}")

if details:
    df_details = pd.DataFrame(details)
    print("\\n各体质平均重合度：")
    grouped = df_details.groupby('预测体质')['重合度'].apply(lambda x: x.str.rstrip('%').astype(float).mean())
    for c, m in grouped.items():
        print(f"  {c}: {m:.1f}%")
    df_details.to_csv('shap_consistency_details.csv', index=False, encoding='utf-8-sig')
    print("\\n详细结果已保存为 shap_consistency_details.csv")
else:
    print("没有有效数据，请检查预测准确率或金标准定义。")