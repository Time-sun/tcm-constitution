import pandas as pd
import numpy as np

np.random.seed(42)

constitutions = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

# 定义每个体质的典型特征（均值4-5分）以及可能出现的次要特征（均值3-4分）
# 为了更真实，增加体质间的症状重叠
features_profile = {
    '平和质': {'精力充沛': 4.5, '情绪稳定': 4.2, '睡眠良好': 4.3, '适应力强': 4.4},
    '气虚质': {'神疲乏力': 4.5, '气短懒言': 4.3, '自汗': 4.0, '易感冒': 4.2, '畏寒怕冷': 2.5, '精神不振': 3.8},
    '阳虚质': {'畏寒怕冷': 4.6, '四肢不温': 4.5, '喜热饮食': 4.3, '神疲乏力': 3.8, '精神不振': 4.0, '自汗': 3.5},
    '阴虚质': {'口干咽燥': 4.6, '五心烦热': 4.4, '潮热盗汗': 4.0, '大便干结': 3.9, '失眠多梦': 3.5},
    '痰湿质': {'形体肥胖': 4.3, '肢体困重': 4.4, '胸闷痰多': 4.1, '口黏腻': 4.0, '大便黏滞': 3.8},
    '湿热质': {'面垢油光': 4.5, '口苦口臭': 4.4, '身重困倦': 4.0, '大便黏滞': 4.1, '肢体困重': 3.5},
    '血瘀质': {'肤色晦暗': 4.3, '刺痛部位': 4.2, '唇色紫暗': 4.0, '经血色暗': 3.9},
    '气郁质': {'情绪低落': 4.6, '胸胁胀满': 4.4, '咽部异物感': 4.0, '失眠多梦': 3.9},
    '特禀质': {'过敏史': 4.7, '喷嚏流涕': 4.5, '皮肤瘙痒': 4.3, '哮喘': 3.9}
}

# 所有特征
all_features = sorted(set([f for feats in features_profile.values() for f in feats]))

# 生成数据
n_samples = 8000
data = []

for _ in range(n_samples):
    true_constitution = np.random.choice(constitutions, p=[0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])
    profile = features_profile[true_constitution]

    sample = {}
    for feat in all_features:
        if feat in profile:
            # 典型特征：均值较高，标准差0.7，限制1-5
            score = np.clip(np.random.normal(profile[feat], 0.7), 1, 5)
        else:
            # 非典型特征：均值2.5，标准差1.0，但有一定概率出现高分（模拟真实混淆）
            # 这里让均值略高，更真实
            score = np.clip(np.random.normal(2.5, 1.0), 1, 5)
        sample[feat] = round(score, 1)
    # 添加少量随机噪声：10%的概率随机改变一个特征的分数（模拟填错或个体差异）
    if np.random.rand() < 0.1:
        rand_feat = np.random.choice(all_features)
        sample[rand_feat] = np.clip(sample[rand_feat] + np.random.randint(-2, 3), 1, 5)
    sample['label'] = true_constitution
    data.append(sample)

df = pd.DataFrame(data)
cols = [c for c in df.columns if c != 'label'] + ['label']
df = df[cols]
df.to_csv('constitution_data.csv', index=False, encoding='utf-8-sig')
print(f"数据已生成，共{len(df)}条，特征数：{len(all_features)}")
print(df.head())