import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 页面配置
st.set_page_config(page_title="中医体质辨识系统", layout="wide")
st.title("🌿 基于人工智能的中医体质辨识系统")
st.markdown("请根据您最近一个月的感觉，为每个问题打分（1=完全没有，5=非常严重）")

@st.cache_resource
def load_model():
    model = joblib.load('xgb_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

model, le = load_model()

df_sample = pd.read_csv('constitution_data.csv', encoding='utf-8-sig')
feature_names = [c for c in df_sample.columns if c != 'label']

@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(model)

explainer = load_explainer()

# 养生建议字典（带出处）
health_advice = {
    '平和质': '✅ 继续保持：均衡饮食，规律作息，适度运动。\\n📚 依据：《中医体质学》王琦（2005）',
    '气虚质': '🌾 建议：多吃山药、大枣、黄芪；避免过度劳累，适当练习八段锦。\\n📚 依据：《中医体质分类与判定》ZYYXH/T157-2009',
    '阳虚质': '🔥 建议：多吃羊肉、生姜、韭菜；注意保暖，艾灸足三里。\\n📚 依据：《中医体质分类与判定》ZYYXH/T157-2009',
    '阴虚质': '💧 建议：多吃百合、银耳、枸杞；避免熬夜，少吃辛辣。\\n📚 依据：《中医体质分类与判定》ZYYXH/T157-2009',
    '痰湿质': '🏃 建议：多吃薏米、赤小豆；坚持运动，避免油腻甜食。\\n📚 依据：《中医体质分类与判定》ZYYXH/T157-2009',
    '湿热质': '🍵 建议：多吃绿豆、苦瓜；保持居住环境干燥，戒烟限酒。\\n📚 依据：《中医体质分类与判定》ZYYXH/T157-2009',
    '血瘀质': '🌸 建议：多吃山楂、黑木耳；适当按摩，保持心情舒畅。\\n📚 依据：《中医体质分类与判定》ZYYXH/T157-2009',
    '气郁质': '😊 建议：多吃玫瑰花、佛手；多参加社交活动，练习冥想。\\n📚 依据：《中医体质分类与判定》ZYYXH/T157-2009',
    '特禀质': '⚠️ 建议：避免过敏原，多吃富含维生素C的食物，必要时就医。\\n📚 依据：《中医体质分类与判定》ZYYXH/T157-2009'
}

# 生成问卷输入控件
col1, col2 = st.columns(2)
user_input = {}
for i, feat in enumerate(feature_names):
    with col1 if i % 2 == 0 else col2:
        user_input[feat] = st.slider(feat, 1, 5, 3, key=feat)

if st.button("🔍 开始辨识", type="primary"):
    input_df = pd.DataFrame([user_input])
    pred_encoded = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(input_df)[0]

    confidence = max(proba)
    st.success(f"### 您的体质类型：**{pred_label}**（置信度：{confidence:.1%}）")
    st.caption("置信度表示模型对该判断的确信程度，数值越高说明特征越典型。")

    # 概率条形图
    st.subheader("📊 各类体质概率")
    proba_dict = {le.classes_[i]: proba[i] for i in range(len(le.classes_))}
    proba_df = pd.DataFrame(proba_dict.items(), columns=["体质", "概率"])
    st.bar_chart(proba_df.set_index("体质"))

    # SHAP 瀑布图
    st.subheader("🔍 模型决策解释（SHAP分析）")
    st.markdown("下图展示了每个问题对预测结果的贡献：**红色**推动向该体质，**蓝色**则相反。")

    shap_values = explainer.shap_values(input_df)
    pred_idx = pred_encoded

    # 处理多分类 SHAP 输出
    if isinstance(shap_values, list):
        # 多分类：列表形式
        shap_values_sample = shap_values[pred_idx][0]
        expected_value = explainer.expected_value[pred_idx]
    else:
        # 可能是三维数组 (样本, 特征, 类别)
        shap_values_sample = shap_values[0, :, pred_idx]
        expected_value = explainer.expected_value[pred_idx]

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_sample,
            base_values=expected_value,
            data=input_df.iloc[0].values,
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # 养生建议
    st.subheader("📝 养生建议")
    st.info(health_advice.get(pred_label, "请咨询专业医师"))

    # 免责声明
    st.warning("⚠️ 本系统仅用于健康科普参考，不能替代专业医师诊断。如有健康问题请咨询中医师。")