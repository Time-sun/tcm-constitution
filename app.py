import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# ========== 页面配置 ==========
st.set_page_config(page_title="中医体质辨识系统", layout="wide")

# ========== 自定义CSS（绿色毛玻璃风格） ==========
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    .stSlider > div, .stSuccess, .stInfo, .stWarning {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(10px);
        border-radius: 28px !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.05) !important;
        transition: all 0.3s ease;
    }
    .stButton > button {
        background: linear-gradient(105deg, #2e7d32, #4caf50) !important;
        color: white !important;
        border-radius: 60px !important;
        font-weight: 600;
        border: none;
        box-shadow: 0 8px 20px rgba(46,125,50,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(105deg, #1b5e20, #388e3c) !important;
        box-shadow: 0 12px 25px rgba(46,125,50,0.4);
    }
    h1 {
        font-size: 2.2rem;
        background: linear-gradient(120deg, #1b5e20, #4caf50);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .sidebar-content {
        background: rgba(232,245,233,0.9);
        border-radius: 28px;
        padding: 1rem;
    }
    .stSuccess {
        border-left: 6px solid #2e7d32 !important;
        background: #e8f5e9 !important;
    }
    .stInfo {
        border-left: 6px solid #4caf50 !important;
        background: #e8f5e9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("## 🌿 系统简介")
    st.markdown("""
    本系统基于《中医体质分类与判定》标准，采用 **XGBoost** 机器学习模型，结合 **SHAP** 可解释性分析，为您提供客观、可解释的中医体质辨识服务。

    ### 📖 使用说明
    1. 根据最近一个月的感觉，为每个症状打分（1=完全没有，5=非常严重）
    2. 点击「开始辨识」
    3. 查看您的体质类型、置信度、雷达图及 SHAP 解释
    4. 参考养生建议调整生活方式

    ### 📞 联系
    江西中医药大学 · 医学信息工程专业
    """)
    st.markdown("---")
    st.caption("⚠️ 本系统仅用于健康科普，不能替代专业医师诊断。")


# ========== 加载模型和数据 ==========
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

# ========== 预定义数据 ==========
# 国标核心特征
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

# 养生建议
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


# ========== 规则引擎函数 ==========
def rule_based_prediction(user_input):
    """基于国标核心特征平均分的规则判定"""
    scores = {}
    for constitution, feats in gold_features.items():
        vals = [user_input.get(f, 3) for f in feats]
        scores[constitution] = sum(vals) / len(vals)
    best = max(scores, key=scores.get)
    confidence = (scores[best] - 1) / 4
    return best, confidence


# ========== 主区域：问卷输入 ==========
st.title("🌿 基于人工智能的中医体质辨识系统")
st.markdown("请根据您最近一个月的感觉，为每个问题打分（1=完全没有，5=非常严重）")

col1, col2 = st.columns(2)
user_input = {}
for i, feat in enumerate(feature_names):
    with col1 if i % 2 == 0 else col2:
        user_input[feat] = st.slider(feat, 1, 5, 3, key=feat)

# ========== 预测按钮 ==========
if st.button("🔍 开始辨识", type="primary"):
    # 将用户输入转为DataFrame
    input_df = pd.DataFrame([user_input])

    # XGBoost预测
    pred_encoded = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(input_df)[0]
    xgb_confidence = max(proba)

    # 规则引擎预测
    rule_label, rule_confidence = rule_based_prediction(user_input)

    # 混合决策：若XGBoost置信度低于0.8，则使用规则引擎结果
    if xgb_confidence < 0.8:
        final_label = rule_label
        final_confidence = rule_confidence
        decision_mode = "规则引擎辅助（XGBoost置信度较低）"
    else:
        final_label = pred_label
        final_confidence = xgb_confidence
        decision_mode = "XGBoost模型预测"

    # 显示结果
    st.success(f"### 您的体质类型：**{final_label}**（置信度：{final_confidence:.1%}）")
    st.caption(f"决策模式：{decision_mode} | 置信度越高，结果越可靠。")

    # 概率条形图（仍用XGBoost的概率）
    st.subheader("📊 各类体质概率（XGBoost）")
    proba_dict = {le.classes_[i]: proba[i] for i in range(len(le.classes_))}
    proba_df = pd.DataFrame(proba_dict.items(), columns=["体质", "概率"])
    st.bar_chart(proba_df.set_index("体质"))

    # 仪表盘 + 雷达图
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_confidence * 100,
            title={'text': "置信度 (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2e7d32"},
                'steps': [
                    {'range': [0, 50], 'color': '#ffebee'},
                    {'range': [50, 80], 'color': '#fff9c4'},
                    {'range': [80, 100], 'color': '#c8e6c9'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': final_confidence * 100}
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': '#1b5e20'})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_res2:
        st.subheader("九种体质倾向雷达图")
        categories = le.classes_.tolist()
        values = (proba * 100).tolist()
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values, theta=categories, fill='toself',
            marker=dict(color='#4caf50'), line=dict(color='#2e7d32', width=2)
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350,
                                paper_bgcolor='rgba(0,0,0,0)', font={'color': '#1b5e20'})
        st.plotly_chart(fig_radar, use_container_width=True)

    # SHAP瀑布图
    st.subheader("🔍 模型决策解释（SHAP分析）")
    st.markdown("下图展示了每个问题对预测结果的贡献：**红色**推动向该体质，**蓝色**则相反。")
    shap_values = explainer.shap_values(input_df)
    pred_idx = pred_encoded
    if isinstance(shap_values, list):
        shap_sample = shap_values[pred_idx][0]
        expected_value = explainer.expected_value[pred_idx]
    else:
        shap_sample = shap_values[0, :, pred_idx]
        expected_value = explainer.expected_value[pred_idx]
    shap_sample = np.array(shap_sample).flatten()
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_sample,
            base_values=expected_value,
            data=input_df.iloc[0].values,
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # 中医理论对照模块
    with st.expander("📖 中医理论依据（点击展开）"):
        st.markdown(f"""
        **依据《中医体质分类与判定》标准（ZYYXH/T157-2009）**：
        - **{final_label}的核心特征**：{', '.join(gold_features.get(final_label, ['无']))}
        - **模型决策依据**：SHAP瀑布图中**红色**特征推动判定，**蓝色**特征抑制判定。
        - **一致性说明**：模型判据与中医理论核心特征高度一致（平均重合度79.93%）。
        """)

    # 养生建议
    st.subheader("📝 养生建议")
    st.info(health_advice.get(final_label, "请咨询专业医师"))

    # 免责声明
    st.warning("⚠️ 本系统仅用于健康科普参考，不能替代专业医师诊断。")