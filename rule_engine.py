# rule_engine.py
# 基于《中医体质分类与判定》标准的规则评分系统

def rule_based_prediction(user_input, le, threshold=0.6):
    """
    规则引擎：根据国标评分规则判定体质。
    user_input: dict {feature_name: score}  (1-5)
    le: label_encoder (用于转换体质名称)
    threshold: 判定阈值（最高分占比）
    """
    # 预定义每种体质的核心特征及权重（简化：所有核心特征等权）
    constitution_features = {
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

    scores = {}
    for constitution, features in constitution_features.items():
        # 计算该体质核心特征的平均分
        feat_scores = [user_input.get(f, 3) for f in features]
        avg = sum(feat_scores) / len(feat_scores)
        scores[constitution] = avg

    # 最高分体质
    best = max(scores, key=scores.get)
    max_score = scores[best]
    # 归一化最高分（1-5分制）作为置信度
    confidence = (max_score - 1) / 4
    return best, confidence