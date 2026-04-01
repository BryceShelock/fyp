"""
Label space aligned with EFA (Emotional First Aid / 心理咨询问答语料库) thread-level taxonomy.

S1: 19-class single-label (烦恼类型 1.1–1.19)
S2: 8-class single-label (心理疾病 2.1–2.8)
S3: 6-class single-label (SOS 3.1–3.6)

Indices 0..4 of S3 require mandatory human handoff (suicide action/ideation, self-harm, harm to others).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Union

# 合并 LoRA 后的默认可推理目录（相对仓库根，与正式训练 `--output_dir multitask_output_efa` 对应）
DEFAULT_INFERENCE_MODEL_DIR = "multitask_output_efa/best_model"

# --- S1: EFA 1.1 .. 1.19 (single-label class index = list index) ---
S1_LABELS = [
    "efa_s1_1_1_academic",  # 1.1
    "efa_s1_1_2_career",
    "efa_s1_1_3_family",
    "efa_s1_1_4_substance",
    "efa_s1_1_5_grief",
    "efa_s1_1_6_insomnia",
    "efa_s1_1_7_stress",
    "efa_s1_1_8_interpersonal",
    "efa_s1_1_9_relationship",
    "efa_s1_1_10_divorce",
    "efa_s1_1_11_breakup",
    "efa_s1_1_12_self_awareness",
    "efa_s1_1_13_low_self_esteem",
    "efa_s1_1_14_adolescent",
    "efa_s1_1_15_ocd",
    "efa_s1_1_16_other",
    "efa_s1_1_17_lgbt",
    "efa_s1_1_18_sex",
    "efa_s1_1_19_parent_child",
]

# --- S2: EFA 2.1 .. 2.8 ---
S2_LABELS = [
    "efa_s2_depression",  # 2.1
    "efa_s2_anxiety",
    "efa_s2_bipolar",
    "efa_s2_ptsd",
    "efa_s2_panic",
    "efa_s2_eating_disorder",
    "efa_s2_not_disorder",  # 2.7 尚未达到 S2
    "efa_s2_other_disorder",  # 2.8
]

# --- S3: EFA 3.1 .. 3.6 ---
S3_LABELS = [
    "efa_s3_suicide_action",  # 3.1
    "efa_s3_suicide_ideation",  # 3.2
    "efa_s3_self_harm",  # 3.3
    "efa_s3_harm_others_ongoing",  # 3.4
    "efa_s3_harm_others_planned",  # 3.5
    "efa_s3_none",  # 3.6
]

# Mandatory human takeover for these S3 class indices (3.1–3.5).
S3_FORCE_HANDOFF_INDICES: Set[int] = {0, 1, 2, 3, 4}

EFA_S1_TAG_TO_INDEX: Dict[str, int] = {f"1.{m}": idx for idx, m in enumerate(range(1, 20))}
EFA_S2_TAG_TO_INDEX: Dict[str, int] = {f"2.{m}": idx for idx, m in enumerate(range(1, 9))}
EFA_S3_TAG_TO_INDEX: Dict[str, int] = {f"3.{m}": idx for idx, m in enumerate(range(1, 7))}

S1_ID2ZH = {
    "efa_s1_1_1_academic": "学业/规划",
    "efa_s1_1_2_career": "事业与工作",
    "efa_s1_1_3_family": "家庭与矛盾",
    "efa_s1_1_4_substance": "物质滥用与成瘾",
    "efa_s1_1_5_grief": "悲恸",
    "efa_s1_1_6_insomnia": "失眠",
    "efa_s1_1_7_stress": "压力",
    "efa_s1_1_8_interpersonal": "人际关系",
    "efa_s1_1_9_relationship": "情感关系",
    "efa_s1_1_10_divorce": "离婚",
    "efa_s1_1_11_breakup": "分手",
    "efa_s1_1_12_self_awareness": "自我探索",
    "efa_s1_1_13_low_self_esteem": "低自尊",
    "efa_s1_1_14_adolescent": "青春期问题",
    "efa_s1_1_15_ocd": "强迫症",
    "efa_s1_1_16_other": "其它烦恼",
    "efa_s1_1_17_lgbt": "LGBT+",
    "efa_s1_1_18_sex": "性问题",
    "efa_s1_1_19_parent_child": "亲子关系",
}

S2_ID2ZH = {
    "efa_s2_depression": "抑郁（疑似）",
    "efa_s2_anxiety": "焦虑（疑似）",
    "efa_s2_bipolar": "躁郁（疑似）",
    "efa_s2_ptsd": "创伤后应激（疑似）",
    "efa_s2_panic": "恐慌（疑似）",
    "efa_s2_eating_disorder": "进食障碍（疑似）",
    "efa_s2_not_disorder": "尚未达到心理疾病层面",
    "efa_s2_other_disorder": "其它心理疾病（疑似）",
}

S3_ID2ZH = {
    "efa_s3_suicide_action": "正在进行的自杀行为",
    "efa_s3_suicide_ideation": "策划/意念自杀",
    "efa_s3_self_harm": "自残",
    "efa_s3_harm_others_ongoing": "对他人正在人身伤害",
    "efa_s3_harm_others_planned": "计划对他人人身伤害",
    "efa_s3_none": "无紧急身体伤害倾向",
}


def efa_tag_to_s1_index(tag: str) -> Optional[int]:
    t = (tag or "").strip()
    return EFA_S1_TAG_TO_INDEX.get(t)


def efa_tag_to_s2_index(tag: str) -> Optional[int]:
    t = (tag or "").strip()
    return EFA_S2_TAG_TO_INDEX.get(t)


def efa_tag_to_s3_index(tag: str) -> Optional[int]:
    t = (tag or "").strip()
    return EFA_S3_TAG_TO_INDEX.get(t)


def _any_hit(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def map_s1_weak_single(text: str) -> int:
    """
    Weak supervision: map social text to one S1 class (for Weibo / unlabeled expansion).
    Returns class index 0..len(S1_LABELS)-1.
    """
    raw = text if isinstance(text, str) else str(text)
    compact = "".join(raw.split())

    def hit(keys: list[str]) -> bool:
        return _any_hit(raw, keys) or _any_hit(compact, keys)

    if hit(["酗酒", "吸毒", "赌博", "烟瘾", "戒酒", "复吸"]):
        return 3
    if hit(["去世", "离世", "过世", "走了", "亲人离开", "丧亲"]):
        return 4
    if hit(
        [
            "作业",
            "考试",
            "论文",
            "毕设",
            "考研",
            "复习",
            "挂科",
            "课",
            "老师",
            "导师",
            "学分",
        ]
    ):
        return 0
    if hit(
        [
            "工作",
            "上班",
            "加班",
            "老板",
            "同事",
            "面试",
            "辞职",
            "离职",
            "工资",
            "裁员",
            "实习",
            "项目",
        ]
    ):
        return 1
    if hit(["爸妈", "父母", "家人", "亲戚", "妈妈", "爸爸", "婆婆", "儿媳", "孩子", "儿子", "女儿"]):
        return 2
    if hit(["离婚", "抚养权", "前夫", "前妻"]):
        return 9
    if hit(
        [
            "男朋友",
            "女朋友",
            "老公",
            "老婆",
            "分手",
            "复合",
            "恋爱",
            "劈腿",
            "出轨",
            "结婚",
            "异地恋",
        ]
    ):
        return 8
    if hit(["朋友", "同学", "室友", "社交", "孤立", "排挤", "拉黑", "吵架", "闹掰"]):
        return 7
    if hit(["失眠", "睡不着", "早醒", "入睡困难"]):
        return 5
    if hit(
        [
            "没用",
            "废物",
            "自卑",
            "讨厌自己",
            "恨自己",
            "愧疚",
            "丢脸",
            "不够好",
            "不长心眼",
            "啥也不行",
        ]
    ):
        return 12
    if hit(
        [
            "压力",
            "崩溃",
            "烦",
            "烦躁",
            "焦虑",
            "抑郁",
            "难受",
            "撑不住",
            "喘不过气",
            "累",
        ]
    ):
        return 6
    if hit(["同性恋", "双性恋", "跨性别", "出柜", "LGBT"]):
        return 16
    if hit(["性教育", "性焦虑", "性生活"]):
        return 17
    if hit(["青春期", "叛逆", "早恋"]):
        return 13
    if hit(["强迫症", "反复想", "反复检查"]):
        return 14
    if hit(["星座", "性格测试", "MBTI", "兴趣测评"]):
        return 11
    if hit(["心累", "迷茫", "无力", "绝望", "痛苦", "郁闷", "糟", "好难"]):
        return 15
    return 15


def map_s2_weibo_sentiment(weibo_sent_label: int) -> int:
    """Weibo sentiment 0–3 -> EFA S2 index."""
    if weibo_sent_label == 3:
        return 0
    if weibo_sent_label == 1:
        return 1
    return 6


def map_s3_weak(text: str) -> int:
    """Weak S3 for social text -> EFA S3 index."""
    raw = text if isinstance(text, str) else str(text)
    compact = "".join(raw.split())

    def hit(keys: list[str]) -> bool:
        return _any_hit(raw, keys) or _any_hit(compact, keys)

    if hit(["正在割腕", "已经跳了", "喝药了", "上吊了"]):
        return 0
    if hit(["杀了他", "砍死", "捅死", "弄死他", "打死他"]):
        return 3
    if hit(["想杀", "准备杀", "计划杀", "要买刀杀"]):
        return 4
    if hit(["不想活", "活不下去", "一了百了", "想死", "去死", "结束生命", "自杀算了"]):
        return 1
    if hit(["割腕", "自残", "轻生", "跳楼", "吞药", "刀片"]):
        return 2
    return 5


# Backward-compatible names for weak-supervision loaders (Weibo CSV path).
map_s1 = map_s1_weak_single
map_s2 = map_s2_weibo_sentiment
map_s3 = map_s3_weak


def s3_to_risk_level(s3_pred_id: int) -> str:
    """Coarse display tier from discrete S3 (before risk_score fusion)."""
    if s3_pred_id == 5:
        return "低风险"
    if s3_pred_id == 1:
        return "中风险"
    return "高风险"


def format_prediction(
    s1_pred_id: Union[int, List[int]],
    s2_pred_id: int,
    s3_pred_id: int,
    s1_threshold: float = 0.15,
):
    """
    Build UI-facing dict. s1 is single-class index (int) or legacy list of ids (first used).
    s1_threshold is ignored for single-label S1 (kept for API compatibility).
    """
    if isinstance(s1_pred_id, list):
        s1_idx = int(s1_pred_id[0]) if s1_pred_id else 0
    else:
        s1_idx = int(s1_pred_id)
    if not (0 <= s1_idx < len(S1_LABELS)):
        s1_idx = min(max(s1_idx, 0), len(S1_LABELS) - 1) if S1_LABELS else 0

    s2_idx = int(s2_pred_id)
    if not (0 <= s2_idx < len(S2_LABELS)):
        s2_idx = min(max(s2_idx, 0), len(S2_LABELS) - 1) if S2_LABELS else 0
    s3_idx = int(s3_pred_id)
    if not (0 <= s3_idx < len(S3_LABELS)):
        s3_idx = min(max(s3_idx, 0), len(S3_LABELS) - 1) if S3_LABELS else 0

    zh1 = S1_ID2ZH.get(S1_LABELS[s1_idx], S1_LABELS[s1_idx])
    return {
        "问题类型": [zh1],
        "心理状态": S2_ID2ZH[S2_LABELS[s2_idx]],
        "风险等级": s3_to_risk_level(s3_idx),
    }

