S1_LABELS = [
    "academic",      # 学业
    "career",        # 工作
    "family",        # 家庭
    "relationship",  # 情感
    "social",        # 人际
    "self",          # 自我/低自尊
    "stress",        # 压力
    "other",         # 其他
]

S2_LABELS = [
    "depression",  # 抑郁
    "anxiety",     # 焦虑
    "none",        # 无明显
]

S3_LABELS = [
    "suicide_ideation",   # 自杀意念
    "self_harm",          # 自伤
    "none",               # 无风险
]

S1_ID2ZH = {
    "academic": "学业",
    "career": "工作",
    "family": "家庭",
    "relationship": "情感",
    "social": "人际",
    "self": "自我/低自尊",
    "stress": "压力",
    "other": "其他",
}

S2_ID2ZH = {
    "depression": "抑郁",
    "anxiety": "焦虑",
    "none": "无明显心理状态",
}

S3_ID2ZH = {
    "suicide_ideation": "自杀意念",
    "self_harm": "自伤",
    "none": "无风险",
}


def _any_hit(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def map_s1(text: str):
    """
    S1 多标签：弱规则 + 扩展词表。

    注意：微博清洗语料常见「字间空格 / 分词空格」，直接用 ``"作业" in text`` 往往永远匹配不到，
    会导致 S1 全 0。这里同时匹配：
    - 原始字符串
    - 去掉空白后的拼接串（例如 ``作 业`` -> ``作业``）
    """
    raw = text if isinstance(text, str) else str(text)
    compact = "".join(raw.split())

    def hit(keys: list[str]) -> bool:
        return _any_hit(raw, keys) or _any_hit(compact, keys)

    s1 = [0] * len(S1_LABELS)

    # academic
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
        s1[0] = 1

    # career
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
        s1[1] = 1

    # family
    if hit(["爸妈", "父母", "家人", "亲戚", "妈妈", "爸爸", "婆婆", "儿媳", "孩子", "儿子", "女儿"]):
        s1[2] = 1

    # relationship（恋爱/婚姻）
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
            "离婚",
            "结婚",
            "异地恋",
        ]
    ):
        s1[3] = 1

    # social
    if hit(
        [
            "朋友",
            "同学",
            "室友",
            "社交",
            "孤立",
            "排挤",
            "拉黑",
            "吵架",
            "闹掰",
        ]
    ):
        s1[4] = 1

    # self（自我/低自尊）
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
        ]
    ):
        s1[5] = 1

    # stress（压力/情绪负荷；与 academic 可并存）
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
            "失眠",
            "累",
        ]
    ):
        s1[6] = 1

    # other：若已有任一维度则不必标 other；若仍为全 0，用“泛化困扰词”给弱正例，避免全 0 标签塌缩
    if sum(s1) == 0 and hit(["心累", "迷茫", "无力", "绝望", "痛苦", "郁闷", "糟", "好难"]):
        s1[7] = 1

    return s1


def map_s2(weibo_sent_label: int):
    """
    S2单标签：微博情感 -> 心理状态（你给的映射规则）。
    数据集微博情感label来自 README：
      0 喜悦, 1 愤怒, 2 厌恶, 3 低落
    """
    if weibo_sent_label == 3:  # 低落 -> depression
        return 0
    if weibo_sent_label == 1:  # 愤怒 -> anxiety
        return 1
    return 2  # none


def map_s3(text: str):
    """
    S3单标签：风险检测（你给的映射规则，先实现关键两类 + none）。
    """
    raw = text if isinstance(text, str) else str(text)
    compact = "".join(raw.split())

    def hit(keys: list[str]) -> bool:
        return _any_hit(raw, keys) or _any_hit(compact, keys)

    if hit(["不想活", "活不下去", "一了百了"]):
        return 0  # suicide_ideation
    if hit(["割腕", "自残", "自杀", "轻生"]):
        return 1  # self_harm

    return 2  # none


def s3_to_risk_level(s3_pred_id: int):
    """
    给前端更易读的风险等级（用于展示）。
    规则：意念->中风险，行为/自伤/伤他->高风险，其它->低风险
    """
    if s3_pred_id == 2:  # none
        return "低风险"
    if s3_pred_id == 0:  # suicide_ideation
        return "中风险"
    return "高风险"  # self_harm


def format_prediction(s1_pred_ids, s2_pred_id, s3_pred_id, s1_threshold: float = 0.15):
    """
    将模型输出转成前端需要的结构。
    """
    problem_types = [S1_ID2ZH[S1_LABELS[i]] for i in s1_pred_ids]

    # 去重 + 排序（避免阈值导致乱序）
    problem_types = sorted(set(problem_types))

    return {
        "问题类型": problem_types if problem_types else ["其他"],
        "心理状态": S2_ID2ZH[S2_LABELS[s2_pred_id]],
        "风险等级": s3_to_risk_level(s3_pred_id),
    }

