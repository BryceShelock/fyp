from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

from multitask_config import S3_LABELS


@dataclass
class RiskResult:
    score: float
    level: str  # "低风险" | "中风险" | "高风险"
    reasons: List[str]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_risk_score(
    text: str,
    s3_probs: Dict[str, float],
    history_scores: Optional[List[float]] = None,
) -> RiskResult:
    """
    连续 risk_score（0~1）= 模型概率 + 规则加分 + 历史累积

    - **模型概率**：对高危类概率加权求和
    - **规则加分**：对明确高危短语给 boost（防止模型欠拟合时漏检）
    - **历史累积**：最近多条的指数滑动（更符合产品风险管理）
    """
    raw = text or ""
    compact = "".join(raw.split())

    def hit(keys: List[str]) -> bool:
        return any(k in raw for k in keys) or any(k in compact for k in keys)

    # 1) base：由 S3 概率（softmax）计算
    p_action = float(s3_probs.get("suicide_action", 0.0))
    p_idea = float(s3_probs.get("suicide_ideation", 0.0))
    p_self = float(s3_probs.get("self_harm", 0.0))
    p_others = float(s3_probs.get("harm_others", 0.0))
    p_none = float(s3_probs.get("none", 0.0))

    base = (
        1.00 * p_action
        + 0.70 * p_idea
        + 0.90 * p_self
        + 0.90 * p_others
    )

    reasons: List[str] = []
    reasons.append(f"model_base={base:.3f}")

    # 2) rule boosts：明确表达优先（可在论文中解释为“安全优先的规则兜底”）
    boost = 0.0

    # 意念
    if hit(["不想活", "活不下去", "想死", "去死", "一了百了", "不如死了"]):
        boost = max(boost, 0.20)
        reasons.append("rule:suicide_ideation_phrase")

    # 自伤/自杀方式
    if hit(["割腕", "上吊", "跳楼", "吞药", "自残", "轻生", "刀片", "安眠药"]):
        boost = max(boost, 0.35)
        reasons.append("rule:self_harm_method")

    # 伤害他人
    if hit(["杀了他", "杀人", "砍死", "同归于尽", "报复社会", "炸了"]):
        boost = max(boost, 0.35)
        reasons.append("rule:harm_others_phrase")

    # 极端绝望（低阈值加一点点，避免过度触发）
    if hit(["绝望", "崩溃", "撑不住", "受不了了", "没有意义", "活着没意思"]):
        boost = max(boost, 0.10)
        reasons.append("rule:hopelessness_phrase")

    # none 概率很高时，略微下调（防止 base 太敏感）
    damp = 0.10 * p_none
    if damp > 0:
        reasons.append(f"none_damp={damp:.3f}")

    score_now = _clamp01(base + boost - damp)

    # 3) history：指数滑动（最近越近权重越大）
    if history_scores:
        # alpha 越大越看重历史（这里取 0.4，偏向近期）
        alpha = 0.4
        hist = 0.0
        w = 1.0
        norm = 0.0
        # 只取最后 20 条，避免无限增长
        for s in history_scores[-20:][::-1]:
            hist += w * float(s)
            norm += w
            w *= (1.0 - alpha)
        hist = hist / max(norm, 1e-9)
        # 用 sigmoid 把“当前 + 历史”压到 0~1，避免堆叠超过 1
        fused = _sigmoid(2.2 * (0.65 * score_now + 0.35 * hist) - 1.0)
        reasons.append(f"history_fused={fused:.3f}")
        score = _clamp01(fused)
    else:
        score = score_now

    # 4) level：映射为低/中/高
    if score >= 0.60:
        level = "高风险"
    elif score >= 0.30:
        level = "中风险"
    else:
        level = "低风险"

    return RiskResult(score=score, level=level, reasons=reasons)

