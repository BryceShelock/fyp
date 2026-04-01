from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

from multitask_config import S3_LABELS

# 将 S3 分布与均匀分布混合：p'=(1-ε)p+ε/6，缓解错误过置信导致的 model_base 顶满（默认 0 不改变行为）。
_DEFAULT_RISK_UNIFORM_BLEND = float(os.environ.get("MULTITASK_RISK_UNIFORM_BLEND", "0"))
# 在 uniform blend 之后，对「无紧急」概率放大阻尼系数，略抑误报（默认 0.10 与旧逻辑一致；可提到 0.15~0.25）。
_NONE_DAMP_SCALE = float(os.environ.get("MULTITASK_RISK_NONE_DAMP_SCALE", "0.10"))

# 连续 risk_score → 低/中/高档位；`multitask_predict` 里 S3 强制转人工的默认置信度与此对齐，避免两套阈值混用。
RISK_LEVEL_HIGH_THRESHOLD = 0.60
RISK_LEVEL_MID_THRESHOLD = 0.30


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

    # 1) base：由 S3 概率（softmax）计算（EFA 六类）；可选与均匀分布混合，降低 OOD 上过尖分布对加权和的冲击
    keys = S3_LABELS
    p = {k: float(s3_probs.get(k, 0.0)) for k in keys}
    blend_eps = min(max(_DEFAULT_RISK_UNIFORM_BLEND, 0.0), 0.95)
    if blend_eps > 0:
        inv = 1.0 / len(keys)
        p = {k: (1.0 - blend_eps) * p[k] + blend_eps * inv for k in keys}

    p_action = p.get(keys[0], 0.0)
    p_idea = p.get(keys[1], 0.0)
    p_self = p.get(keys[2], 0.0)
    p_harm_on = p.get(keys[3], 0.0)
    p_harm_plan = p.get(keys[4], 0.0)
    p_none = p.get(keys[5], 0.0)

    base = (
        1.00 * p_action
        + 0.72 * p_idea
        + 0.88 * p_self
        + 0.95 * p_harm_on
        + 0.85 * p_harm_plan
    )

    reasons: List[str] = []
    if blend_eps > 0:
        reasons.append(f"s3_uniform_blend={blend_eps:.3f}")
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

    # none 概率较高时，下调综合分（尺度可由 MULTITASK_RISK_NONE_DAMP_SCALE 调整）
    damp = max(0.0, _NONE_DAMP_SCALE) * p_none
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
    if score >= RISK_LEVEL_HIGH_THRESHOLD:
        level = "高风险"
    elif score >= RISK_LEVEL_MID_THRESHOLD:
        level = "中风险"
    else:
        level = "低风险"

    return RiskResult(score=score, level=level, reasons=reasons)

