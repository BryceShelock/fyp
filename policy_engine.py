from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from risk_scoring import RiskResult


@dataclass
class PolicyDecision:
    action: str  # "ai_reply" | "ai_reply+hotline" | "handoff_to_human"
    ui_tags: Dict[str, str]
    resources: List[Dict[str, str]]


def decide_policy(risk: RiskResult) -> PolicyDecision:
    """
    输入：risk_score + risk_level
    输出：系统动作（低/中/高风险不同策略） + UI 可展示资源
    """
    resources: List[Dict[str, str]] = []

    if risk.level == "低风险":
        return PolicyDecision(
            action="ai_reply",
            ui_tags={"风险等级": risk.level},
            resources=resources,
        )

    if risk.level == "中风险":
        # 中风险：在 AI 回复基础上给支持资源（你可按学校/地区替换为本地热线）
        resources.extend(
            [
                {"type": "hotline", "title": "心理援助热线（示例）", "value": "当地心理援助热线 / 学校心理中心电话"},
                {"type": "link", "title": "紧急求助建议", "value": "https://www.who.int/zh/news-room/fact-sheets/detail/suicide"},
            ]
        )
        return PolicyDecision(
            action="ai_reply+hotline",
            ui_tags={"风险等级": risk.level},
            resources=resources,
        )

    # 高风险：真人接管
    resources.extend(
        [
            {"type": "warning", "title": "高风险提示", "value": "已触发高风险流程：将转接管理员/真人支持。"},
            {"type": "hotline", "title": "紧急求助", "value": "如有立即危险请联系当地紧急电话/身边可信任的人"},
        ]
    )
    return PolicyDecision(
        action="handoff_to_human",
        ui_tags={"风险等级": risk.level},
        resources=resources,
    )

