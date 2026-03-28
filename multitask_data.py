"""
多任务训练/评测共用：从 CSV 读取 (text, 标签)，支持两种列格式。

1) 微博弱监督格式：列包含 text, label（情感 0–3），S1/S3 由规则 map_s1/map_s3 生成，S2 由 map_s2(label)。
2) 显式多任务格式：列包含 text, s1, s2, s3（与 dataset_multitask.csv 一致）。
   - s1 可为 JSON 字符串如 "[0,1,0,...]" 或等长列表。
   - 兼容遗留编码：chat 默认 s2=5 视为 none(2)；s3 使用 1/2/4 映射到 0/1/2。
"""

from __future__ import annotations

import ast
import json
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from multitask_config import map_s1, map_s2, map_s3, S1_LABELS, S2_LABELS, S3_LABELS


def _parse_s1_cell(val: Any) -> List[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return [0.0] * len(S1_LABELS)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return [0.0] * len(S1_LABELS)
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
            except (SyntaxError, ValueError):
                return [0.0] * len(S1_LABELS)
    elif isinstance(val, (list, tuple)):
        parsed = val
    else:
        return [0.0] * len(S1_LABELS)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(S1_LABELS):
        return [0.0] * len(S1_LABELS)
    out = []
    for x in parsed:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _coerce_s2_task_id(raw: Any) -> int:
    """将 CSV 中的 S2 收束到 [0, len(S2_LABELS))。"""
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return len(S2_LABELS) - 1
    if 0 <= v < len(S2_LABELS):
        return v
    # build_training_dataset 中 chat 默认 s2=5 表示 none → 索引 2
    if v == 5:
        return 2
    return min(max(v, 0), len(S2_LABELS) - 1)


def _coerce_s3_task_id(raw: Any) -> int:
    """将 CSV 中的 S3 收束到 [0, len(S3_LABELS))；兼容 1/2/4 遗留编码。"""
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return len(S3_LABELS) - 1
    if 0 <= v < len(S3_LABELS):
        return v
    # 常见遗留：1=意念, 2=自伤, 4=无风险（与 map_s3 的 0/1/2 对齐）
    legacy = {1: 0, 2: 1, 4: 2}
    if v in legacy:
        return legacy[v]
    return len(S3_LABELS) - 1


def load_multitask_supervision(df: pd.DataFrame) -> Tuple[List[str], List[List[float]], List[int], List[int], str]:
    """
    Returns:
        texts, labels_s1 (multi-hot list per row), labels_s2, labels_s3, format_name
    """
    # pandas columns case-sensitive；dataset_multitask.csv 使用 text,s1,s2,s3
    if "s1" in df.columns and "s2" in df.columns and "s3" in df.columns:
        texts = df["text"].astype(str).tolist()
        labels_s1 = [_parse_s1_cell(x) for x in df["s1"].tolist()]
        labels_s2 = [_coerce_s2_task_id(x) for x in df["s2"].tolist()]
        labels_s3 = [_coerce_s3_task_id(x) for x in df["s3"].tolist()]
        return texts, labels_s1, labels_s2, labels_s3, "explicit_s123"

    if "label" in df.columns and "text" in df.columns:
        texts = df["text"].astype(str).tolist()
        weibo_labels = df["label"].tolist()
        labels_s1 = [map_s1(t) for t in texts]
        labels_s2 = [map_s2(int(x)) for x in weibo_labels]
        labels_s3 = [map_s3(t) for t in texts]
        return texts, labels_s1, labels_s2, labels_s3, "weibo_label"

    raise ValueError(
        "CSV 缺少可用列：需要 (text, label) 或 (text, s1, s2, s3)。"
        f"当前列：{list(df.columns)}"
    )
