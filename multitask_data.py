"""
Load multitask CSV for training/eval.

Columns:
- Explicit: text, s1, s2, s3
  - s1: integer class 0..18, or EFA tag \"1.13\", or JSON one-hot list of length 19 (argmax),
        or legacy length-8 multi-hot (mapped to one EFA S1 class).
  - s2: 0..7 or \"2.7\"
  - s3: 0..5 or \"3.4\"
- Weibo: text, label (sentiment 0–3) -> weak labels via map_s1/map_s2/map_s3
"""

from __future__ import annotations

import ast
import json
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from multitask_config import (
    map_s1,
    map_s2,
    map_s3,
    S1_LABELS,
    S2_LABELS,
    S3_LABELS,
    efa_tag_to_s1_index,
    efa_tag_to_s2_index,
    efa_tag_to_s3_index,
)

# Legacy 8-dim multihot (old repo) -> single EFA S1 index
_LEGACY_8_TO_S1 = [0, 1, 2, 8, 7, 12, 6, 15]


def _legacy_8_multihot_to_s1(parsed: List) -> int:
    for i in range(min(8, len(parsed))):
        try:
            v = float(parsed[i])
        except (TypeError, ValueError):
            continue
        if v >= 0.5:
            return _LEGACY_8_TO_S1[i]
    return 15


def _parse_s1_cell(val: Any) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 15
    if isinstance(val, (int, np.integer)):
        v = int(val)
        return v if 0 <= v < len(S1_LABELS) else 15
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return 15
        idx = efa_tag_to_s1_index(s)
        if idx is not None:
            return idx
        try:
            v = int(s)
            return v if 0 <= v < len(S1_LABELS) else 15
        except ValueError:
            pass
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
            except (SyntaxError, ValueError):
                return 15
    elif isinstance(val, (list, tuple)):
        parsed = val
    else:
        return 15

    if isinstance(parsed, (list, tuple)):
        if len(parsed) == len(S1_LABELS):
            best = 0
            best_v = -1.0
            for i, x in enumerate(parsed):
                try:
                    fv = float(x)
                except (TypeError, ValueError):
                    continue
                if fv > best_v:
                    best_v = fv
                    best = i
            return best if best_v > 0 else 15
        if len(parsed) == 8:
            return _legacy_8_multihot_to_s1(list(parsed))
    return 15


def _coerce_s2_task_id(raw: Any) -> int:
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return 6
        idx = efa_tag_to_s2_index(s)
        if idx is not None:
            return idx
        if s.isdigit():
            n = int(s)
            if n == 5:
                return 6
            if 0 <= n < len(S2_LABELS):
                return n
        return 6
    if isinstance(raw, (int, np.integer)):
        n = int(raw)
        if n == 5:
            return 6
        if 0 <= n < len(S2_LABELS):
            return n
    return 6


def _coerce_s3_task_id(raw: Any) -> int:
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return 5
        idx = efa_tag_to_s3_index(s)
        if idx is not None:
            return idx
        if s.isdigit():
            n = int(s)
            if 0 <= n < len(S3_LABELS):
                return n
            legacy = {1: 1, 2: 2, 4: 5}
            if n in legacy:
                return legacy[n]
        return 5
    if isinstance(raw, (int, np.integer)):
        n = int(raw)
        if 0 <= n < len(S3_LABELS):
            return n
        legacy = {1: 1, 2: 2, 4: 5}
        if n in legacy:
            return legacy[n]
    return 5


def load_multitask_supervision(df: pd.DataFrame) -> Tuple[List[str], List[int], List[int], List[int], str]:
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
        "CSV needs columns (text, label) or (text, s1, s2, s3). "
        f"Got: {list(df.columns)}"
    )
