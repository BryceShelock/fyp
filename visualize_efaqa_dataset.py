"""
EFA 多任务 CSV（或由 export_efaqa_to_csv.py 生成）的简易 EDA 图。

不调用 efaqa_corpus_zh.load()，仅读 CSV，避免许可证与大宗下载。

用法:
  python visualize_efaqa_dataset.py --csv dataset_efaqa_multitask.csv --out efaqa_eda.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multitask_config import S1_LABELS, S1_ID2ZH, S2_LABELS, S2_ID2ZH, S3_LABELS, S3_ID2ZH


def _short_s1(i: int) -> str:
    key = S1_LABELS[i] if 0 <= i < len(S1_LABELS) else str(i)
    zh = S1_ID2ZH.get(key, key)
    s = str(zh)
    return s[:8] if len(s) > 8 else s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="dataset_efaqa_multitask.csv")
    p.add_argument("--out", type=str, default="efaqa_eda.png")
    p.add_argument("--text_col", type=str, default="text")
    args = p.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8")
    need = {"s1", "s2", "s3"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV need columns {need}, got {list(df.columns)}")
    tc = args.text_col
    if tc not in df.columns:
        raise SystemExit(f"No column {tc}")

    s1 = df["s1"].astype(int)
    s2 = df["s2"].astype(int)
    s3 = df["s3"].astype(int)
    lens = df[tc].astype(str).str.len()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # S1 counts
    ax = axes[0, 0]
    c1 = np.bincount(s1.clip(0, 18), minlength=19)
    x = np.arange(19)
    ax.bar(x, c1, color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_s1(i) for i in x], rotation=75, ha="right", fontsize=7)
    ax.set_title(f"S1 烦恼类型 (n={len(df)})")
    ax.set_ylabel("count")

    ax = axes[0, 1]
    c2 = np.bincount(s2.clip(0, 7), minlength=8)
    ax.bar(range(8), c2, color="C1")
    ax.set_xticks(range(8))
    ax.set_xticklabels(
        [str(S2_ID2ZH.get(S2_LABELS[i], S2_LABELS[i]))[:10] for i in range(8)],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_title("S2 心理疾病(疑似)")
    ax.set_ylabel("count")

    ax = axes[1, 0]
    c3 = np.bincount(s3.clip(0, 5), minlength=6)
    ax.bar(range(6), c3, color="C2")
    ax.set_xticks(range(6))
    ax.set_xticklabels(
        [str(S3_ID2ZH.get(S3_LABELS[i], S3_LABELS[i]))[:12] for i in range(6)],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    ax.set_title("S3 SOS")
    ax.set_ylabel("count")

    ax = axes[1, 1]
    ax.hist(lens, bins=40, color="C3", alpha=0.85)
    ax.set_title(f"文本长度分布 ({tc} 字符数)")
    ax.set_xlabel("chars")
    ax.set_ylabel("count")

    fig.suptitle(os.path.abspath(args.csv), fontsize=9)
    out_path = os.path.abspath(args.out)
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(
        "S3 类占比:",
        {S3_LABELS[i]: int(c3[i]) for i in range(6)},
    )


if __name__ == "__main__":
    main()
