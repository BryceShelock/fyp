"""
Export Chatopera Emotional First Aid (efaqa-corpus-zh) records to this repo's multitask CSV.

Prerequisites:
  export EFAQA_DL_LICENSE=your_certificate_id  # non-empty certificate id (e.g. LTXxxxx)
  pip install efaqa-corpus-zh

Note: A local JSONL dump like dataset/efaqa-corpus-zh.utf8.txt is typically *raw* chat
(with message-level fields only). This exporter needs the official labeled corpus from
efaqa_corpus_zh.load(), which includes thread-level label.s1 / s2 / s3.

Official thread-level label (与数据集中 JSON 示例一致；若某文档表格把列名写反，以示例为准):
  - label.s1 → 烦恼类型 1.1–1.19  → CSV 列 s1 → 模型 S1 头
  - label.s2 → 心理疾病 2.1–2.8   → s2 → 模型 S2 头
  - label.s3 → SOS 3.1–3.6        → s3 → 模型 S3 头（紧急程度最高；连续 risk_score 主要用 S3 softmax）

话题标签默认依据咨询者初始陈述（title + description）；`chats` 内消息级 label 不参与本导出。

Text column: title + newline + description（与标注一致；无 description 时仅用 title）。

Output columns: text, s1, s2, s3  (integer indices 0..18, 0..7, 0..5)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from multitask_config import efa_tag_to_s1_index, efa_tag_to_s2_index, efa_tag_to_s3_index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="dataset_efaqa_multitask.csv")
    p.add_argument("--limit", type=int, default=0, help="0 = all records")
    args = p.parse_args()

    if not os.environ.get("EFAQA_DL_LICENSE"):
        print("Set environment variable EFAQA_DL_LICENSE before running.", file=sys.stderr)
        sys.exit(1)

    try:
        import efaqa_corpus_zh
    except ImportError as e:
        print("pip install efaqa-corpus-zh", file=sys.stderr)
        raise e

    records = list(efaqa_corpus_zh.load())
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    rows_out = 0
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "s1", "s2", "s3", "md5"])
        w.writeheader()
        for rec in records:
            label = rec.get("label") or {}
            s1t = str(label.get("s1", "")).strip()
            s2t = str(label.get("s2", "")).strip()
            s3t = str(label.get("s3", "")).strip()
            i1 = efa_tag_to_s1_index(s1t)
            i2 = efa_tag_to_s2_index(s2t)
            i3 = efa_tag_to_s3_index(s3t)
            if i1 is None or i2 is None or i3 is None:
                continue
            title = str(rec.get("title") or "").strip()
            desc = str(rec.get("description") or "").strip()
            text = f"{title}\n{desc}".strip() if desc else title
            if not text:
                continue
            w.writerow(
                {
                    "text": text,
                    "s1": i1,
                    "s2": i2,
                    "s3": i3,
                    "md5": rec.get("md5", ""),
                }
            )
            rows_out += 1

    print(f"Wrote {rows_out} rows to {args.output}")


if __name__ == "__main__":
    main()
