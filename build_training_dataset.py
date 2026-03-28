import os
import csv
import json
import glob
import random
import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd

from multitask_config import (
    map_s1,
    map_s2,
    map_s3,
    S1_LABELS,
    S2_LABELS,
    S3_LABELS,
)


def _safe_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


class RunningStats:
    def __init__(self):
        self.n = 0
        self.s1_nonzero = 0
        self.s1_counts = [0] * len(S1_LABELS)
        self.s2_counts = [0] * len(S2_LABELS)
        self.s3_counts = [0] * len(S3_LABELS)

    def update(self, s1: List[int], s2: int, s3: int):
        self.n += 1
        if any(int(v) == 1 for v in s1):
            self.s1_nonzero += 1
        for i, v in enumerate(s1):
            self.s1_counts[i] += int(v)
        self.s2_counts[int(s2)] += 1
        self.s3_counts[int(s3)] += 1

    def summary(self) -> Dict:
        n = max(self.n, 1)
        return {
            "n_samples": self.n,
            "s1_nonzero_rows": self.s1_nonzero,
            "s1_nonzero_rate": self.s1_nonzero / n,
            "s1_counts": dict(zip(S1_LABELS, self.s1_counts)),
            "s1_rates": {k: v / n for k, v in zip(S1_LABELS, self.s1_counts)},
            "s2_counts": dict(zip(S2_LABELS, self.s2_counts)),
            "s2_rates": {k: v / n for k, v in zip(S2_LABELS, self.s2_counts)},
            "s3_counts": dict(zip(S3_LABELS, self.s3_counts)),
            "s3_rates": {k: v / n for k, v in zip(S3_LABELS, self.s3_counts)},
        }


def iter_chat_corpus_texts(clean_chat_corpus_dir: str):
    """
    遍历 clean_chat_corpus 目录下的 *.tsv 文件，按行读取 query/answer 两列。
    产出 (text, source_file, field)：
      - field 为 "query" 或 "answer"
    """
    pattern = os.path.join(clean_chat_corpus_dir, "*.tsv")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                q, a = parts[0].strip(), parts[1].strip()
                if q:
                    yield q, os.path.basename(path), "query"
                if a:
                    yield a, os.path.basename(path), "answer"


def main():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--clean_chat_corpus_dir",
        type=str,
        default=os.path.join(base_dir, "dataset", "chinese-chatbot-corpus", "clean_chat_corpus"),
    )
    parser.add_argument(
        "--weibo_csv_path",
        type=str,
        default=os.path.join(
            base_dir,
            "dataset",
            "weibo",
            "微博评论情感数据集(清洗之后的，有标注，中文,csv格式)",
            "clean_weibo_text.csv",
        ),
    )
    parser.add_argument("--output_path", type=str, default=os.path.join(base_dir, "dataset_multitask.jsonl"))
    parser.add_argument("--format", type=str, choices=["jsonl", "csv"], default="jsonl")

    parser.add_argument("--max_chat", type=int, default=0, help="0=不限制；chat 每行会产出 query+answer 两条文本")
    parser.add_argument("--max_weibo", type=int, default=0, help="0=不限制")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="输出前打乱（会占用内存）")

    # chat 语料没有 S2 真实标签：默认标为 none(=5)，你也可以改为别的 id
    parser.add_argument("--chat_s2_default", type=int, default=5, help="chat 文本的 S2 默认标签（默认 none=5）")

    args = parser.parse_args()

    random.seed(args.seed)

    items = []
    stats = RunningStats()

    # 1) chat corpus -> S1/S3 用规则，S2 用默认
    n_chat = 0
    for text, src, field in iter_chat_corpus_texts(args.clean_chat_corpus_dir):
        if args.max_chat and n_chat >= args.max_chat:
            break
        text = _safe_text(text)
        if not text:
            continue
        s1 = map_s1(text)
        s2 = int(args.chat_s2_default)
        s3 = map_s3(text)

        rec = {"text": text, "s1": s1, "s2": s2, "s3": int(s3), "source": f"chat:{src}:{field}"}
        items.append(rec)
        stats.update(s1, s2, int(s3))
        n_chat += 1

    # 2) weibo csv -> S2 from label, S1/S3 规则
    df = pd.read_csv(args.weibo_csv_path)
    if args.max_weibo and args.max_weibo > 0:
        df = df.sample(n=args.max_weibo, random_state=args.seed).reset_index(drop=True)

    for _, row in df.iterrows():
        text = _safe_text(row.get("text", ""))
        if not text:
            continue
        weibo_label = int(row.get("label", 0))
        s1 = map_s1(text)
        s2 = map_s2(weibo_label)
        s3 = map_s3(text)

        rec = {"text": text, "s1": s1, "s2": int(s2), "s3": int(s3), "source": "weibo"}
        items.append(rec)
        stats.update(s1, int(s2), int(s3))

    if args.shuffle:
        random.shuffle(items)

    # 3) write output
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if args.format == "jsonl":
        with open(args.output_path, "w", encoding="utf-8") as f:
            for rec in items:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        with open(args.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "s1", "s2", "s3", "source"])
            writer.writeheader()
            for rec in items:
                rec2 = dict(rec)
                rec2["s1"] = json.dumps(rec2["s1"], ensure_ascii=False)
                writer.writerow(rec2)

    # 4) print stats
    summary = stats.summary()
    print("\n=== Dataset build done ===")
    print(f"output_path={args.output_path}")
    print(f"format={args.format}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

