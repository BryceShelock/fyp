import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from transformers import BertTokenizer

from multitask_config import map_s1, map_s2, map_s3, S1_LABELS, S2_LABELS, S3_LABELS
from multitask_model import MultiTaskBertForPsychology


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EvalDataset(Dataset):
    def __init__(self, texts, encodings, labels_s1, labels_s2, labels_s3):
        self.texts = texts
        self.encodings = encodings
        self.labels_s1 = labels_s1
        self.labels_s2 = labels_s2
        self.labels_s3 = labels_s3

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["text"] = self.texts[idx]
        item["labels_s1"] = torch.tensor(self.labels_s1[idx], dtype=torch.float32)
        item["labels_s2"] = torch.tensor(self.labels_s2[idx], dtype=torch.long)
        item["labels_s3"] = torch.tensor(self.labels_s3[idx], dtype=torch.long)
        return item


def safe_split(*arrays, test_size, seed, stratify):
    try:
        return train_test_split(
            *arrays,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            *arrays,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )


@torch.no_grad()
def evaluate(model, dataloader, device, s1_threshold: float, debug_samples: int):
    model.eval()

    all_y1, all_y2, all_y3 = [], [], []
    all_p1, all_p2, all_p3 = [], [], []

    dbg = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        y1 = batch["labels_s1"].cpu().numpy()
        y2 = batch["labels_s2"].cpu().numpy()
        y3 = batch["labels_s3"].cpu().numpy()

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        s1_logits = out["s1_logits"]
        s2_logits = out["s2_logits"]
        s3_logits = out["s3_logits"]

        s1_probs = torch.sigmoid(s1_logits).cpu().numpy()
        p1 = (s1_probs >= s1_threshold).astype(int)
        p2 = torch.argmax(s2_logits, dim=-1).cpu().numpy()
        p3 = torch.argmax(s3_logits, dim=-1).cpu().numpy()

        if debug_samples > 0 and len(dbg) < debug_samples:
            texts = batch["text"]
            for i in range(min(len(texts), debug_samples - len(dbg))):
                y1_ids = [j for j, v in enumerate(y1[i].astype(int).tolist()) if v == 1]
                p1_ids = [j for j, v in enumerate(p1[i].astype(int).tolist()) if v == 1]
                top5 = sorted(list(enumerate(s1_probs[i].tolist())), key=lambda x: x[1], reverse=True)[:5]
                dbg.append(
                    {
                        "text": texts[i],
                        "s1_true_ids": y1_ids,
                        "s1_pred_ids": p1_ids,
                        "s1_top5_probs": top5,
                        "s2_true": int(y2[i]),
                        "s2_pred": int(p2[i]),
                        "s3_true": int(y3[i]),
                        "s3_pred": int(p3[i]),
                    }
                )

        all_y1.append(y1)
        all_y2.append(y2)
        all_y3.append(y3)
        all_p1.append(p1)
        all_p2.append(p2)
        all_p3.append(p3)

    y1_true = np.concatenate(all_y1, axis=0)
    y2_true = np.concatenate(all_y2, axis=0)
    y3_true = np.concatenate(all_y3, axis=0)

    y1_pred = np.concatenate(all_p1, axis=0)
    y2_pred = np.concatenate(all_p2, axis=0)
    y3_pred = np.concatenate(all_p3, axis=0)

    # S1 multi-label
    s1_subset_acc = accuracy_score(y1_true, y1_pred)
    s1_precision_micro = precision_score(y1_true, y1_pred, average="micro", zero_division=0)
    s1_recall_micro = recall_score(y1_true, y1_pred, average="micro", zero_division=0)
    s1_precision_macro = precision_score(y1_true, y1_pred, average="macro", zero_division=0)
    s1_recall_macro = recall_score(y1_true, y1_pred, average="macro", zero_division=0)

    # S2 single-label
    s2_acc = accuracy_score(y2_true, y2_pred)
    s2_precision_macro = precision_score(y2_true, y2_pred, average="macro", zero_division=0)
    s2_recall_macro = recall_score(y2_true, y2_pred, average="macro", zero_division=0)

    # S3 single-label
    s3_acc = accuracy_score(y3_true, y3_pred)
    s3_precision_macro = precision_score(y3_true, y3_pred, average="macro", zero_division=0)
    s3_recall_macro = recall_score(y3_true, y3_pred, average="macro", zero_division=0)

    return {
        "metrics": {
            "s1_subset_acc": float(s1_subset_acc),
            "s1_precision_micro": float(s1_precision_micro),
            "s1_recall_micro": float(s1_recall_micro),
            "s1_precision_macro": float(s1_precision_macro),
            "s1_recall_macro": float(s1_recall_macro),
            "s2_accuracy": float(s2_acc),
            "s2_precision_macro": float(s2_precision_macro),
            "s2_recall_macro": float(s2_recall_macro),
            "s3_accuracy": float(s3_acc),
            "s3_precision_macro": float(s3_precision_macro),
            "s3_recall_macro": float(s3_recall_macro),
        },
        "s2_confusion_matrix": confusion_matrix(y2_true, y2_pred, labels=list(range(len(S2_LABELS)))),
        "s3_confusion_matrix": confusion_matrix(y3_true, y3_pred, labels=list(range(len(S3_LABELS)))),
        "debug": dbg,
    }


def main():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_weibo_csv_path = os.path.join(
        base_dir,
        "dataset",
        "weibo",
        "微博评论情感数据集(清洗之后的，有标注，中文,csv格式)",
        "clean_weibo_text.csv",
    )

    parser.add_argument("--model_dir", type=str, required=True, help="例如 multitask_output_demo2/best_model")
    parser.add_argument("--weibo_csv_path", type=str, default=default_weibo_csv_path)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--s1_threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--debug_eval_samples", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("你指定了 --device cuda，但当前环境 torch.cuda.is_available()==False")

    # 读取数据
    df = pd.read_csv(args.weibo_csv_path)
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    weibo_labels = df["label"].tolist()

    labels_s1 = [map_s1(t) for t in texts]
    labels_s2 = [map_s2(int(x)) for x in weibo_labels]
    labels_s3 = [map_s3(t) for t in texts]

    s1_mat = np.asarray(labels_s1, dtype=np.float32)
    n_pos = int((s1_mat.sum(axis=1) > 0).sum())
    print(f"[S1 stats] nonzero_rows={n_pos}/{len(labels_s1)} rate={n_pos/max(len(labels_s1),1):.3f} per_dim={s1_mat.sum(axis=0).tolist()}")

    # 切分
    strat1 = labels_s2 if len(set(labels_s2)) > 1 else None
    x_trainval, x_test, y1_trainval, y1_test, y2_trainval, y2_test, y3_trainval, y3_test = safe_split(
        texts,
        labels_s1,
        labels_s2,
        labels_s3,
        test_size=args.test_ratio,
        seed=args.seed,
        stratify=strat1,
    )
    strat2 = y2_trainval if len(set(y2_trainval)) > 1 else None
    _, x_val, _, y1_val, _, y2_val, _, y3_val = safe_split(
        x_trainval,
        y1_trainval,
        y2_trainval,
        y3_trainval,
        test_size=args.val_ratio / (1.0 - args.test_ratio),
        seed=args.seed,
        stratify=strat2,
    )

    # tokenizer & model：必须从 model_dir 加载（确保一致）
    tokenizer = BertTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = MultiTaskBertForPsychology.from_pretrained(args.model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    if device == "cuda":
        try:
            model.half()
        except Exception:
            pass

    def enc(batch_texts):
        return tokenizer(batch_texts, truncation=True, padding="max_length", max_length=args.max_length)

    val_ds = EvalDataset(x_val, enc(x_val), y1_val, y2_val, y3_val)
    test_ds = EvalDataset(x_test, enc(x_test), y1_test, y2_test, y3_test)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("\n=== VAL ===")
    val_out = evaluate(model, val_loader, device, s1_threshold=args.s1_threshold, debug_samples=args.debug_eval_samples)
    print(val_out["metrics"])
    print("[S2 confusion_matrix]\n", val_out["s2_confusion_matrix"])
    print("[S3 confusion_matrix]\n", val_out["s3_confusion_matrix"])
    for i, row in enumerate(val_out["debug"]):
        print(f"\n[VAL #{i}] {row['text']}")
        print(f"  S1 true_ids={row['s1_true_ids']} pred_ids={row['s1_pred_ids']} top5_probs={row['s1_top5_probs']}")
        print(f"  S2 true={row['s2_true']}({S2_LABELS[row['s2_true']]}) pred={row['s2_pred']}({S2_LABELS[row['s2_pred']]})")
        print(f"  S3 true={row['s3_true']}({S3_LABELS[row['s3_true']]}) pred={row['s3_pred']}({S3_LABELS[row['s3_pred']]})")

    print("\n=== TEST ===")
    test_out = evaluate(model, test_loader, device, s1_threshold=args.s1_threshold, debug_samples=args.debug_eval_samples)
    print(test_out["metrics"])
    print("[S2 confusion_matrix]\n", test_out["s2_confusion_matrix"])
    print("[S3 confusion_matrix]\n", test_out["s3_confusion_matrix"])
    for i, row in enumerate(test_out["debug"]):
        print(f"\n[TEST #{i}] {row['text']}")
        print(f"  S1 true_ids={row['s1_true_ids']} pred_ids={row['s1_pred_ids']} top5_probs={row['s1_top5_probs']}")
        print(f"  S2 true={row['s2_true']}({S2_LABELS[row['s2_true']]}) pred={row['s2_pred']}({S2_LABELS[row['s2_pred']]})")
        print(f"  S3 true={row['s3_true']}({S3_LABELS[row['s3_true']]}) pred={row['s3_pred']}({S3_LABELS[row['s3_pred']]})")


if __name__ == "__main__":
    main()

