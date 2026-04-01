import argparse
import json
import os
import random

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

from multitask_config import DEFAULT_INFERENCE_MODEL_DIR, S1_LABELS, S2_LABELS, S3_LABELS
from multitask_model import MultiTaskBertForPsychology
from multitask_data import load_multitask_supervision


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
        item["labels_s1"] = torch.tensor(self.labels_s1[idx], dtype=torch.long)
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

        s1_sm = torch.softmax(s1_logits, dim=-1).cpu().numpy()
        p1 = torch.argmax(s1_logits, dim=-1).cpu().numpy()
        p2 = torch.argmax(s2_logits, dim=-1).cpu().numpy()
        p3 = torch.argmax(s3_logits, dim=-1).cpu().numpy()

        if debug_samples > 0 and len(dbg) < debug_samples:
            texts = batch["text"]
            for i in range(min(len(texts), debug_samples - len(dbg))):
                top5 = sorted(list(enumerate(s1_sm[i].tolist())), key=lambda x: x[1], reverse=True)[:5]
                dbg.append(
                    {
                        "text": texts[i],
                        "s1_true": int(y1[i]),
                        "s1_pred": int(p1[i]),
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

    s1_acc = accuracy_score(y1_true, y1_pred)
    s1_precision_macro = precision_score(y1_true, y1_pred, average="macro", zero_division=0)
    s1_recall_macro = recall_score(y1_true, y1_pred, average="macro", zero_division=0)

    s2_acc = accuracy_score(y2_true, y2_pred)
    s2_precision_macro = precision_score(y2_true, y2_pred, average="macro", zero_division=0)
    s2_recall_macro = recall_score(y2_true, y2_pred, average="macro", zero_division=0)

    s3_acc = accuracy_score(y3_true, y3_pred)
    s3_precision_macro = precision_score(y3_true, y3_pred, average="macro", zero_division=0)
    s3_recall_macro = recall_score(y3_true, y3_pred, average="macro", zero_division=0)

    return {
        "metrics": {
            "s1_accuracy": float(s1_acc),
            "s1_precision_macro": float(s1_precision_macro),
            "s1_recall_macro": float(s1_recall_macro),
            "s2_accuracy": float(s2_acc),
            "s2_precision_macro": float(s2_precision_macro),
            "s2_recall_macro": float(s2_recall_macro),
            "s3_accuracy": float(s3_acc),
            "s3_precision_macro": float(s3_precision_macro),
            "s3_recall_macro": float(s3_recall_macro),
        },
        "s1_confusion_matrix": confusion_matrix(y1_true, y1_pred, labels=list(range(len(S1_LABELS)))),
        "s2_confusion_matrix": confusion_matrix(y2_true, y2_pred, labels=list(range(len(S2_LABELS)))),
        "s3_confusion_matrix": confusion_matrix(y3_true, y3_pred, labels=list(range(len(S3_LABELS)))),
        "debug": dbg,
    }


def _plot_confusion_mats(out_dir: str, val_out: dict, test_out: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    labels_lists = (list(S1_LABELS), list(S2_LABELS), list(S3_LABELS))

    def one(fname: str, cm: np.ndarray, labels: list, title: str) -> None:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.35), max(5, len(labels) * 0.35)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xlabel="Pred", ylabel="True", title=title)
        tick_labels = [str(lab)[:12] for lab in labels]
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        thresh = float(cm.max()) / 2.0 if cm.size and cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(int(cm[i, j]), "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6,
                )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150)
        plt.close(fig)

    for split_name, out in (("val", val_out), ("test", test_out)):
        p = f"{split_name}_"
        one(f"{p}s1_cm", out["s1_confusion_matrix"], labels_lists[0], f"{split_name} S1")
        one(f"{p}s2_cm", out["s2_confusion_matrix"], labels_lists[1], f"{split_name} S2")
        one(f"{p}s3_cm", out["s3_confusion_matrix"], labels_lists[2], f"{split_name} S3")
    print(f"[plots] saved under {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="在 CSV 上评估已合并的 multitask best_model（与 train 末尾 merged 权重一致）。"
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_weibo_csv_path = os.path.join(
        base_dir,
        "dataset",
        "weibo",
        "微博评论情感数据集(清洗之后的，有标注，中文,csv格式)",
        "clean_weibo_text.csv",
    )
    default_efa_csv = os.path.join(base_dir, "dataset_efaqa_multitask.csv")
    default_csv = default_efa_csv if os.path.isfile(default_efa_csv) else default_weibo_csv_path
    default_model_dir = os.path.join(base_dir, *DEFAULT_INFERENCE_MODEL_DIR.split("/"))

    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir,
        help=f"合并后的推理目录，默认 <项目>/{DEFAULT_INFERENCE_MODEL_DIR}",
    )
    parser.add_argument(
        "--weibo_csv_path",
        type=str,
        default=default_csv,
        help="含 (text,label) 或 (text,s1,s2,s3) 的 CSV；若存在 dataset_efaqa_multitask.csv 则默认可用其",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--s1_threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--debug_eval_samples", type=int, default=5)
    parser.add_argument(
        "--split_npz",
        type=str,
        default="",
        help="训练时加 --save_split_indices 得到的 npz；需与训练使用同一 CSV、--seed、--val_ratio、--test_ratio（且勿用不同的 --max_samples）",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="",
        help="若非空，写入 val_/test_ 的 s1/s2/s3 混淆矩阵热力图 PNG",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="若非空，将 metrics 与混淆矩阵（转 list）写入该 JSON 路径",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"model_dir 不存在: {args.model_dir}\n"
            "请先训练并生成 merged best_model，或传入 --model_dir 指向含 pytorch_model.bin 的目录。"
        )

    set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("你指定了 --device cuda，但当前环境 torch.cuda.is_available()==False")

    df = pd.read_csv(args.weibo_csv_path)
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    texts, labels_s1, labels_s2, labels_s3, csv_format = load_multitask_supervision(df)
    print(f"[data] format={csv_format} n={len(texts)}")

    y1_arr = np.asarray(labels_s1, dtype=np.int64)
    print(
        f"[S1 stats] not_class_15={int((y1_arr != 15).sum())}/{len(labels_s1)} "
        f"per_class={np.bincount(y1_arr, minlength=len(S1_LABELS)).tolist()}"
    )

    if args.split_npz.strip():
        z = np.load(args.split_npz)
        idx_val = np.asarray(z["val"], dtype=np.int64)
        idx_test = np.asarray(z["test"], dtype=np.int64)
        x_val = [texts[i] for i in idx_val]
        y1_val = [labels_s1[i] for i in idx_val]
        y2_val = [labels_s2[i] for i in idx_val]
        y3_val = [labels_s3[i] for i in idx_val]
        x_test = [texts[i] for i in idx_test]
        y1_test = [labels_s1[i] for i in idx_test]
        y2_test = [labels_s2[i] for i in idx_test]
        y3_test = [labels_s3[i] for i in idx_test]
        print(f"[split] from npz val={len(x_val)} test={len(x_test)} ({args.split_npz})")
    else:
        strat1 = labels_s3 if len(set(labels_s3)) > 1 else None
        x_trainval, x_test, y1_trainval, y1_test, y2_trainval, y2_test, y3_trainval, y3_test = safe_split(
            texts,
            labels_s1,
            labels_s2,
            labels_s3,
            test_size=args.test_ratio,
            seed=args.seed,
            stratify=strat1,
        )
        strat2 = y3_trainval if len(set(y3_trainval)) > 1 else None
        _, x_val, _, y1_val, _, y2_val, _, y3_val = safe_split(
            x_trainval,
            y1_trainval,
            y2_trainval,
            y3_trainval,
            test_size=args.val_ratio / (1.0 - args.test_ratio),
            seed=args.seed,
            stratify=strat2,
        )

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
    print("[S1 confusion_matrix shape]", val_out["s1_confusion_matrix"].shape)
    print("[S2 confusion_matrix]\n", val_out["s2_confusion_matrix"])
    print("[S3 confusion_matrix]\n", val_out["s3_confusion_matrix"])
    for i, row in enumerate(val_out["debug"]):
        print(f"\n[VAL #{i}] {row['text']}")
        print(
            f"  S1 true={row['s1_true']}({S1_LABELS[row['s1_true']]}) "
            f"pred={row['s1_pred']}({S1_LABELS[row['s1_pred']]}) top5={row['s1_top5_probs']}"
        )
        print(f"  S2 true={row['s2_true']}({S2_LABELS[row['s2_true']]}) pred={row['s2_pred']}({S2_LABELS[row['s2_pred']]})")
        print(f"  S3 true={row['s3_true']}({S3_LABELS[row['s3_true']]}) pred={row['s3_pred']}({S3_LABELS[row['s3_pred']]})")

    print("\n=== TEST ===")
    test_out = evaluate(model, test_loader, device, s1_threshold=args.s1_threshold, debug_samples=args.debug_eval_samples)
    print(test_out["metrics"])
    print("[S1 confusion_matrix shape]", test_out["s1_confusion_matrix"].shape)
    print("[S2 confusion_matrix]\n", test_out["s2_confusion_matrix"])
    print("[S3 confusion_matrix]\n", test_out["s3_confusion_matrix"])
    for i, row in enumerate(test_out["debug"]):
        print(f"\n[TEST #{i}] {row['text']}")
        print(
            f"  S1 true={row['s1_true']}({S1_LABELS[row['s1_true']]}) "
            f"pred={row['s1_pred']}({S1_LABELS[row['s1_pred']]}) top5={row['s1_top5_probs']}"
        )
        print(f"  S2 true={row['s2_true']}({S2_LABELS[row['s2_true']]}) pred={row['s2_pred']}({S2_LABELS[row['s2_pred']]})")
        print(f"  S3 true={row['s3_true']}({S3_LABELS[row['s3_true']]}) pred={row['s3_pred']}({S3_LABELS[row['s3_pred']]})")

    if args.plots_dir.strip():
        _plot_confusion_mats(args.plots_dir.strip(), val_out, test_out)

    if args.save_json.strip():
        payload = {
            "model_dir": os.path.abspath(args.model_dir),
            "csv_path": os.path.abspath(args.weibo_csv_path),
            "split_npz": os.path.abspath(args.split_npz) if args.split_npz.strip() else "",
            "val_metrics": val_out["metrics"],
            "test_metrics": test_out["metrics"],
            "val_s1_cm": val_out["s1_confusion_matrix"].astype(int).tolist(),
            "val_s2_cm": val_out["s2_confusion_matrix"].astype(int).tolist(),
            "val_s3_cm": val_out["s3_confusion_matrix"].astype(int).tolist(),
            "test_s1_cm": test_out["s1_confusion_matrix"].astype(int).tolist(),
            "test_s2_cm": test_out["s2_confusion_matrix"].astype(int).tolist(),
            "test_s3_cm": test_out["s3_confusion_matrix"].astype(int).tolist(),
        }
        _jd = os.path.dirname(os.path.abspath(args.save_json.strip()))
        if _jd:
            os.makedirs(_jd, exist_ok=True)
        with open(args.save_json.strip(), "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
        print(f"[json] saved {args.save_json.strip()}")


if __name__ == "__main__":
    main()
