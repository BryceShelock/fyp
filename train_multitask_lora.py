import os
import json
import random
from dataclasses import dataclass
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from transformers import BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
except ImportError as e:
    raise ImportError(
        "缺少依赖包 `peft`，需要先安装后才能训练 LoRA。"
        "\n建议：使用你已有的环境文件 `environment.yml` 创建环境（其中已包含 peft==0.11.1）。"
        "\n当前错误：{}".format(e)
    )

from multitask_config import S1_LABELS, S2_LABELS, S3_LABELS
from multitask_model import MultiTaskBertForPsychology
from multitask_data import load_multitask_supervision

# 本地预训练 BERT 基座目录（需含 config.json、vocab.txt、pytorch_model.bin 或 model.safetensors）
# 可用环境变量覆盖，便于换机器：FYP_BACKBONE_BERT_CHINESE / FYP_BACKBONE_MENTALBERT
_BACKBONE_ROOT = os.environ.get("FYP_MODELS_ROOT", r"D:\CS_project\FYP\models")
BACKBONE_BERT_CHINESE = os.environ.get(
    "FYP_BACKBONE_BERT_CHINESE",
    os.path.join(_BACKBONE_ROOT, "chinese_L-12_H-768_A-12"),
)
BACKBONE_CHINESE_MENTALBERT = os.environ.get(
    "FYP_BACKBONE_MENTALBERT",
    os.path.join(_BACKBONE_ROOT, "Chinese-MentalBERT"),
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _balanced_class_weights_vector(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    sklearn 的 compute_class_weight 要求 classes 里的每个 id 都在 y 中出现。
    多任务数据极度不平衡时，训练子集可能缺某一类；只对 y 中出现的类算权重，其余类权重保持 1.0。
    """
    y = np.asarray(y, dtype=np.int64)
    full = np.ones(num_classes, dtype=np.float32)
    present = np.unique(y)
    present = present[(present >= 0) & (present < num_classes)]
    if present.size == 0:
        return full
    raw = compute_class_weight(class_weight="balanced", classes=present, y=y).astype(np.float32)
    for c, wt in zip(present, raw):
        full[int(c)] = float(wt)
    return full


class MultiTaskWeiboDataset(Dataset):
    def __init__(self, texts, encodings, labels_s1, labels_s2, labels_s3):
        self.texts = texts
        self.encodings = encodings
        self.labels_s1 = labels_s1
        self.labels_s2 = labels_s2
        self.labels_s3 = labels_s3

    def __len__(self):
        return len(self.labels_s2)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["text"] = self.texts[idx]
        item["labels_s1"] = torch.tensor(self.labels_s1[idx], dtype=torch.float32)
        item["labels_s2"] = torch.tensor(self.labels_s2[idx], dtype=torch.long)
        item["labels_s3"] = torch.tensor(self.labels_s3[idx], dtype=torch.long)
        return item


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    s1_threshold: float = 0.15,
    debug_samples: int = 0,
):
    model.eval()

    debug_samples = int(debug_samples or 0)
    dbg_texts = []
    dbg_y1 = []
    dbg_y2 = []
    dbg_y3 = []
    dbg_p1 = []
    dbg_p2 = []
    dbg_p3 = []
    dbg_prob1 = []

    all_y1 = []
    all_y2 = []
    all_y3 = []

    all_pred1 = []
    all_pred2 = []
    all_pred3 = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_texts = batch.get("text", None)
        y1 = batch["labels_s1"].cpu().numpy()
        y2 = batch["labels_s2"].cpu().numpy()
        y3 = batch["labels_s3"].cpu().numpy()

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        s1_logits = out["s1_logits"]
        s2_logits = out["s2_logits"]
        s3_logits = out["s3_logits"]

        s1_probs = torch.sigmoid(s1_logits).cpu().numpy()
        pred1 = (s1_probs >= s1_threshold).astype(int)
        pred2 = torch.argmax(s2_logits, dim=-1).cpu().numpy()
        pred3 = torch.argmax(s3_logits, dim=-1).cpu().numpy()

        if debug_samples > 0 and len(dbg_texts) < debug_samples and batch_texts is not None:
            # batch_texts 是 list[str]
            need = debug_samples - len(dbg_texts)
            take = min(need, len(batch_texts))
            dbg_texts.extend(list(batch_texts[:take]))
            dbg_y1.extend(list(y1[:take]))
            dbg_y2.extend(list(y2[:take]))
            dbg_y3.extend(list(y3[:take]))
            dbg_p1.extend(list(pred1[:take]))
            dbg_p2.extend(list(pred2[:take]))
            dbg_p3.extend(list(pred3[:take]))
            dbg_prob1.extend(list(s1_probs[:take]))

        all_y1.append(y1)
        all_y2.append(y2)
        all_y3.append(y3)

        all_pred1.append(pred1)
        all_pred2.append(pred2)
        all_pred3.append(pred3)

    y1_true = np.concatenate(all_y1, axis=0)
    y2_true = np.concatenate(all_y2, axis=0)
    y3_true = np.concatenate(all_y3, axis=0)

    y1_pred = np.concatenate(all_pred1, axis=0)
    y2_pred = np.concatenate(all_pred2, axis=0)
    y3_pred = np.concatenate(all_pred3, axis=0)

    # S1 multi-label
    s1_precision_micro = precision_score(y1_true, y1_pred, average="micro", zero_division=0)
    s1_recall_micro = recall_score(y1_true, y1_pred, average="micro", zero_division=0)
    s1_precision_macro = precision_score(y1_true, y1_pred, average="macro", zero_division=0)
    s1_recall_macro = recall_score(y1_true, y1_pred, average="macro", zero_division=0)
    s1_subset_acc = accuracy_score(y1_true, y1_pred)

    # S2 single-label
    s2_acc = accuracy_score(y2_true, y2_pred)
    s2_precision_macro = precision_score(y2_true, y2_pred, average="macro", zero_division=0)
    s2_recall_macro = recall_score(y2_true, y2_pred, average="macro", zero_division=0)

    # S3 single-label
    s3_acc = accuracy_score(y3_true, y3_pred)
    s3_precision_macro = precision_score(y3_true, y3_pred, average="macro", zero_division=0)
    s3_recall_macro = recall_score(y3_true, y3_pred, average="macro", zero_division=0)

    if debug_samples > 0 and len(dbg_texts) > 0:
        print("\n===== DEBUG EVAL SAMPLES =====")
        print(f"s1_threshold={s1_threshold}")
        for i in range(min(debug_samples, len(dbg_texts))):
            t = dbg_texts[i]
            y1_i = np.asarray(dbg_y1[i]).astype(int).tolist()
            p1_i = np.asarray(dbg_p1[i]).astype(int).tolist()
            prob1_i = np.asarray(dbg_prob1[i]).tolist()
            y2_i = int(dbg_y2[i])
            p2_i = int(dbg_p2[i])
            y3_i = int(dbg_y3[i])
            p3_i = int(dbg_p3[i])

            # 打印时只显示 S1 命中的索引，避免 8维全展开太难看
            y1_ids = [j for j, v in enumerate(y1_i) if v == 1]
            p1_ids = [j for j, v in enumerate(p1_i) if v == 1]
            top_s1 = sorted(list(enumerate(prob1_i)), key=lambda x: x[1], reverse=True)[:5]

            print(f"\n[#{i}] text={t}")
            print(f"  S1 true_ids={y1_ids} pred_ids={p1_ids} top5_probs={top_s1}")
            print(f"  S2 true={y2_i}({S2_LABELS[y2_i]}) pred={p2_i}({S2_LABELS[p2_i]})")
            print(f"  S3 true={y3_i}({S3_LABELS[y3_i]}) pred={p3_i}({S3_LABELS[p3_i]})")

        print("\n[S2 confusion_matrix] rows=true cols=pred")
        print(confusion_matrix(y2_true, y2_pred, labels=list(range(len(S2_LABELS)))))
        print("\n[S3 confusion_matrix] rows=true cols=pred")
        print(confusion_matrix(y3_true, y3_pred, labels=list(range(len(S3_LABELS)))))
        print("===== END DEBUG =====\n")

    return {
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
    }


def build_dataloaders(
    tokenizer,
    texts,
    labels_s1,
    labels_s2,
    labels_s3,
    max_length: int,
    batch_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    def safe_split(*arrays, test_size, stratify):
        """
        小样本时 stratify 可能失败（例如 test_size < 类别数）。
        这里自动降级为非分层划分，保证 smoke test 可跑通。
        """
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

    inds = np.arange(len(texts), dtype=np.int64)

    strat1 = labels_s2 if len(set(labels_s2)) > 1 else None
    x_trainval, x_test, i_trainval, i_test, y1_trainval, y1_test, y2_trainval, y2_test, y3_trainval, y3_test = safe_split(
        texts,
        inds,
        labels_s1,
        labels_s2,
        labels_s3,
        test_size=test_ratio,
        stratify=strat1,
    )

    strat2 = y2_trainval if len(set(y2_trainval)) > 1 else None
    x_train, x_val, i_train, i_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val = safe_split(
        x_trainval,
        i_trainval,
        y1_trainval,
        y2_trainval,
        y3_trainval,
        test_size=val_ratio / (1.0 - test_ratio),
        stratify=strat2,
    )

    def tokenize_batch(batch_texts):
        return tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_enc = tokenize_batch(x_train)
    val_enc = tokenize_batch(x_val)
    test_enc = tokenize_batch(x_test)

    train_ds = MultiTaskWeiboDataset(x_train, train_enc, y1_train, y2_train, y3_train)
    val_ds = MultiTaskWeiboDataset(x_val, val_enc, y1_val, y2_val, y3_val)
    test_ds = MultiTaskWeiboDataset(x_test, test_enc, y1_test, y2_test, y3_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    stats = {
        "y1_train": y1_train,
        "y2_train": y2_train,
        "y3_train": y3_train,
        "idx_train": i_train.tolist(),
        "idx_val": i_val.tolist(),
        "idx_test": i_test.tolist(),
    }
    return train_loader, val_loader, test_loader, stats


def main():
    import argparse

    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_weibo_csv_path = os.path.join(
        base_dir,
        "dataset",
        "weibo",
        "微博评论情感数据集(清洗之后的，有标注，中文,csv格式)",
        "clean_weibo_text.csv",
    )
    parser.add_argument("--weibo_csv_path", type=str, default=default_weibo_csv_path)
    parser.add_argument("--output_dir", type=str, default="multitask_output")
    parser.add_argument(
        "--model_name",
        type=str,
        default=BACKBONE_BERT_CHINESE,
        help=f"预训练 BERT 目录。默认中文 BERT：{BACKBONE_BERT_CHINESE}；MentalBERT：{BACKBONE_CHINESE_MENTALBERT}",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--s1_threshold", type=float, default=0.15)
    parser.add_argument("--s1_loss_weight", type=float, default=1.0)
    parser.add_argument("--s2_loss_weight", type=float, default=2.0)
    parser.add_argument("--s3_loss_weight", type=float, default=5.0)
    parser.add_argument("--debug_s1_stats", action="store_true")
    parser.add_argument("--debug_eval_samples", type=int, default=0, help=">0 时打印验证/测试样本与混淆矩阵")
    parser.add_argument("--use_s1_pos_weight", action="store_true", help="给 S1 的 BCE 加 pos_weight，缓解全0预测")
    parser.add_argument("--use_s2_class_weight", action="store_true", help="给 S2 的 CE 加 class weight，缓解全预测 none")
    parser.add_argument("--use_s3_class_weight", action="store_true", help="给 S3 的 CE 加 class weight，缓解全预测 none")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="按 S1 是否为正样本做采样，让 batch 更容易含正例")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="0表示不限制样本数")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--save_split_indices",
        action="store_true",
        help="将 train/val/test 在原 CSV 中的行索引写入 output_dir/split_indices.npz（大表会占磁盘）",
    )
    parser.set_defaults(use_s1_pos_weight=True, use_s2_class_weight=True, use_s3_class_weight=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.weibo_csv_path)
    # 允许你用 max_samples 快速跑通流程
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    texts, labels_s1, labels_s2, labels_s3, csv_format = load_multitask_supervision(df)
    print(f"[data] format={csv_format} n={len(texts)} path={args.weibo_csv_path}")

    s1_mat = np.asarray(labels_s1, dtype=np.float32)
    n_pos = int((s1_mat.sum(axis=1) > 0).sum())
    pos_rate = n_pos / max(len(labels_s1), 1)
    per_dim = s1_mat.sum(axis=0).tolist()
    print(f"[S1 stats] nonzero_rows={n_pos}/{len(labels_s1)} ({pos_rate:.3f})  per_label_counts={per_dim}")
    if args.debug_s1_stats:
        print("[S1 sample]", labels_s1[:10])

    # 构建多头模型 + LoRA（完全本地加载，不联网）
    # 兼容两种本地模型格式：
    # - PyTorch: 目录下存在 pytorch_model.bin
    # - TensorFlow ckpt: 目录下存在 bert_model.ckpt.index（先自动转换为 PyTorch 再加载，避免 transformers 将 ckpt 前缀当 repo id）
    model_dir = args.model_name
    pt_bin = os.path.join(model_dir, "pytorch_model.bin")
    st_bin = os.path.join(model_dir, "model.safetensors")
    has_hf_weights = os.path.isfile(pt_bin) or os.path.isfile(st_bin)
    tf_index = os.path.join(model_dir, "bert_model.ckpt.index")

    if os.path.isfile(tf_index) and (not has_hf_weights):
        converted_dir = os.path.join(args.output_dir, "converted_bert_pytorch")
        os.makedirs(converted_dir, exist_ok=True)
        converted_pt_bin = os.path.join(converted_dir, "pytorch_model.bin")

        # 复制 tokenizer/config 相关文件，确保后续完全离线
        for fname in ["config.json", "vocab.txt"]:
            src = os.path.join(model_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(converted_dir, fname))

        if not os.path.isfile(converted_pt_bin):
            tf_ckpt_prefix = os.path.join(model_dir, "bert_model.ckpt")
            config_path = os.path.join(model_dir, "config.json")
            try:
                from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import (
                    convert_tf_checkpoint_to_pytorch,
                )
            except Exception as e:
                raise RuntimeError(
                    "检测到你提供的是 TensorFlow checkpoint（bert_model.ckpt.*），需要先转换为 PyTorch 才能训练。\n"
                    "但当前 transformers 环境中找不到转换工具。\n"
                    f"底层错误：{repr(e)}"
                )
            try:
                convert_tf_checkpoint_to_pytorch(
                    tf_checkpoint_path=tf_ckpt_prefix,
                    bert_config_file=config_path,
                    pytorch_dump_path=converted_pt_bin,
                )
            except Exception as e:
                raise RuntimeError(
                    "TensorFlow ckpt -> PyTorch 转换失败。请确认：\n"
                    f"- TF ckpt 前缀：{tf_ckpt_prefix}（应同时存在 .index/.data/.meta）\n"
                    f"- config.json：{config_path}\n"
                    "- 环境已安装 TensorFlow（转换需要）。\n"
                    f"底层错误：{repr(e)}"
                )

        # 转换成功后，训练/加载都指向 PyTorch 目录
        model_dir = converted_dir

    # TF 转换后 model_dir 已变，需重新判断权重文件
    has_hf_weights = os.path.isfile(os.path.join(model_dir, "pytorch_model.bin")) or os.path.isfile(
        os.path.join(model_dir, "model.safetensors")
    )

    # tokenizer：优先从最终 model_dir 加载（转 PyTorch 时会复制 vocab 相关文件）
    tokenizer = BertTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    train_loader, val_loader, test_loader, split_stats = build_dataloaders(
        tokenizer=tokenizer,
        texts=texts,
        labels_s1=labels_s1,
        labels_s2=labels_s2,
        labels_s3=labels_s3,
        max_length=args.max_length,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.save_split_indices:
        np.savez_compressed(
            os.path.join(args.output_dir, "split_indices.npz"),
            train=np.asarray(split_stats["idx_train"], dtype=np.int64),
            val=np.asarray(split_stats["idx_val"], dtype=np.int64),
            test=np.asarray(split_stats["idx_test"], dtype=np.int64),
        )
        print(f"[split] saved {os.path.join(args.output_dir, 'split_indices.npz')}")

    train_log_path = os.path.join(args.output_dir, "training_log.jsonl")
    if os.path.isfile(train_log_path):
        os.remove(train_log_path)

    # 可选：WeightedRandomSampler（让 batch 更容易包含 S1 正例，避免“全空预测最安全”）
    if args.use_weighted_sampler:
        y1_train = np.asarray(split_stats["y1_train"], dtype=np.float32)
        is_pos = (y1_train.sum(axis=1) > 0).astype(np.int64)
        n_pos = int(is_pos.sum())
        n_neg = int(len(is_pos) - n_pos)
        w_pos = (n_pos + n_neg) / max(n_pos, 1)
        w_neg = (n_pos + n_neg) / max(n_neg, 1)
        weights = np.where(is_pos == 1, w_pos, w_neg).astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, sampler=sampler)
        print(f"[Sampler] enabled. n_pos={n_pos} n_neg={n_neg} w_pos={w_pos:.3f} w_neg={w_neg:.3f}")

    resolved_backbone_dir = model_dir

    if has_hf_weights:
        # transformers 会识别 pytorch_model.bin 或 model.safetensors；三任务头在 checkpoint 中不存在时会自动初始化
        model = MultiTaskBertForPsychology.from_pretrained(
            model_dir,
            local_files_only=True,
            num_s1_labels=len(S1_LABELS),
            num_s2_labels=len(S2_LABELS),
            num_s3_labels=len(S3_LABELS),
        )
    else:
        raise FileNotFoundError(
            "无法从本地加载模型。请确认：\n"
            "- 目录存在 pytorch_model.bin 或 model.safetensors（HuggingFace 快照）\n"
            f"当前 model_dir={model_dir}"
        )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    # PEFT 会根据 torch 版本认为支持 DTensor，但部分 Windows/torch 组合实际上不可用，
    # 会导致访问 `torch.distributed.tensor.DTensor` 报错。
    # 这里显式关闭 peft 的 DTensor/DTensor并行分支，强制走普通 nn.Linear 路径。
    import peft.tuners.tuners_utils as peft_tuners_utils

    peft_tuners_utils._torch_supports_dtensor = False  # type: ignore[attr-defined]
    peft_tuners_utils._torch_supports_distributed = False  # type: ignore[attr-defined]

    model = get_peft_model(model, lora_config)

    # 让分类头也可训练（LoRA默认只训练adapter参数）
    for name, p in model.named_parameters():
        if ("s1_head" in name) or ("s2_head" in name) or ("s3_head" in name):
            p.requires_grad = True

    model.to(device)

    # 训练器（手写循环，便于多头损失）
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ===== Loss：支持 class weight / pos_weight =====
    # S1 pos_weight：pos_weight[i] = N_neg[i] / N_pos[i]
    bce_kwargs = {}
    if args.use_s1_pos_weight:
        y1_train = np.asarray(split_stats["y1_train"], dtype=np.float32)
        pos = y1_train.sum(axis=0)
        neg = y1_train.shape[0] - pos
        # 避免除0：若某类完全没有正样本，则该维度 pos_weight 设为 1（等价不加权）
        pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0).astype(np.float32)
        pos_weight_t = torch.tensor(pos_weight, device=device)
        print(f"[S1 pos_weight] {pos_weight.tolist()}")
        bce_kwargs["pos_weight"] = pos_weight_t

    bce = nn.BCEWithLogitsLoss(**bce_kwargs)

    # S2 / S3 class weight：weight[c] ~ 1 / freq[c]
    ce_s2_kwargs = {}
    ce_s3_kwargs = {}

    if args.use_s2_class_weight:
        y2_train = np.asarray(split_stats["y2_train"], dtype=np.int64)
        weights = _balanced_class_weights_vector(y2_train, len(S2_LABELS))
        w = torch.tensor(weights, device=device)
        counts = np.bincount(y2_train, minlength=len(S2_LABELS)).astype(np.int64)
        missing = [i for i in range(len(S2_LABELS)) if counts[i] == 0]
        if missing:
            print(f"[S2 class_weight] 训练子集中未出现的类别（权重保持1.0）: {missing}")
        print(f"[S2 class_weight] counts={counts.tolist()} weights={weights.tolist()}")
        ce_s2_kwargs["weight"] = w

    if args.use_s3_class_weight:
        y3_train = np.asarray(split_stats["y3_train"], dtype=np.int64)
        weights = _balanced_class_weights_vector(y3_train, len(S3_LABELS))
        w = torch.tensor(weights, device=device)
        counts = np.bincount(y3_train, minlength=len(S3_LABELS)).astype(np.int64)
        missing = [i for i in range(len(S3_LABELS)) if counts[i] == 0]
        if missing:
            print(f"[S3 class_weight] 训练子集中未出现的类别（权重保持1.0）: {missing}")
        print(f"[S3 class_weight] counts={counts.tolist()} weights={weights.tolist()}")
        ce_s3_kwargs["weight"] = w

    ce_s2 = nn.CrossEntropyLoss(**ce_s2_kwargs)
    ce_s3 = nn.CrossEntropyLoss(**ce_s3_kwargs)

    best_metric = -1.0
    best_state_path = os.path.join(args.output_dir, "best_model")
    best_adapter_path = os.path.join(args.output_dir, "best_peft_adapter")
    best_full_base_path = os.path.join(args.output_dir, "best_full_base_with_heads")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_s1 = batch["labels_s1"].to(device)
            labels_s2 = batch["labels_s2"].to(device)
            labels_s3 = batch["labels_s3"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            s1_logits = out["s1_logits"]
            s2_logits = out["s2_logits"]
            s3_logits = out["s3_logits"]

            loss_s1 = bce(s1_logits, labels_s1)
            loss_s2 = ce_s2(s2_logits, labels_s2)
            loss_s3 = ce_s3(s3_logits, labels_s3)

            loss = (
                args.s1_loss_weight * loss_s1
                + args.s2_loss_weight * loss_s2
                + args.s3_loss_weight * loss_s3
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": float(loss.item())})

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            s1_threshold=args.s1_threshold,
            debug_samples=args.debug_eval_samples,
        )
        # 一个简单的综合指标：S2/S3准确率 + S1 micro precision + S1 micro recall
        composite = (
            val_metrics["s2_accuracy"]
            + val_metrics["s3_accuracy"]
            + 0.5 * val_metrics["s1_precision_micro"]
            + 0.5 * val_metrics["s1_recall_micro"]
        )
        print(f"[Val] epoch={epoch+1} metrics={val_metrics}")
        with open(train_log_path, "a", encoding="utf-8") as tlf:
            tlf.write(
                json.dumps(
                    {"epoch": epoch + 1, "val_metrics": val_metrics, "composite": composite},
                    ensure_ascii=False,
                )
                + "\n"
            )

        if composite > best_metric:
            best_metric = composite
            # 注意：训练过程中不要 merge_and_unload()，否则 PeftModel 会被破坏、后续 epoch 无法继续训练。
            # 这里保存两份：
            # 1) LoRA adapter（轻量）
            # 2) 含三头(s1/s2/s3 head)的 base 权重（必须保存，否则后面只加载 adapter 会导致 head 随机初始化，test 会“几乎全错”）
            model.save_pretrained(best_adapter_path)
            tokenizer.save_pretrained(best_adapter_path)
            model.base_model.model.save_pretrained(best_full_base_path, safe_serialization=False)
            tokenizer.save_pretrained(best_full_base_path)

    # 最终在 test 集测试 best
    print(f"Best composite={best_metric:.6f}")
    if best_metric <= -1.0:
        raise RuntimeError("训练未产生有效的 best checkpoint（composite 未更新）。")

    base = MultiTaskBertForPsychology.from_pretrained(
        best_full_base_path,
        local_files_only=True,
        num_s1_labels=len(S1_LABELS),
        num_s2_labels=len(S2_LABELS),
        num_s3_labels=len(S3_LABELS),
    )
    peft_wrap = PeftModel.from_pretrained(base, best_adapter_path, is_trainable=False)
    merged_model = peft_wrap.merge_and_unload()
    merged_model.save_pretrained(best_state_path, safe_serialization=False)
    tokenizer.save_pretrained(best_state_path)
    with open(os.path.join(best_state_path, "multitask_labels.json"), "w", encoding="utf-8") as f:
        json.dump({"S1_LABELS": S1_LABELS, "S2_LABELS": S2_LABELS, "S3_LABELS": S3_LABELS}, f, ensure_ascii=False, indent=2)

    merged_model.to(device)

    val_metrics_final = evaluate(
        merged_model,
        val_loader,
        device,
        s1_threshold=args.s1_threshold,
        debug_samples=args.debug_eval_samples,
    )
    test_metrics = evaluate(
        merged_model,
        test_loader,
        device,
        s1_threshold=args.s1_threshold,
        debug_samples=args.debug_eval_samples,
    )
    print(f"[Val/merged] metrics={val_metrics_final}")
    print(f"[Test] metrics={test_metrics}")

    report = {
        "data_format": csv_format,
        "csv_path": os.path.abspath(args.weibo_csv_path),
        "backbone": os.path.abspath(args.model_name),
        "output_dir": os.path.abspath(args.output_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "n_samples": len(texts),
        "n_train": len(split_stats["idx_train"]),
        "n_val": len(split_stats["idx_val"]),
        "n_test": len(split_stats["idx_test"]),
        "best_val_composite_during_train": best_metric,
        "metrics_val_merged_best_ckpt": val_metrics_final,
        "metrics_test_merged_best_ckpt": test_metrics,
        "s1_threshold": args.s1_threshold,
    }
    report_path = os.path.join(args.output_dir, "metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)
    print(f"[report] saved {report_path}")


if __name__ == "__main__":
    main()

