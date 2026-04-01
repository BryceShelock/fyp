"""
从 train_multitask_lora.py 写入的 training_log.jsonl 绘制曲线。

每行 JSON 可含（新训练才会有人均 train loss）：
  epoch, train_loss_mean, train_loss_s1_mean, train_loss_s2_mean, train_loss_s3_mean,
  val_metrics{...}, composite

用法:
  python plot_training_log.py --log multitask_output/training_log.jsonl --out training_curves.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str, default="multitask_output_efa/training_log.jsonl")
    p.add_argument("--out", type=str, default="training_curves.png")
    args = p.parse_args()

    rows = []
    with open(args.log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit(f"No rows in {args.log}")

    epochs = [r["epoch"] for r in rows]
    has_train = any("train_loss_mean" in r for r in rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    if has_train:
        tl = [r.get("train_loss_mean") for r in rows]
        ax = axes[0, 0]
        ax.plot(epochs, tl, "o-", color="C0", label="train total (weighted)")
        if any("train_loss_s1_mean" in r for r in rows):
            ax.plot(epochs, [r.get("train_loss_s1_mean") for r in rows], ".--", alpha=0.7, label="train L_s1")
            ax.plot(epochs, [r.get("train_loss_s2_mean") for r in rows], ".--", alpha=0.7, label="train L_s2")
            ax.plot(epochs, [r.get("train_loss_s3_mean") for r in rows], ".--", alpha=0.7, label="train L_s3")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (mean over batches)")
        ax.set_title("Training loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "No train_loss_* in log.\nRetrain with latest train_multitask_lora.py", ha="center")
        axes[0, 0].set_axis_off()

    vm = [r["val_metrics"] for r in rows]
    ax = axes[0, 1]
    ax.plot(epochs, [m["s1_accuracy"] for m in vm], "o-", label="val s1_acc")
    ax.plot(epochs, [m["s2_accuracy"] for m in vm], "s-", label="val s2_acc")
    ax.plot(epochs, [m["s3_accuracy"] for m in vm], "^-", label="val s3_acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Validation accuracy (three tasks)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [m["s1_precision_macro"] for m in vm], "o-", label="S1 prec_macro")
    ax.plot(epochs, [m["s1_recall_macro"] for m in vm], "o--", label="S1 rec_macro")
    ax.plot(epochs, [m["s2_precision_macro"] for m in vm], "s-", label="S2 prec_macro")
    ax.plot(epochs, [m["s2_recall_macro"] for m in vm], "s--", label="S2 rec_macro")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Val S1 / S2 macro P/R")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, [m["s3_precision_macro"] for m in vm], "^-", label="S3 prec_macro")
    ax.plot(epochs, [m["s3_recall_macro"] for m in vm], "^--", label="S3 rec_macro")
    ax.plot(epochs, [r["composite"] for r in rows], "k:", linewidth=2, label="composite(s1+s2+s3 acc)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Val S3 macro P/R + composite")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(os.path.abspath(args.log), fontsize=9)
    out_path = os.path.abspath(args.out)
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
