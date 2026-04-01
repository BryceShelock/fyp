"""
若训练在「合并 LoRA → best_model」前中断，可用已有目录手动合并：

  python merge_peft_to_best_model.py --output_dir multitask_output

需要目录内存在：
  - {output_dir}/converted_bert_pytorch/   （训练时写入的骨干快照）
  - {output_dir}/best_full_base_with_heads/  （含 LoRA 键名；本脚本只从中读取三头权重）
  - {output_dir}/best_peft_adapter/

说明：不能仅用 best_full_base_with_heads 做 from_pretrained，否则 BERT 权重键名与标准 BERT 不一致，
会导致合并后推理指标异常（与 train_multitask_lora.py 末尾合并逻辑一致）。
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import BertTokenizer

from multitask_config import S1_LABELS, S2_LABELS, S3_LABELS
from multitask_model import MultiTaskBertForPsychology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    base_full = os.path.join(output_dir, "best_full_base_with_heads")
    adapter_dir = os.path.join(output_dir, "best_peft_adapter")
    out_dir = os.path.join(output_dir, "best_model")
    backbone_dir = os.path.join(output_dir, "converted_bert_pytorch")

    bin_path = os.path.join(base_full, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"缺少三头权重: {bin_path}")

    if not os.path.isdir(backbone_dir):
        raise FileNotFoundError(f"缺少骨干目录: {backbone_dir}")

    raw_sd = torch.load(bin_path, map_location="cpu")
    head_sd = {
        k: v
        for k, v in raw_sd.items()
        if k.startswith(("s1_head.", "s2_head.", "s3_head."))
    }
    if len(head_sd) < 6:
        raise RuntimeError(f"{bin_path} 中未找到完整 s1/s2/s3_head 权重")

    tokenizer = BertTokenizer.from_pretrained(backbone_dir, local_files_only=True)
    base = MultiTaskBertForPsychology.from_pretrained(
        backbone_dir,
        local_files_only=True,
        num_s1_labels=len(S1_LABELS),
        num_s2_labels=len(S2_LABELS),
        num_s3_labels=len(S3_LABELS),
    )
    miss, unexpected = base.load_state_dict(head_sd, strict=False)
    if miss or unexpected:
        print(f"[merge] load heads strict=False missing={len(miss)} unexpected={len(unexpected)}")

    peft_wrap = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    merged = peft_wrap.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=False)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "multitask_labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"S1_LABELS": S1_LABELS, "S2_LABELS": S2_LABELS, "S3_LABELS": S3_LABELS},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Merged model saved to: {out_dir}")


if __name__ == "__main__":
    main()
