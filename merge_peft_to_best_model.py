"""
若训练在「合并 LoRA → best_model」前中断，可用已有目录手动合并：

  python merge_peft_to_best_model.py --output_dir multitask_output_v3

需要目录内存在：
  - {output_dir}/best_full_base_with_heads/
  - {output_dir}/best_peft_adapter/
"""
from __future__ import annotations

import argparse
import json
import os

from transformers import BertTokenizer
from peft import PeftModel

from multitask_config import S1_LABELS, S2_LABELS, S3_LABELS
from multitask_model import MultiTaskBertForPsychology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    base_dir = os.path.join(args.output_dir, "best_full_base_with_heads")
    adapter_dir = os.path.join(args.output_dir, "best_peft_adapter")
    out_dir = os.path.join(args.output_dir, "best_model")

    tokenizer = BertTokenizer.from_pretrained(base_dir, local_files_only=True)
    base = MultiTaskBertForPsychology.from_pretrained(
        base_dir,
        local_files_only=True,
        num_s1_labels=len(S1_LABELS),
        num_s2_labels=len(S2_LABELS),
        num_s3_labels=len(S3_LABELS),
    )
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
