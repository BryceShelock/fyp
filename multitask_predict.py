import os
import json
from typing import Optional, Tuple, Dict

import torch
from transformers import BertTokenizer

from multitask_config import (
    S1_LABELS,
    S2_LABELS,
    S3_LABELS,
    format_prediction,
)
from multitask_model import MultiTaskBertForPsychology
from risk_scoring import compute_risk_score
from policy_engine import decide_policy


_MODEL_CACHE: Dict[Tuple[str, str], tuple] = {}
_PRINTED_DEVICE: set = set()


def load_model(model_dir: str, device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError("当前环境未检测到 CUDA（torch.cuda.is_available()==False），无法使用 GPU。")

    cache_key = (os.path.abspath(model_dir), device)
    if cache_key in _MODEL_CACHE:
        tokenizer, model = _MODEL_CACHE[cache_key]
        return tokenizer, model, device

    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = MultiTaskBertForPsychology.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    # GPU 上可用 half 精度加速（对 demo 很明显）
    if device.startswith("cuda"):
        try:
            model.half()
        except Exception:
            pass

    # 只打印一次，方便确认实际使用设备
    if cache_key not in _PRINTED_DEVICE:
        _PRINTED_DEVICE.add(cache_key)
        print(f"[predict] loaded model_dir={model_dir} on device={device}")

    _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model, device


@torch.no_grad()
def predict_text(
    text: str,
    model_dir: str,
    device: str = "auto",
    max_length: int = 128,
    s1_threshold: float = 0.15,
    history_scores: Optional[list[float]] = None,
):
    tokenizer, model, device = load_model(model_dir, device=device)

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

    out = model(**inputs)
    s1_logits = out["s1_logits"]
    s2_logits = out["s2_logits"]
    s3_logits = out["s3_logits"]

    s1_probs = torch.sigmoid(s1_logits)[0].cpu().numpy()
    s1_pred_ids = [i for i, p in enumerate(s1_probs) if p >= s1_threshold]

    s2_pred_id = int(torch.argmax(s2_logits, dim=-1)[0].item())
    s3_pred_id = int(torch.argmax(s3_logits, dim=-1)[0].item())

    # S3 概率（softmax）用于 risk_score
    s3_probs_tensor = torch.softmax(s3_logits, dim=-1)[0].float().cpu()
    s3_probs = {S3_LABELS[i]: float(s3_probs_tensor[i].item()) for i in range(len(S3_LABELS))}
    risk = compute_risk_score(text=text, s3_probs=s3_probs, history_scores=history_scores)
    decision = decide_policy(risk)

    result = format_prediction(s1_pred_ids, s2_pred_id, s3_pred_id, s1_threshold=s1_threshold)
    result["risk_score"] = round(float(risk.score), 4)
    result["风险等级"] = risk.level  # 用 score 的等级覆盖（更符合业务）
    result["action"] = decision.action
    result["resources"] = decision.resources
    result["_risk_reasons"] = risk.reasons  # 便于调试/论文解释；前端可不展示
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="multitask_output/best_model")
    parser.add_argument("--text", type=str, default="我压力好大，作业又好多，朋友也不想理我")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--s1_threshold", type=float, default=0.15)
    args = parser.parse_args()

    result = predict_text(
        text=args.text,
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
        s1_threshold=args.s1_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

