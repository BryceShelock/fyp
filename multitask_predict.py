import os
import json
import re
from typing import Optional, Tuple, Dict

import torch
from transformers import BertTokenizer

from multitask_config import (
    DEFAULT_INFERENCE_MODEL_DIR,
    S3_LABELS,
    S3_ID2ZH,
    S3_FORCE_HANDOFF_INDICES,
    format_prediction,
)
from multitask_model import MultiTaskBertForPsychology
from risk_scoring import RISK_LEVEL_HIGH_THRESHOLD, compute_risk_score
from policy_engine import PolicyDecision, decide_policy


_MODEL_CACHE: Dict[Tuple[str, str], tuple] = {}
_PRINTED_DEVICE: set = set()

# S3 预测为 3.1–3.5 时，只有 softmax 概率 ≥ 此阈值才「强制转人工」。
# 默认与 risk_scoring 的「高风险」阈值 RISK_LEVEL_HIGH_THRESHOLD 一致（现为 0.60），与 risk_score 语义统一。
# 若需更敏感（宁可误报也要抓疑似危机），可设环境变量为 0.42 等更低值。
_DEFAULT_S3_HANDOFF_MIN_PROB = float(
    os.environ.get("MULTITASK_S3_HANDOFF_MIN_PROB", str(RISK_LEVEL_HIGH_THRESHOLD))
)
# 对 S3 logits 先除以 T 再 softmax（T>1 分布更平），可缓解域外短句（如「你好」）上 argmax 过置信导致的误报。默认 1.0 不改变行为。
_DEFAULT_S3_SOFTMAX_TEMPERATURE = float(os.environ.get("MULTITASK_S3_SOFTMAX_TEMPERATURE", "1.0"))


def _strip_bracket_event_prefix(text: str) -> str:
    """去掉 wechat mock 传入的 [moments] / [status] 前缀，再送入 BERT，减少分布偏移误触发。"""
    raw = (text or "").strip()
    m = re.match(r"^\[(moments|status)\]\s*", raw, flags=re.I)
    return raw[m.end() :].strip() if m else raw


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
    s3_handoff_min_prob: Optional[float] = None,
    s3_softmax_temperature: Optional[float] = None,
    strip_event_prefix: bool = True,
):
    tokenizer, model, device = load_model(model_dir, device=device)

    text_in = _strip_bracket_event_prefix(text) if strip_event_prefix else (text or "").strip()
    if not text_in:
        text_in = (text or "").strip() or " "

    min_p = _DEFAULT_S3_HANDOFF_MIN_PROB if s3_handoff_min_prob is None else float(s3_handoff_min_prob)
    s3_temp = (
        _DEFAULT_S3_SOFTMAX_TEMPERATURE
        if s3_softmax_temperature is None
        else float(s3_softmax_temperature)
    )
    s3_temp = max(s3_temp, 1e-6)

    inputs = tokenizer(
        text_in,
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
    s3_scaled = s3_logits / s3_temp

    s1_pred_id = int(torch.argmax(s1_logits, dim=-1)[0].item())

    s2_pred_id = int(torch.argmax(s2_logits, dim=-1)[0].item())
    s3_pred_id = int(torch.argmax(s3_scaled, dim=-1)[0].item())

    # S3 概率（temperature softmax）用于 argmax / risk_score / 转人工门控
    s3_probs_tensor = torch.softmax(s3_scaled, dim=-1)[0].float().cpu()
    s3_probs = {S3_LABELS[i]: float(s3_probs_tensor[i].item()) for i in range(len(S3_LABELS))}
    s3_pred_prob = float(s3_probs_tensor[s3_pred_id].item())
    # Gradio 调试：一眼看六类 softmax，判断是不是「3.6 无紧急」被压成 0
    s3_probs_zh = {
        S3_ID2ZH.get(S3_LABELS[i], S3_LABELS[i]): round(float(s3_probs_tensor[i].item()), 4)
        for i in range(len(S3_LABELS))
    }
    risk = compute_risk_score(text=text_in, s3_probs=s3_probs, history_scores=history_scores)

    force_s3_handoff = s3_pred_id in S3_FORCE_HANDOFF_INDICES and s3_pred_prob >= min_p

    if force_s3_handoff:
        decision = PolicyDecision(
            action="handoff_to_human",
            ui_tags={"风险等级": "高风险"},
            resources=[
                {
                    "type": "warning",
                    "title": "高风险提示",
                    "value": "S3 紧急类（EFA 3.1–3.5）：已强制转接人工审核。",
                },
                {
                    "type": "hotline",
                    "title": "紧急求助",
                    "value": "如有立即危险请联系当地紧急电话或身边可信任的人。",
                },
            ],
        )
    else:
        decision = decide_policy(risk)

    result = format_prediction(s1_pred_id, s2_pred_id, s3_pred_id, s1_threshold=s1_threshold)
    result["risk_score"] = round(float(risk.score), 4)
    result["风险等级"] = "高风险" if force_s3_handoff else risk.level
    result["action"] = decision.action
    result["resources"] = decision.resources
    result["_risk_reasons"] = risk.reasons
    result["s1_pred_id"] = s1_pred_id
    result["s2_pred_id"] = s2_pred_id
    result["s3_pred_id"] = s3_pred_id
    result["s3_pred_prob"] = round(s3_pred_prob, 4)
    result["s3_handoff_min_prob"] = round(min_p, 4)
    result["s3_force_handoff"] = bool(force_s3_handoff)
    result["_text_used_for_model"] = text_in
    result["s3_softmax_temperature"] = round(s3_temp, 4)
    result["s3_probs_zh"] = s3_probs_zh
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=DEFAULT_INFERENCE_MODEL_DIR)
    parser.add_argument("--text", type=str, default="我压力好大，作业又好多，朋友也不想理我")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--s1_threshold", type=float, default=0.15)
    parser.add_argument(
        "--s3_handoff_min_prob",
        type=float,
        default=None,
        help=f"S3 为 3.1–3.5 时 softmax 概率≥此值才强制转人工；默认与 risk_score 高风险阈值一致（{RISK_LEVEL_HIGH_THRESHOLD}），或读 MULTITASK_S3_HANDOFF_MIN_PROB",
    )
    parser.add_argument(
        "--s3_softmax_temperature",
        type=float,
        default=None,
        help="S3 logits 温度缩放（>1 更平）；默认读 MULTITASK_S3_SOFTMAX_TEMPERATURE 或 1.0",
    )
    args = parser.parse_args()

    result = predict_text(
        text=args.text,
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
        s1_threshold=args.s1_threshold,
        s3_handoff_min_prob=args.s3_handoff_min_prob,
        s3_softmax_temperature=args.s3_softmax_temperature,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

