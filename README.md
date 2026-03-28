# Final Year Project — Multi-Task Psychological Modeling & Risk Demo

End-to-end pipeline for **weakly supervised** multi-task learning on Chinese text: **S1** (problem types, multi-label), **S2** (mental state, single-label), **S3** (risk, single-label), built on **Chinese BERT + LoRA**, plus a small **FastAPI** service, **SQLite** persistence, **WebSocket** updates, and prototype UIs (**`wechat_mock.html`**, **Gradio**).

## Features

- **Model**: `MultiTaskBertForPsychology` — shared BERT encoder + three heads; training with **LoRA** (`peft`).
- **Training**: `train_multitask_lora.py` — stratified split, metrics, optional **S1 `pos_weight`**, **S2/S3 class weights**, task loss weights, **S1 threshold** (default `0.15`).
- **Inference**: `multitask_predict.py` — structured JSON (may include Chinese display keys from `multitask_config`).
- **Risk & policy**: `risk_scoring.py`, `policy_engine.py` — `action` such as `ai_reply` / `handoff_to_human`.
- **Backend**: `backend_api.py` — REST + WebSocket, `chat_app.db` (`conversations`, `messages`, `inference_logs`), `/monitor/latest` for Gradio monitor tab.
- **UI**: `wechat_mock.html` (chat + Moments); `admin.html` (queue takeover + LLM API settings stored in SQLite); `gradio.py` → `app_gradio.py` (local predict + backend log monitor).

## Repository layout (main files)

| Path | Role |
|------|------|
| `multitask_config.py` | Labels, weak rules (`map_s1` / `map_s2` / `map_s3`), formatting |
| `multitask_model.py` | Multi-head BERT model |
| `train_multitask_lora.py` | LoRA training |
| `multitask_predict.py` | CLI / library inference |
| `eval_only.py` | Standalone eval on CSV split |
| `merge_peft_to_best_model.py` | Merge adapter + base if training ends after saving checkpoints |
| `backend_api.py` | FastAPI app |
| `wechat_mock.html` | User chat + Moments demo client |
| `admin.html` | Admin queue + configurable reply LLM (`/admin/llm-config`) |
| `app_gradio.py` | Gradio UI |
| `gradio.py` | Thin launcher (avoids shadowing the `gradio` package) |
| `后端多任务模型说明文档.md` | Design notes (English body in repo version) |

## Environment

**Option A — Conda (recommended GPU stack)**

```bash
conda env create -f environment.yml
conda activate fyp
pip install -r requirements-pip.txt
```

**Option B — pip only**

```bash
pip install -r requirements-pip.txt
```

Install a **CUDA-enabled PyTorch** build matching your GPU / driver if you train on GPU.

## Data

Default Weibo CSV (relative to repo root):

`dataset/weibo/微博评论情感数据集(清洗之后的，有标注，中文,csv格式)/clean_weibo_text.csv`

Format: `label,text` (see dataset README for label meanings).

## Training (GPU example)

Smoke test:

```bash
python train_multitask_lora.py --device cuda --max_samples 500 --epochs 1 --batch_size 8 --max_length 128 --output_dir multitask_output_smoke
```

Full run (example):

```bash
python train_multitask_lora.py --device cuda --epochs 8 --batch_size 16 --max_length 128 --output_dir multitask_output_v3
```

Artifacts typically include:

- `best_model/` — merged weights for inference  
- `best_peft_adapter/` — LoRA adapter only  
- `best_full_base_with_heads/` — base + heads for merge step  

If training stops after saving `best_full_base_with_heads` + `best_peft_adapter` but before merge:

```bash
python merge_peft_to_best_model.py --output_dir multitask_output_v3
```

## Inference

```bash
python multitask_predict.py --model_dir multitask_output_v3/best_model --text "示例中文句子"
```

## Evaluation

```bash
python eval_only.py --model_dir multitask_output_v3/best_model --device cuda
```

## Backend API + realtime demo

Start API (port may fall back if `8010` is busy; check console):

```bash
python backend_api.py
```

Open `wechat_mock.html` in a browser. If the backend is not on `8010`, use:

`wechat_mock.html?api=http://127.0.0.1:<PORT>`

## Gradio

```bash
python gradio.py
```

- **Local model**: load `model_dir`, run predict on Chinese (or any) text.  
- **Backend monitor**: point at the same API base URL as `wechat_mock.html` and refresh logs.

## Label space (current code)

Align with `multitask_config.py`:

- **S1**: 8 multi-label problem types.  
- **S2**: `depression`, `anxiety`, `none` (3 classes).  
- **S3**: `suicide_ideation`, `self_harm`, `none` (3 classes).

## License

See [LICENSE](LICENSE) in the repository root (MIT if present).

## Citation / coursework

If this README is used for a thesis or report, cite your own institution’s rules; the technical design is summarized in `后端多任务模型说明文档.md`.
