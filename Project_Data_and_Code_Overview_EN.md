# Project Data & Code Overview

*(Chinese version: `项目数据与代码总览.md`)*

This document summarizes: the **label taxonomy**, **class proportions in the current CSVs (statistical snapshot)**, **roles of main code files**, and **underlying ideas**. Counts follow the EFA-aligned definitions in `multitask_config.py` and were obtained by tallying the CSVs on a local machine; **re-run statistics after replacing or re-exporting data**.

---

## 1. Conceptual overview (what the project implements)

| Module | Idea |
|--------|------|
| **Multi-task BERT** | Shared encoder (Chinese BERT, etc.) + **three classification heads**: S1, S2, S3 predicted jointly with a combined loss. |
| **S1** | **19-class single-label** (worry types 1.1–1.19), cross-entropy; optional class weights and sampling. |
| **S2** | **8-class single-label** (mental-health dimension 2.1–2.8, including “not at disorder level”), cross-entropy. |
| **S3** | **6-class single-label** (SOS urgency 3.1–3.6), cross-entropy; **continuous `risk_score` is mainly a weighted sum of the S3 softmax**, plus rule phrases and optional history. |
| **LoRA** | Low-rank adapters fine-tune the backbone with fewer trainable parameters; merged weights: `merge_peft_to_best_model.py`. |
| **Inference policy** | `multitask_predict`: `risk_scoring` produces a continuous score and low/medium/high tiers; if S3 argmax falls in 3.1–3.5 and confidence ≥ threshold, **force human handoff** (default threshold aligned with `risk_scoring.RISK_LEVEL_HIGH_THRESHOLD`). |
| **Product policy** | `policy_engine.decide_policy` maps risk tier to `action` and resource slots; `backend_api` can chain LLM, sessions, and human queue. |

---

## 2. Label taxonomy (all classes and indices)

Indices match the list order in **`multitask_config.py`**. Raw EFA tag strings `1.x` / `2.x` / `3.x` map to integers via `efa_tag_to_*_index`.

### 2.1 S1: Worry type (19 classes, `s1` ∈ 0..18)

| Index | EFA | Display (ZH) |
|------|-----|----------------|
| 0 | 1.1 | Academics / planning |
| 1 | 1.2 | Career / work |
| 2 | 1.3 | Family / conflict |
| 3 | 1.4 | Substance abuse / addiction |
| 4 | 1.5 | Grief |
| 5 | 1.6 | Insomnia |
| 6 | 1.7 | Stress |
| 7 | 1.8 | Interpersonal |
| 8 | 1.9 | Romantic relationship |
| 9 | 1.10 | Divorce |
| 10 | 1.11 | Breakup |
| 11 | 1.12 | Self-exploration |
| 12 | 1.13 | Low self-esteem |
| 13 | 1.14 | Adolescent issues |
| 14 | 1.15 | OCD |
| 15 | 1.16 | Other worries |
| 16 | 1.17 | LGBT+ |
| 17 | 1.18 | Sexuality |
| 18 | 1.19 | Parent–child |

### 2.2 S2: Suspected mental disorder (8 classes, `s2` ∈ 0..7)

| Index | EFA | Display (ZH) |
|------|-----|----------------|
| 0 | 2.1 | Depression (suspected) |
| 1 | 2.2 | Anxiety (suspected) |
| 2 | 2.3 | Bipolar (suspected) |
| 3 | 2.4 | PTSD (suspected) |
| 4 | 2.5 | Panic (suspected) |
| 5 | 2.6 | Eating disorder (suspected) |
| 6 | 2.7 | Not at mental-disorder level |
| 7 | 2.8 | Other disorder (suspected) |

### 2.3 S3: SOS / bodily harm urgency (6 classes, `s3` ∈ 0..5)

| Index | EFA | Internal id | Display (ZH) |
|------|-----|-------------|----------------|
| 0 | 3.1 | efa_s3_suicide_action | Suicide in progress |
| 1 | 3.2 | efa_s3_suicide_ideation | Suicide ideation / planning |
| 2 | 3.3 | efa_s3_self_harm | Self-harm |
| 3 | 3.4 | efa_s3_harm_others_ongoing | Ongoing harm to others |
| 4 | 3.5 | efa_s3_harm_others_planned | Planned harm to others |
| 5 | 3.6 | efa_s3_none | No urgent bodily-harm tendency |

**Policy convention:** indices **0–4** (3.1–3.5) may trigger **forced human handoff** at inference (see `S3_FORCE_HANDOFF_INDICES`).

---

## 3. Data statistics snapshot (CSVs in this repo)

Row counts and proportions below were computed on **2026-03-28** for **`dataset_efaqa_multitask.csv`** (20,000 rows) and **`dataset_multitask.csv`** (400,000 rows, parsed via `multitask_data.load_multitask_supervision`).

### 3.1 `dataset_efaqa_multitask.csv` (EFA export / aligned corpus)

- **Columns:** `text, s1, s2, s3, md5`
- **Rows:** 20,000

**S3 (critical)** — extremely imbalanced:

| s3 | Label (EN gloss) | Count | Share |
|----|------------------|-------|-------|
| 0 | Suicide in progress | 0 | 0% |
| 1 | Suicide ideation | 247 | 1.24% |
| 2 | Self-harm | 88 | 0.44% |
| **3** | **Ongoing harm to others** | **19,612** | **98.06%** |
| 4 | Planned harm to others | 4 | 0.02% |
| 5 | None (no urgent harm) | 49 | 0.25% |

**Implication:** Under this distribution, a model that fits the training prior will heavily favor predicting S3 **index 3 (3.4)**, which clashes with everyday “stress / venting” text that should often be **3.6**. **First check:** thread-level `label.s3` semantics in the corpus, correctness of the export mapping, or whether the subset is pathologically skewed—**do not blame post-processing weights alone**.

**S2 summary:** index 6 (“not at disorder level”) ≈ **89.0%**; all other classes ≈ 11% combined.

**S1 summary:** index 8 (“relationship”) ≈ **24.8%**; index 15 (“other”) ≈ **35.8%**; for the rest use plots from `visualize_efaqa_dataset.py` or `value_counts` on the CSV.

### 3.2 `dataset_multitask.csv` (large mixed table)

- **Columns:** `text, s1, s2, s3, source` (`s1` etc. may be strings; parsed by `multitask_data`)
- **Rows:** 400,000 (loader format `explicit_s123`)

**S3 (critical)** — again extremely skewed, **almost all index 4**:

| s3 | Label (EN gloss) | Count | Share |
|----|------------------|-------|-------|
| 0–3 | (several) | 0 / tiny | ≈0 |
| **4** | **Planned harm to others** | **399,248** | **99.81%** |
| 5 | None | 0 | 0% |

**Implication:** This table is **not suitable** as a balanced six-way S3 training set; training S3 on it biases the model toward **3.5 (index 4)**. The **dominant S3 majority class differs** from the EFA subset (mostly **3.4 / index 3**), so mixing or swapping files without relabeling causes **behavior drift**.

**S1/S2 summary:** S1 “other” ≈ **84.1%**; S2 “not at disorder level” ≈ **88.2%**; many classes have zero count—narrow label coverage.

---

## 4. Other data & resources (paths)

| Path / file | Role |
|-------------|------|
| `export_efaqa_to_csv.py` | Export CSV with `label.s1/s2/s3` from `efaqa_corpus_zh.load()` (license env var required). |
| `dataset/weibo/.../clean_weibo_text.csv` | Weibo sentiment weak labels; `multitask_data` can use `text,label` columns. |
| `dataset/chinese-chatbot-corpus/` | Third-party dialogue tooling; `build_training_dataset.py` can scan `clean_chat_corpus/*.tsv` into weak-label CSV. |
| `build_training_dataset.py` | Merge chat + Weibo into multitask CSV; S3 weak labels: `multitask_config.map_s3_weak` (defaults to **3.6** if no keyword hit). |
| `wechat_mock.html` / `admin.html` | Front-end mocks; same-origin mount in `backend_api`. |

---

## 5. Code file index (repo root focus)

### 5.1 Config & data

| File | Role |
|------|------|
| **`multitask_config.py`** | **Single source of truth:** `S1/S2/S3_LABELS`, ZH/EN-style display maps, `efa_tag_to_*_index`, weak supervision `map_s1/map_s2/map_s3`, `format_prediction`, `DEFAULT_INFERENCE_MODEL_DIR`. |
| **`multitask_data.py`** | Load CSV: explicit `text,s1,s2,s3` or weak `text,label`; normalize to integer labels for training. |

### 5.2 Model & training

| File | Role |
|------|------|
| **`multitask_model.py`** | `MultiTaskBertForPsychology`: shared BERT + three linear heads; forward returns `s1/s2/s3_logits` and joint loss. |
| **`train_multitask_lora.py`** | Main training: DataLoader, optional S2/S3 **class weights**, optional **`WeightedRandomSampler`** (S3 crisis vs 3.6), optional **S3 class-balanced sampler**, `s3_loss_weight`, **S3 focal loss / label smoothing**, **best checkpoint by macro-F1**, `training_log.jsonl`, adapter + three-head checkpoints. |
| **`merge_peft_to_best_model.py`** | Merge LoRA + backbone + heads into a standalone `best_model` for `from_pretrained`. |
| **`eval_only.py`** | Evaluation only: accuracy, macro P/R/F1, confusion matrices; optional plots and JSON. |
| **`plot_training_log.py`** | Plot train/val curves from `training_log.jsonl`. |
| **`visualize_efaqa_dataset.py`** | EDA plots for EFA-style CSV (S1/S2/S3, text length); does **not** call `efaqa.load()`. |

### 5.3 Inference, risk & policy

| File | Role |
|------|------|
| **`multitask_predict.py`** | **Main inference:** tokenizer + forward, S3 temperature softmax, **`risk_scoring`**, **handoff gate**, JSON output (including `s3_probs_zh`). |
| **`risk_scoring.py`** | Weighted sum over six S3 softmax probs + rule boosts + `none` damping + optional history → `risk_score` and tiers; **`RISK_LEVEL_HIGH_THRESHOLD = 0.60`**. |
| **`policy_engine.py`** | Map `RiskResult` to `action` (AI / AI+hotline / handoff) and `resources`. |

### 5.4 Services & demos

| File | Role |
|------|------|
| **`backend_api.py`** | FastAPI: sessions, WebSocket, admin, **`predict_text`**, static pages, optional LLM (`chat_llm.py`). |
| **`chat_llm.py`** | LLM client wrapper (config via env / admin routes). |
| **`app_gradio.py`** | Gradio UI: `model_dir` / device, calls `predict_text`. |
| **`gradio.py`** | Thin entry to avoid name clash with the `gradio` package; calls `app_gradio.main()`. |

### 5.5 Dataset build & misc. scripts

| File | Role |
|------|------|
| **`export_efaqa_to_csv.py`** | Official EFA corpus → this repo’s CSV format. |
| **`build_training_dataset.py`** | Build weak-supervision multitask CSV from chat TSV + Weibo. |
| **`download.py`** | Download helpers (project-specific). |
| **`modle_train.py`** | Legacy / alternate training script (filename typo kept). |
| **`wechat_data_simulate.py` / `data_simulate.py`** | Data or session simulation helpers. |

### 5.6 Documentation

| File | Role |
|------|------|
| **`README.md`** | Quick start, dependencies, common commands. |
| **`后端多任务模型说明文档.md`** | Backend model spec (Chinese); API, risk, training detail. |
| **`项目数据与代码总览.md`** | This content in Chinese: full label tables, proportion snapshot, code map. |
| **`Project_Data_and_Code_Overview_EN.md`** | **This file** (English). |

---

## 6. Reproducing proportion statistics

```bash
python visualize_efaqa_dataset.py --csv dataset_efaqa_multitask.csv --out efaqa_eda.png
```

Or in Python, use `value_counts` / `bincount` on `s1,s2,s3` in `dataset_efaqa_multitask.csv`. For **`dataset_multitask.csv`**, always parse via **`load_multitask_supervision`** before counting, so string `s1` cells are not misread.

---

## 7. Link to “always predicted 3.4 / high risk”

In the current **`dataset_efaqa_multitask.csv` snapshot**, **~98% of `s3` labels are index 3 (3.4 ongoing harm to others)** and **3.6 is ~0.25%**. Minimizing cross-entropy on this distribution **incentivizes predicting class 3 often**. That mismatches common sense for benign stress text—the **root cause is the supervision distribution**, not `risk_score` alone. Mitigations: **verify official EFA `label.s3` and the exporter**, **add many 3.6 and balanced crisis examples**, **tune loss weights and sampling**, and if needed **retrain the S3 head or full model**.

---

## 8. Improving `risk_score` and the model (priority order)

**Principle:** Data and S3 calibration come first; `risk_score` and inference gates **reshape** behavior on top of existing logits.

### 8.1 Data & labels (highest priority)

- **Verify** thread-level EFA `label.s3` and `export_efaqa_to_csv.py` against the intended 3.x semantics; **~98% on 3.4** is suspicious—confirm real prevalence vs. export/field bugs.
- **Add many `s3=5` (3.6)** everyday / school / interpersonal lines so all six classes are learnable; aim for a **substantial 3.6 share** in training (e.g. 30%+ depending on product goals).
- **Avoid** blindly mixing `dataset_multitask.csv` (almost all 3.5) with the EFA slice unless S3 labels are unified or relabeled.

### 8.2 Training (supported in code)

| Knob | Effect |
|------|--------|
| `--s3_label_smoothing` (e.g. 0.05–0.1) | Softer S3 softmax (**ignored if `--s3_focal_gamma` > 0**). |
| **`--s3_focal_gamma`** (e.g. 1.5–2) | S3 **focal loss**; down-weights easy examples; can combine with `--use_s3_class_weight`. |
| Lower `--s3_loss_weight` (e.g. 5→2) | Reduces S3 dominance over S1/S2 and the representation. |
| Turn off `--use_weighted_sampler` | On already skewed S3, extra crisis upsampling can **increase** benign false alarms. |
| `--use_s3_class_weight` | Strong weights for rare classes; tune together with smoothing, lower S3 weight, and more data. |
| **`--use_s3_class_balanced_sampler`** | Inverse-frequency sampling over **six** S3 classes; often **raises macro-F1, lowers plain accuracy**. |
| **`--best_select s3_macro_f1`** or **`macro_f1_sum`** | Pick best checkpoint by **S3 macro-F1** or **sum of three macro-F1s**, not sum of accuracies. |
| Tune `MULTITASK_S3_SOFTMAX_TEMPERATURE` on val | Inference temperature + calibration search. |

### 8.3 `risk_score` & inference gates (env vars)

| Variable | Role |
|----------|------|
| **`MULTITASK_RISK_UNIFORM_BLEND`** (e.g. 0.12–0.2) | Blend S3 probs with uniform: `p'=(1-ε)p+ε/6` before weighted sum (**default 0**). |
| **`MULTITASK_RISK_NONE_DAMP_SCALE`** (default `0.10`) | Scales downward pull from `p_none`; try 0.15–0.25 if `p_none` is underestimated. |
| `MULTITASK_S3_SOFTMAX_TEMPERATURE` | Soften S3 logits before softmax in `multitask_predict` (grid search). |
| `MULTITASK_S3_HANDOFF_MIN_PROB` | Min softmax prob for forced handoff when argmax is 3.1–3.5 (default aligned with **0.60**). |

**Note:** Blend / temperature ease symptoms; **stable 3.6 vs. crisis separation still needs labels and training**.

### 8.4 Advanced (optional)

- **Temperature scaling** or vector calibration on the validation set, then feed calibrated probs into `risk_score`.
- **Multi-task confidence fusion** (e.g. raise risk only if S2 ≠ “not disorder” **and** S3 is high)—product-dependent, watch for missed crises.

---

## 9. Default inference model path

The repo uses **`multitask_config.DEFAULT_INFERENCE_MODEL_DIR`** (currently `multitask_output_efa/best_model`). For production training use **`--output_dir multitask_output_efa`** so merged weights match backend / Gradio / `eval_only` defaults; to change the folder, edit **that constant** in one place.

---

*Proportion numbers in §3 depend on the CSVs in the repo; re-run counts and update §3 after data changes.*
