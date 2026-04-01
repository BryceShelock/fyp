# FYP Progress Report — Full-English PPT Outline & Slide Content

**Purpose:** Slide-by-slide blueprint for a progress presentation, then thesis draft planning.  
**Project working title (from repo docs):** *WeChat-Style Prototype for Fine-Grained Psychological Risk Detection and Lightweight Response*

---

## Recommended talk length

- **15–20 slides** for ~12–18 minutes + Q&A  
- **Heaviest emphasis:** Slides in **Part B** (what is built and demonstrated)

---

## Part A — Opening (Slides 1–3)

### Slide 1 — Title

**Title:** Fine-Grained Psychological Risk Detection — Progress Report  
**Subtitle:** Multi-task BERT + Human-in-the-Loop Prototype (WeChat-style UI)  
**Your name, ID, supervisor, department, date**

**Suggested figure:** Clean title slide only, or small project logo / WeChat-style mock screenshot (blurred if needed for privacy).

---

### Slide 2 — Motivation & problem

**Bullets:**

- Mental health support in messaging apps needs **scalable triage** without replacing clinicians.
- Goal: **detect** fine-grained distress categories (EFA-aligned), estimate **risk**, and route **high-risk** cases to **human moderators** while keeping **low-friction** AI replies when safe.
- Scope: **Chinese** short-text (chat, moments/status events); **prototype**, not a medical device.

**Suggested figure:** Simple diagram: User message → Model + Policy → {AI reply | Human queue}.

---

### Slide 3 — Objectives (what “done” means for this FYP)

**Bullets:**

1. **Multi-task model:** S1 (worry type), S2 (mental-health dimension), S3 (SOS / harm urgency) on shared BERT + **LoRA** fine-tuning.  
2. **Risk & policy layer:** Continuous `risk_score`, discrete levels, **keyword rules**, **forced handoff** for critical S3 classes.  
3. **End-to-end demo:** Backend API, **SQLite** persistence, **WebSocket** updates, **WeChat mock** + **Admin console**, optional **external LLM** for replies and EN translation.  
4. **Evaluation & documentation:** Train/val/test split, metrics, confusion matrices, reproducible commands.

**Suggested figure:** Checklist icon set or three-column “Model | Policy | System” table.

---

## Part B — Implemented work (core progress) (Slides 4–12)

### Slide 4 — High-level system architecture

**Bullets:**

- **Client layer:** `wechat_mock.html` (user), `admin.html` (moderator + LLM settings).  
- **API layer:** FastAPI (`backend_api.py`) — REST + SSE (chat stream) + WebSocket.  
- **Detection:** `multitask_predict.py` → BERT multi-head (`multitask_model.py`).  
- **Data:** SQLite — conversations, messages, user events, LLM config.  
- **Optional:** `chat_llm.py` + `app_gradio.py` for offline model testing.

**Suggested figure:** **Architecture diagram** (boxes + arrows). Export from draw.io / PowerPoint SmartArt. Label: HTTP, WS, DB, model path `…/best_model`.

---

### Slide 5 — Multi-task learning design (S1 / S2 / S3)

**Bullets:**

- **Shared encoder:** Chinese BERT (or MentalBERT path in config).  
- **S1:** 19 classes — EFA worry types 1.1–1.19 (single-label CE).  
- **S2:** 8 classes — EFA mental-health 2.1–2.8 (single-label CE).  
- **S3:** 6 classes — EFA SOS 3.1–3.6 (single-label CE); classes **3.1–3.5** drive **mandatory human handoff** when confident enough.  
- **Training:** Joint loss with configurable weights; **LoRA** on attention projections; heads fully trained; **merge** script produces standalone `best_model`.

**Suggested figure:** BERT block + three heads branching; small table of class counts (optional, can point to “data issues” later).

---

### Slide 6 — Risk scoring & policy (beyond raw logits)

**Bullets:**

- **`risk_score`:** Weighted combination of **S3 softmax** over six classes + **phrase-based boosts** (e.g. self-harm / hopelessness) + optional **history** + **`none`-damping**.  
- **Levels:** Low / medium / high thresholds (e.g. 0.30 / 0.60).  
- **`policy_engine`:** Maps risk to `action` (AI only, AI + resources, handoff).  
- **Inference knobs:** S3 softmax temperature, optional uniform blend for stability; handoff min-probability aligned with high-risk threshold by default.

**Suggested figure:** Flowchart: `s3_probs` → `compute_risk_score` → `decide_policy` OR `force_handoff`.

---

### Slide 7 — Dataset & training pipeline

**Bullets:**

- **Primary supervised CSV:** EFA-derived `dataset_efaqa_multitask.csv` (`text`, `s1`, `s2`, `s3`) via `export_efaqa_to_csv.py` (licensed corpus).  
- **Loader:** `multitask_data.py` — also supports weak-label Weibo format.  
- **Training script:** `train_multitask_lora.py` — stratified split, class weights, optional **S3 class-balanced sampler**, **macro-F1-based checkpoint selection**, **label smoothing** / **focal loss** for S3, logging to `training_log.jsonl`.  
- **Evaluation:** `eval_only.py` — accuracy + macro P/R/F1 + confusion matrices.

**Suggested figure:** Pipeline icons: CSV → DataLoader → LoRA train → merge → `best_model` → `eval_only` plots (use **one real confusion matrix** screenshot if allowed).

---

### Slide 8 — Backend services (implemented)

**Bullets:**

- **Chat:** POST `/chat/send` (streaming SSE), prediction injection, handoff to `waiting_admin`.  
- **User events:** Moments / status → prefixed user text → same pipeline.  
- **Admin:** Queue, takeover, release to AI (with **English system message** to user), reply, delete/clear messages, DB viewer.  
- **LLM admin:** SQLite-stored API base, model id, streaming toggle, named presets, test ping.  
- **Translation:** POST `/chat/translate` for EN line under chat bubbles (requires LLM enabled).

**Suggested figure:** Screenshot collage: **Admin queue** + **WeChat chat** (annotate arrows); or API route list from `/docs`.

---

### Slide 9 — Real-time & UX features

**Bullets:**

- **WebSocket:** User and admin rooms; conversation updates, queue refresh, release notifications.  
- **Bilingual display:** Chinese bubble + English subtitle (translation / fallback message if LLM off).  
- **Moderation flow:** Handoff → user sees system hint → admin **Release to AI** → user sees **“W (AI) is handling messages again”** (English).

**Suggested figure:** **Sequence diagram** (User / API / Admin / DB) for handoff + release.

---

### Slide 10 — Gradio & CLI tooling

**Bullets:**

- **Gradio** (`app_gradio.py`): Local `model_dir`, device, raw JSON prediction for quick experiments.  
- **CLI:** `multitask_predict.py` for single-text JSON output.  
- **Plots:** `plot_training_log.py`, `visualize_efaqa_dataset.py` for EDA.

**Suggested figure:** Screenshot of Gradio UI with one benign and one high-risk example (anonymized).

---

### Slide 11 — Documentation & reproducibility

**Bullets:**

- **`README.md`:** Run commands, eval, merge.  
- **`后端多任务模型说明文档.md`:** Model-centric specification (tasks, loss, inference).  
- **`项目数据与代码总览.md`:** Label taxonomy, data statistics snapshot, code file index, optimization notes.  
- **This PPT outline:** Progress narrative for defense + thesis structure.

**Suggested figure:** Three document icons with one-line roles.

---

### Slide 12 — Current quantitative snapshot (honest)

**Bullets:**

- Report **latest** `metrics_report.json` / `eval_only` numbers after your best run (e.g. `multitask_output_efa`).  
- State clearly: **S3 accuracy can be misleading** under class imbalance; **macro-F1** and **per-class confusion** matter more.  
- If improved after balanced sampling / focal / new data, say **before vs after** in one mini-table.

**Suggested figure:** **Bar chart** of macro-F1 for S1/S2/S3; or three small confusion-matrix thumbnails.

---

## Part C — Problems, solutions, open issues (Slides 13–16)

### Slide 13 — Problem: S3 label distribution & false “high risk”

**Problem:**

- EFA CSV snapshot showed **~98%** of rows labeled S3 class **3.4** (“harm to others ongoing”) and almost no **3.6** (none). Model collapses to majority class → **overconfident wrong S3** on benign text.

**Status:** **Partially mitigated; root cause still data/semantics.**

**Mitigations applied (engineering):**

- Documented distribution; training options: **S3 class-balanced sampler**, **macro-F1 checkpoint selection**, **lower S3 loss weight**, **label smoothing**, **focal loss**.  
- Inference: **temperature**, **risk uniform blend**, **handoff threshold** aligned with high-risk score.  
- Product: Admin **release** without forcing prior takeover when unassigned; clearer **English** system strings.

**Suggested figure:** **Pie or bar chart** of S3 class frequencies (from your stats doc); red callout on dominant slice.

---

### Slide 14 — Problem: training / deployment footguns (solved)

**Bullets:**

- **LoRA merge:** Must merge adapter + **correct backbone** + **three heads**; wrong `from_pretrained` caused random BERT → **fixed** in train end + `merge_peft_to_best_model.py`.  
- **Static UI vs CORS:** `file://` blocked; **served** `wechat_mock.html` / `admin.html` from same origin as API — **fixed**.  
- **Gradio vs package name:** thin `gradio.py` entry — **fixed**.  
- **Admin release 403:** Queue items had no `assigned_admin_id` — **fixed** release rule + UX feedback on buttons.

**Suggested figure:** “Before / after” two-panel diagram or simple bug-fix list with green checkmarks.

---

### Slide 15 — Problem: translation & LLM dependency

**Problem:**

- “Refresh EN” appeared dead when **LLM disabled** (empty translation).

**Solution:**

- Explicit **English fallback message** in API; UI **loading state**, **toast** with success/fail counts; Trace for debugging.

**Remaining:**

- No **offline** MT; quality depends on provider; **PII** in third-party APIs — note in thesis limitations.

**Suggested figure:** Screenshot of chat with EN subtitle + Admin “Enable external LLM” toggle.

---

### Slide 16 — Open issues & risks (for thesis “Limitations”)

**Bullets:**

- **Label validity:** Verify EFA `label.s3` semantics vs. export mapping; consider **rebalancing** or **augmenting 3.6** data.  
- **Calibration:** Temperature / blend are heuristics; proper **validation-set calibration** not fully done.  
- **Generalization:** Short social text vs. clinical interview — **domain gap**.  
- **Ethics:** False positives/negatives in crisis settings — system is **decision support**, not diagnosis.  
- **Evaluation:** Need **human** or **adversarial** test set for deployment claims.

**Suggested figure:** “Risk register” table (Issue | Severity | Mitigation | Future work).

---

## Part D — Next steps & thesis roadmap (Slides 17–20)

### Slide 17 — Near-term engineering (before draft)

**Bullets:**

- Finalize **one** canonical training config + frozen `best_model` + **eval plots** in paper.  
- Optional: **small human-labeled** validation slice for S3.  
- Clean **ablation** table (no balanced sampler vs with; no focal vs with).  
- Freeze **API screenshots** and **version hash** / commit id for reproducibility statement.

**Suggested figure:** Gantt chart (2–4 weeks) — Train → Eval → Freeze → Write.

---

### Slide 18 — Thesis chapter mapping

**Suggested chapter outline:**

1. **Introduction** — motivation, scope, contributions.  
2. **Related work** — mental-health NLP, multi-task learning, crisis detection, chatbots.  
3. **Data** — EFA, weak supervision, statistics, limitations.  
4. **Method** — architecture, losses, LoRA, risk & policy.  
5. **System** — backend, WS, admin workflow, LLM integration.  
6. **Experiments** — metrics, confusion matrices, case studies, ablations.  
7. **Discussion** — ethics, failure modes, future work.  
8. **Conclusion**

**Suggested figure:** Thesis TOC mock-up (one slide).

---

### Slide 19 — Contributions (draft thesis claims)

**Bullets (tune to your department’s style):**

- End-to-end **EFA-aligned multi-task** detector + **explicit policy layer** for handoff.  
- **Reproducible** training/eval toolchain with **imbalance-aware** options documented.  
- **Working prototype** integrating detection, human moderation, and optional LLM reply + EN assist.

**Suggested figure:** Three numbered contribution icons.

---

### Slide 20 — Summary & Q&A

**Bullets:**

- Built: **multi-task BERT+LoRA**, **risk/policy**, **full stack demo**, **admin/LLM/translation**.  
- Main challenge: **S3 data imbalance / semantics** — mitigated in code, **not fully solved** without better labels/data.  
- Next: **freeze experiments**, **write chapters**, **strengthen evaluation** for thesis.

**Suggested figure:** One strong **architecture + screenshot** composite; **“Thank you / Questions”**.

---

## Appendix (optional slides — backup for Q&A)

- **A1:** Full S1/S2/S3 label list (small font table).  
- **A2:** Hyperparameter table (LR, epochs, LoRA rank, batch size).  
- **A3:** Ethics & consent (if any user study planned).  
- **A4:** Repo map (`train_*`, `eval_*`, `backend_api`, `multitask_*`).

---

## Quick figure checklist (what to prepare)

| Slide area | Visual |
|------------|--------|
| Architecture | One system diagram (PNG/SVG) |
| Model | BERT + 3 heads schematic |
| Risk flow | Small flowchart |
| Data | S3 class distribution chart |
| Results | Confusion matrices / macro-F1 bars |
| Demo | WeChat mock + Admin screenshots |
| Timeline | 2–4 week Gantt |

---

*End of outline. Update Slide 12 numbers from your latest `metrics_report.json` before presenting.*
