from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from multitask_config import DEFAULT_INFERENCE_MODEL_DIR
from multitask_predict import predict_text
from chat_llm import build_system_user_lines, generate_reply, stream_generate_reply

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_REPO_ROOT, "chat_app.db")


def _html_file_response(filename: str) -> FileResponse:
    path = os.path.join(_REPO_ROOT, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Missing {filename} under repo root")
    return FileResponse(path, media_type="text/html; charset=utf-8")


def _recent_ai_snippets_for_prompt(hist: List[dict], max_turns: int = 3, max_chars: int = 220) -> List[str]:
    """Last few assistant turns, truncated, to steer the model away from copy-paste repetition."""
    snippets: List[str] = []
    for m in reversed(hist or []):
        if m.get("role") != "ai":
            continue
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if len(c) > max_chars:
            c = c[:max_chars] + "…"
        snippets.append(c)
        if len(snippets) >= max_turns:
            break
    return list(reversed(snippets))


def _hist_to_llm_messages(hist: List[dict], system_text: str) -> List[dict]:
    """Map DB roles to OpenAI-style user/assistant; prepend system if non-empty."""
    out: List[dict] = []
    if system_text.strip():
        out.append({"role": "system", "content": system_text.strip()})
    for m in hist:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append({"role": "user", "content": content})
        elif role == "ai":
            out.append({"role": "assistant", "content": content})
        elif role == "admin":
            out.append({"role": "assistant", "content": f"[Human moderator] {content}"})
    return out


def _default_llm_config_dict() -> dict:
    return {
        "api_base": "",
        "endpoint_path": "/v1/chat/completions",
        "model": "",
        "api_group": "",
        "api_key": "",
        "llm_enabled": 1,
        "updated_at": 0,
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 0,
        "nonstream_http_fallback": 1,
        "chat_llm_http_stream": 1,
    }


def _get_llm_config_row() -> dict:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM llm_config WHERE id=1").fetchone()
    conn.close()
    if not row:
        return _default_llm_config_dict()
    d = {k: row[k] for k in row.keys()}
    base = _default_llm_config_dict()
    base.update({k: d[k] for k in base if k in d})
    return base


def _clamp_llm_sampling(
    temperature: float, top_p: float, frequency_penalty: float, presence_penalty: float, max_tokens: int
) -> tuple:
    t = max(0.0, min(2.0, float(temperature)))
    tp = max(0.0, min(1.0, float(top_p)))
    fp = max(-2.0, min(2.0, float(frequency_penalty)))
    pp = max(-2.0, min(2.0, float(presence_penalty)))
    mt = max(0, min(128000, int(max_tokens)))
    return t, tp, fp, pp, mt


def _llm_sampling_from_cfg(cfg: dict) -> dict:
    t, tp, fp, pp, mt = _clamp_llm_sampling(
        float(cfg.get("temperature") if cfg.get("temperature") is not None else 0.7),
        float(cfg.get("top_p") if cfg.get("top_p") is not None else 1.0),
        float(cfg.get("frequency_penalty") if cfg.get("frequency_penalty") is not None else 0.0),
        float(cfg.get("presence_penalty") if cfg.get("presence_penalty") is not None else 0.0),
        int(cfg.get("max_tokens") or 0),
    )
    return {
        "temperature": t,
        "top_p": tp,
        "frequency_penalty": fp,
        "presence_penalty": pp,
        "max_tokens": mt if mt > 0 else None,
    }


def _prepare_llm_messages(action: str, pred: dict, hist: List[dict]) -> List[dict]:
    snippets = _recent_ai_snippets_for_prompt(hist)
    sys_prompt = build_system_user_lines(action, pred, recent_assistant_snippets=snippets or None)
    messages = _hist_to_llm_messages(hist, sys_prompt)
    if len(messages) < 2:
        messages = [
            {"role": "user", "content": sys_prompt + "\n\n用户说：\n" + (hist[-1].get("content") if hist else "")},
        ]
    return messages


async def _reply_with_llm_or_fallback(action: str, pred: dict, hist: List[dict], fallback: str) -> str:
    messages = _prepare_llm_messages(action, pred, hist)
    cfg = _get_llm_config_row()
    if not bool(cfg.get("llm_enabled", 1)):
        return fallback
    api_base = (cfg.get("api_base") or "").strip() or None
    endpoint_path = (cfg.get("endpoint_path") or "").strip() or "/v1/chat/completions"
    api_key = cfg.get("api_key") or ""
    model = (cfg.get("model") or "").strip() or None
    group = (cfg.get("api_group") or "").strip() or None
    samp = _llm_sampling_from_cfg(cfg)
    allow_ns_fb = bool(int(cfg.get("nonstream_http_fallback", 1)))
    http_stream_chat = bool(int(cfg.get("chat_llm_http_stream", 1)))
    try:
        text = await asyncio.to_thread(
            generate_reply,
            messages=messages,
            api_base=api_base,
            endpoint_path=endpoint_path,
            api_key=api_key,
            model=model,
            group=group,
            stream=http_stream_chat,
            allow_fallback_nonstream=allow_ns_fb if http_stream_chat else False,
            **samp,
        )
        if text and text.strip():
            return text.strip()
    except Exception as e:
        print(f"[chat_llm] {e}")
    return fallback


async def _translate_to_en(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    cfg = _get_llm_config_row()
    if not bool(cfg.get("llm_enabled", 1)):
        return (
            "[Translation unavailable: turn on 'Enable external LLM' in Admin, save, and configure API.]"
        )
    api_base = (cfg.get("api_base") or "").strip() or None
    endpoint_path = (cfg.get("endpoint_path") or "").strip() or "/v1/chat/completions"
    api_key = cfg.get("api_key") or ""
    model = (cfg.get("model") or "").strip() or None
    group = (cfg.get("api_group") or "").strip() or None
    prompt = (
        "Translate the following Chinese text into concise, natural English.\n"
        "Output ONLY the translation text, no explanations, no quotes.\n\n"
        f"Chinese:\n{raw}"
    )
    samp = {**_llm_sampling_from_cfg(cfg), "temperature": 0.2}
    try:
        out = await asyncio.to_thread(
            generate_reply,
            messages=[{"role": "user", "content": prompt}],
            api_base=api_base,
            endpoint_path=endpoint_path,
            api_key=api_key,
            model=model,
            group=group,
            stream=False,
            **samp,
        )
        return (out or "").strip()
    except Exception as e:
        print(f"[translate_en] {e}")
        return ""


class ChatSendReq(BaseModel):
    user_id: str
    text: str
    model_dir: str = DEFAULT_INFERENCE_MODEL_DIR
    device: str = "auto"
    stream: bool = False


class AdminTakeoverReq(BaseModel):
    conversation_id: str
    admin_id: str


class AdminReplyReq(BaseModel):
    conversation_id: str
    admin_id: str
    text: str


class AdminReleaseReq(BaseModel):
    conversation_id: str
    admin_id: str
    manual_risk: str = "Low risk"


class LlmConfigUpdate(BaseModel):
    api_base: str = ""
    endpoint_path: str = "/v1/chat/completions"
    model: str = ""
    api_group: str = ""
    api_key: Optional[str] = None
    llm_enabled: Optional[bool] = None
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 0
    nonstream_http_fallback: bool = True
    chat_llm_http_stream: bool = True


class LlmPresetCreate(BaseModel):
    name: str
    api_base: str = ""
    endpoint_path: str = "/v1/chat/completions"
    model: str = ""
    api_group: str = ""
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 0


class TranslateReq(BaseModel):
    text: str


class UserEventReq(BaseModel):
    user_id: str
    event_type: str  # moments | status
    content: str
    model_dir: str = DEFAULT_INFERENCE_MODEL_DIR
    device: str = "auto"


class ChatClearReq(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None


class AdminDeleteMessagesReq(BaseModel):
    conversation_id: str
    message_ids: List[int]


class AdminClearConversationMessagesReq(BaseModel):
    conversation_id: str


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="FYP Chat Backend", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_wechat_ui():
    """浏览器直接打开 http://127.0.0.1:8010/ 即可用微信 mock（同源，避免 file:// 被拒）。"""
    return _html_file_response("wechat_mock.html")


@app.get("/wechat_mock.html")
async def serve_wechat_ui_named():
    return _html_file_response("wechat_mock.html")


@app.get("/admin.html")
async def serve_admin_ui():
    return _html_file_response("admin.html")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL,
            assigned_admin_id TEXT,
            admin_lock INTEGER NOT NULL DEFAULT 0,
            manual_risk TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conv_cols = [r["name"] for r in cur.execute("PRAGMA table_info(conversations)").fetchall()]
    if "admin_lock" not in conv_cols:
        cur.execute("ALTER TABLE conversations ADD COLUMN admin_lock INTEGER NOT NULL DEFAULT 0")
    if "manual_risk" not in conv_cols:
        cur.execute("ALTER TABLE conversations ADD COLUMN manual_risk TEXT")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts INTEGER NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv_ts ON messages(conversation_id, ts)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            input_text TEXT NOT NULL,
            prediction_json TEXT NOT NULL,
            action TEXT NOT NULL,
            reply_text TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_infer_conv ON inference_logs(conversation_id, created_at)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            content TEXT NOT NULL,
            conversation_id TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_user_ts ON user_events(user_id, created_at)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            api_base TEXT NOT NULL DEFAULT '',
            endpoint_path TEXT NOT NULL DEFAULT '/v1/chat/completions',
            model TEXT NOT NULL DEFAULT '',
            api_group TEXT NOT NULL DEFAULT '',
            api_key TEXT NOT NULL DEFAULT '',
            llm_enabled INTEGER NOT NULL DEFAULT 1,
            updated_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cols = [r["name"] for r in cur.execute("PRAGMA table_info(llm_config)").fetchall()]
    if "endpoint_path" not in cols:
        cur.execute("ALTER TABLE llm_config ADD COLUMN endpoint_path TEXT NOT NULL DEFAULT '/v1/chat/completions'")
    if "llm_enabled" not in cols:
        cur.execute("ALTER TABLE llm_config ADD COLUMN llm_enabled INTEGER NOT NULL DEFAULT 1")
    if cur.execute("SELECT 1 FROM llm_config WHERE id=1").fetchone() is None:
        cur.execute(
            """
            INSERT INTO llm_config (id, api_base, endpoint_path, model, api_group, api_key, llm_enabled, updated_at)
            VALUES (1, '', '/v1/chat/completions', '', '', '', 1, 0)
            """
        )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_presets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            api_base TEXT NOT NULL DEFAULT '',
            endpoint_path TEXT NOT NULL DEFAULT '/v1/chat/completions',
            model TEXT NOT NULL DEFAULT '',
            api_group TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL
        )
        """
    )
    _llm_sampling_migrate_columns(cur, "llm_config")
    _llm_sampling_migrate_columns(cur, "llm_presets")
    cols_cfg = {r["name"] for r in cur.execute("PRAGMA table_info(llm_config)").fetchall()}
    if "nonstream_http_fallback" not in cols_cfg:
        cur.execute(
            "ALTER TABLE llm_config ADD COLUMN nonstream_http_fallback INTEGER NOT NULL DEFAULT 1"
        )
    if "chat_llm_http_stream" not in cols_cfg:
        cur.execute(
            "ALTER TABLE llm_config ADD COLUMN chat_llm_http_stream INTEGER NOT NULL DEFAULT 1"
        )
    conn.commit()
    conn.close()


def _llm_sampling_migrate_columns(cur: sqlite3.Cursor, table: str) -> None:
    cols = {r["name"] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
    specs = [
        ("temperature", "REAL NOT NULL DEFAULT 0.7"),
        ("top_p", "REAL NOT NULL DEFAULT 1.0"),
        ("frequency_penalty", "REAL NOT NULL DEFAULT 0.0"),
        ("presence_penalty", "REAL NOT NULL DEFAULT 0.0"),
        ("max_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ]
    for name, decl in specs:
        if name not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


class WSManager:
    def __init__(self):
        self.user_sockets: Dict[str, set[WebSocket]] = defaultdict(set)
        self.admin_sockets: Dict[str, set[WebSocket]] = defaultdict(set)
        self.conv_sockets: Dict[str, set[WebSocket]] = defaultdict(set)

    async def connect_user(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.user_sockets[user_id].add(websocket)

    async def connect_admin(self, admin_id: str, websocket: WebSocket):
        await websocket.accept()
        self.admin_sockets[admin_id].add(websocket)

    def join_conversation(self, conversation_id: str, websocket: WebSocket):
        self.conv_sockets[conversation_id].add(websocket)

    def disconnect(self, websocket: WebSocket):
        for bucket in (self.user_sockets, self.admin_sockets, self.conv_sockets):
            for key in list(bucket.keys()):
                if websocket in bucket[key]:
                    bucket[key].discard(websocket)
                if not bucket[key]:
                    del bucket[key]

    async def _safe_send(self, websocket: WebSocket, payload: dict):
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self.disconnect(websocket)

    async def broadcast_conversation(self, conversation_id: str, payload: dict):
        sockets = list(self.conv_sockets.get(conversation_id, set()))
        if not sockets:
            return
        await asyncio.gather(*(self._safe_send(ws, payload) for ws in sockets), return_exceptions=True)

    async def broadcast_admins(self, payload: dict):
        all_admin_ws: List[WebSocket] = []
        for items in self.admin_sockets.values():
            all_admin_ws.extend(list(items))
        if not all_admin_ws:
            return
        await asyncio.gather(*(self._safe_send(ws, payload) for ws in all_admin_ws), return_exceptions=True)


ws_manager = WSManager()


def _fetch_messages(conn: sqlite3.Connection, conversation_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT id, role, content, ts FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    return [{"id": r["id"], "role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]


def _conversation_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "conversation_id": row["conversation_id"],
        "user_id": row["user_id"],
        "status": row["status"],
        "assigned_admin_id": row["assigned_admin_id"],
        "admin_lock": row["admin_lock"] if "admin_lock" in row.keys() else 0,
        "manual_risk": row["manual_risk"] if "manual_risk" in row.keys() else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _get_or_create_conv(conn: sqlite3.Connection, user_id: str) -> dict:
    row = conn.execute(
        "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    if row:
        return _conversation_row_to_dict(row)
    now = int(time.time())
    conversation_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO conversations (conversation_id, user_id, status, assigned_admin_id, created_at, updated_at)
        VALUES (?, ?, 'ai', NULL, ?, ?)
        """,
        (conversation_id, user_id, now, now),
    )
    conn.commit()
    return {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "status": "ai",
        "assigned_admin_id": None,
        "created_at": now,
        "updated_at": now,
    }


def _get_conversation(conn: sqlite3.Connection, conversation_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
    if not row:
        return None
    return _conversation_row_to_dict(row)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"service": "FYP Chat Backend", "ok": True, "health": "/health"}


@app.get("/favicon.ico")
def favicon():
    return {"ok": True}


@app.get("/monitor/latest")
def monitor_latest(limit: int = 20):
    limit = max(1, min(limit, 100))
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, conversation_id, user_id, input_text, prediction_json, action, reply_text, created_at
        FROM inference_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    items: List[dict] = []
    for r in rows:
        try:
            pred = json.loads(r["prediction_json"])
        except Exception:
            pred = {"raw": r["prediction_json"]}
        items.append(
            {
                "id": r["id"],
                "conversation_id": r["conversation_id"],
                "user_id": r["user_id"],
                "input_text": r["input_text"],
                "prediction": pred,
                "action": r["action"],
                "reply": r["reply_text"],
                "created_at": r["created_at"],
            }
        )
    return {"items": items}


def _schedule_ws_broadcast(conv_id: str, payload: dict, admin_ping: bool):
    async def _go():
        await ws_manager.broadcast_conversation(conv_id, payload)
        if admin_ping:
            await ws_manager.broadcast_admins({"type": "queue_changed"})

    def _runner():
        asyncio.run(_go())

    threading.Thread(target=_runner, daemon=True).start()


def _apply_keyword_risk_guard(text: str, pred: dict) -> tuple[dict, str]:
    """
    Hard safety guard: explicit high-risk keywords force handoff.
    This prevents obvious misses from model predictions in demo usage.
    """
    raw = (text or "").strip()
    compact = "".join(raw.split())
    keywords = ["自杀", "轻生", "不想活", "活不下去", "一了百了", "结束生命", "想死", "去死", "自残", "割腕"]
    hit = any(k in raw or k in compact for k in keywords)
    action = pred.get("action", "ai_reply")
    if hit:
        pred["风险等级"] = "高风险"
        pred["action"] = "handoff_to_human"
        resources = pred.get("resources")
        if not isinstance(resources, list):
            resources = []
        if "hotline" not in resources:
            resources.append("hotline")
        pred["resources"] = resources
        action = "handoff_to_human"
    return pred, action


def _needs_hotline(action: str, pred: dict) -> bool:
    a = (action or "").strip().lower()
    if a == "handoff_to_human" or a == "ai_reply+hotline":
        return True
    resources = pred.get("resources")
    if isinstance(resources, list):
        return any(str(x).strip().lower() == "hotline" for x in resources)
    return False


def _hotline_block() -> str:
    # Keep this concise in chat UI: platform + resource.
    return (
        "Emergency mental-health resources (24/7):\n"
        "1) National Psychological Aid Hotline: 010-82951332 / 010-82951331\n"
        "2) Beijing Psychological Crisis Intervention Center: http://www.bjpsych.cn/\n"
        "3) SOS Psychological Aid Platform: http://www.psychaide.com.cn/\n"
        "4) In immediate danger, call 120 or 110, or go to the nearest hospital emergency/psychiatry department."
    )


def _build_hotline_system_tip(action: str, pred: dict) -> str:
    if not _needs_hotline(action, pred):
        return ""
    return _hotline_block()


def _handoff_system_tip() -> str:
    return "High-risk signal detected. This conversation has been handed off to a human moderator now."


def _clean_assistant_reply_for_user(text: str) -> str:
    """Strip markdown / pseudo-headings so the UI shows one natural chat line."""
    s = (text or "").strip()
    if not s:
        return s
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", s)
    s = re.sub(r"(?m)^#{1,6}\s*", "", s)
    lines_out: List[str] = []
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(
            r"^(回答你的问题|生成文本|翻译语言|以下是|总结|功能包括|我的能力)[：:\s]+",
            "",
            line,
            flags=re.I,
        )
        line = re.sub(r"^[\-\*•]+\s+", "", line)
        line = re.sub(r"^\d+[.)）、]\s*", "", line)
        if line:
            lines_out.append(line)
    s = " ".join(lines_out)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _split_reply_into_chat_bubbles(text: str) -> List[str]:
    """Split one assistant turn into several DB/UI bubbles (sentence-like), WeChat-style."""
    s = (text or "").strip()
    if not s:
        return []
    parts: List[str] = []
    cur: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        if i + 1 < n and s[i : i + 2] == "……":
            cur.append("……")
            i += 2
            seg = "".join(cur).strip()
            if seg:
                parts.append(seg)
            cur = []
            continue
        ch = s[i]
        cur.append(ch)
        i += 1
        if ch in "。！？":
            seg = "".join(cur).strip()
            if seg:
                parts.append(seg)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts if parts else [s]


def _insert_ai_reply_lines(conn: sqlite3.Connection, conversation_id: str, reply: str, base_ts: int) -> List[dict]:
    cleaned = _clean_assistant_reply_for_user(reply)
    if not cleaned:
        cleaned = "（空回复）"
    bubbles = _split_reply_into_chat_bubbles(cleaned)
    if not bubbles:
        bubbles = [cleaned]
    out: List[dict] = []
    for j, content in enumerate(bubbles):
        ts = base_ts + j
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'ai', ?, ?)",
            (conversation_id, content, ts),
        )
        out.append({"role": "ai", "content": content, "ts": ts})
    return out


@app.post("/chat/send")
async def chat_send(req: ChatSendReq):
    if req.stream:
        return _chat_send_stream_response(req)

    conn = _get_conn()
    conv = _get_or_create_conv(conn, req.user_id)
    # Respect admin takeover lock: do not switch back to AI until admin releases.
    if int(conv.get("admin_lock") or 0) == 1:
        now = int(time.time())
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'user', ?, ?)",
            (conv["conversation_id"], req.text, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
            (now, conv["conversation_id"]),
        )
        conn.commit()
        conn.close()
        await ws_manager.broadcast_conversation(
            conv["conversation_id"],
            {
                "type": "conversation_update",
                "conversation_id": conv["conversation_id"],
                "messages": [{"role": "user", "content": req.text, "ts": now}],
                "status": "admin",
            },
        )
        await ws_manager.broadcast_admins({"type": "queue_changed"})
        return {
            "conversation_id": conv["conversation_id"],
            "status": "admin",
            "prediction": {},
            "reply": "",
            "hotline_tip": "",
            "admin_lock": True,
        }
    now = int(time.time())

    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'user', ?, ?)",
        (conv["conversation_id"], req.text, now),
    )

    pred = predict_text(
        text=req.text,
        model_dir=req.model_dir,
        device=req.device,
        max_length=128,
        s1_threshold=0.15,
    )
    pred, action = _apply_keyword_risk_guard(req.text, pred)

    if action == "handoff_to_human":
        status = "waiting_admin"
        fallback_reply = ""
    else:
        status = "ai"
        risk_level = pred.get("风险等级", "unknown")
        state = pred.get("心理状态", "unknown")
        fallback_reply = f"I'm here with you. Current risk={risk_level}, state={state}. Would you like to share a bit more?"

    reply = ""
    if action != "handoff_to_human":
        hist = _fetch_messages(conn, conv["conversation_id"])
        reply = await _reply_with_llm_or_fallback(action, pred, hist, fallback_reply)
    hotline_tip = _build_hotline_system_tip(action, pred)
    handoff_tip = _handoff_system_tip() if action == "handoff_to_human" else ""

    ai_msgs: List[dict] = []
    if reply:
        ai_msgs = _insert_ai_reply_lines(conn, conv["conversation_id"], reply, now)
    system_cursor = now + len(ai_msgs)
    if handoff_tip:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'system', ?, ?)",
            (conv["conversation_id"], handoff_tip, system_cursor),
        )
        system_cursor += 1
    if hotline_tip:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'system', ?, ?)",
            (conv["conversation_id"], hotline_tip, system_cursor),
        )
    conn.execute(
        "UPDATE conversations SET status=?, updated_at=? WHERE conversation_id=?",
        (status, now, conv["conversation_id"]),
    )
    conn.execute(
        """
        INSERT INTO inference_logs (
            conversation_id, user_id, input_text, prediction_json, action, reply_text, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conv["conversation_id"],
            req.user_id,
            req.text,
            json.dumps(pred, ensure_ascii=False),
            action,
            reply,
            now,
        ),
    )
    conn.commit()
    conn.close()

    await ws_manager.broadcast_conversation(
        conv["conversation_id"],
        {
            "type": "conversation_update",
            "conversation_id": conv["conversation_id"],
            "messages": [
                {"role": "user", "content": req.text, "ts": now},
                *ai_msgs,
                *([{"role": "system", "content": handoff_tip, "ts": now + len(ai_msgs)}] if handoff_tip else []),
                *(
                    [
                        {
                            "role": "system",
                            "content": hotline_tip,
                            "ts": now + len(ai_msgs) + (1 if handoff_tip else 0),
                        }
                    ]
                    if hotline_tip
                    else []
                ),
            ],
            "status": status,
        },
    )
    await ws_manager.broadcast_admins({"type": "queue_changed"})

    return {
        "conversation_id": conv["conversation_id"],
        "status": status,
        "prediction": pred,
        "reply": reply,
        "hotline_tip": hotline_tip,
    }


def _chat_send_stream_response(req: ChatSendReq):
    conn = _get_conn()
    try:
        conv = _get_or_create_conv(conn, req.user_id)
        now = int(time.time())

        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'user', ?, ?)",
            (conv["conversation_id"], req.text, now),
        )

        # If admin takeover is active, do not call LLM; keep conversation under admin control.
        if int(conv.get("admin_lock") or 0) == 1:
            conn.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (now, conv["conversation_id"]))
            conn.commit()
            conn.close()

            def event_gen_admin():
                meta = {"type": "meta", "conversation_id": conv["conversation_id"], "status": "admin", "prediction": {}}
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
                tip = "This conversation is currently handled by a human moderator. You can keep sending messages."
                yield f"data: {json.dumps({'type': 'system', 'content': tip}, ensure_ascii=False)}\n\n"
                done = {"type": "done", "conversation_id": conv["conversation_id"], "reply": "", "status": "admin"}
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
                _schedule_ws_broadcast(
                    conv["conversation_id"],
                    {
                        "type": "conversation_update",
                        "conversation_id": conv["conversation_id"],
                        "messages": [{"role": "user", "content": req.text, "ts": now}],
                        "status": "admin",
                    },
                    admin_ping=True,
                )

            return StreamingResponse(
                event_gen_admin(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        pred = predict_text(
            text=req.text,
            model_dir=req.model_dir,
            device=req.device,
            max_length=128,
            s1_threshold=0.15,
        )
        pred, action = _apply_keyword_risk_guard(req.text, pred)

        if action == "handoff_to_human":
            status = "waiting_admin"
            fallback_reply = ""
        else:
            status = "ai"
            risk_level = pred.get("风险等级", "unknown")
            state = pred.get("心理状态", "unknown")
            fallback_reply = f"I'm here with you. Current risk={risk_level}, state={state}. Would you like to share a bit more?"

        hist = _fetch_messages(conn, conv["conversation_id"])
        messages = _prepare_llm_messages(action, pred, hist)
        cfg = _get_llm_config_row()
        llm_enabled = bool(cfg.get("llm_enabled", 1))
        api_base = (cfg.get("api_base") or "").strip() or None
        endpoint_path = (cfg.get("endpoint_path") or "").strip() or "/v1/chat/completions"
        api_key = cfg.get("api_key") or ""
        model = (cfg.get("model") or "").strip() or None
        group = (cfg.get("api_group") or "").strip() or None
        llm_samp = _llm_sampling_from_cfg(cfg)
        allow_ns_fb = bool(int(cfg.get("nonstream_http_fallback", 1)))
        http_stream_chat = bool(int(cfg.get("chat_llm_http_stream", 1)))
        conn.commit()
        conn.close()

        def event_gen():
            reply_parts: List[str] = []
            try:
                meta = {
                    "type": "meta",
                    "conversation_id": conv["conversation_id"],
                    "status": status,
                    "prediction": pred,
                }
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
                try:
                    if action == "handoff_to_human":
                        pass
                    elif not llm_enabled:
                        reply_parts = [fallback_reply]
                        yield f"data: {json.dumps({'type': 'chunk', 'text': fallback_reply}, ensure_ascii=False)}\n\n"
                    else:
                        if http_stream_chat:
                            for delta in stream_generate_reply(
                                messages=messages,
                                api_base=api_base,
                                endpoint_path=endpoint_path,
                                api_key=api_key,
                                model=model,
                                group=group,
                                allow_fallback_nonstream=allow_ns_fb,
                                **llm_samp,
                            ):
                                reply_parts.append(delta)
                                yield f"data: {json.dumps({'type': 'chunk', 'text': delta}, ensure_ascii=False)}\n\n"
                        else:
                            full = generate_reply(
                                messages=messages,
                                api_base=api_base,
                                endpoint_path=endpoint_path,
                                api_key=api_key,
                                model=model,
                                group=group,
                                stream=False,
                                **llm_samp,
                            )
                            blob = (full or "").strip()
                            if blob:
                                reply_parts.append(blob)
                                yield f"data: {json.dumps({'type': 'chunk', 'text': blob}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"[chat_llm stream] {e}")
                    reply_parts = [fallback_reply]
                    yield f"data: {json.dumps({'type': 'chunk', 'text': fallback_reply}, ensure_ascii=False)}\n\n"
                reply_raw = "".join(reply_parts).strip()
                if action != "handoff_to_human" and not reply_raw:
                    reply_raw = fallback_reply
                hotline_tip = _build_hotline_system_tip(action, pred)
                handoff_tip = _handoff_system_tip() if action == "handoff_to_human" else ""
                if action != "handoff_to_human" and not reply_parts:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': reply_raw}, ensure_ascii=False)}\n\n"
                if handoff_tip:
                    yield f"data: {json.dumps({'type': 'system', 'content': handoff_tip}, ensure_ascii=False)}\n\n"
                if hotline_tip:
                    yield f"data: {json.dumps({'type': 'system', 'content': hotline_tip}, ensure_ascii=False)}\n\n"

                write_conn = _get_conn()
                try:
                    ai_msgs: List[dict] = []
                    if reply_raw:
                        ai_msgs = _insert_ai_reply_lines(write_conn, conv["conversation_id"], reply_raw, now)
                    system_cursor = now + len(ai_msgs)
                    if handoff_tip:
                        write_conn.execute(
                            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'system', ?, ?)",
                            (conv["conversation_id"], handoff_tip, system_cursor),
                        )
                        system_cursor += 1
                    if hotline_tip:
                        write_conn.execute(
                            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'system', ?, ?)",
                            (conv["conversation_id"], hotline_tip, system_cursor),
                        )
                    write_conn.execute(
                        "UPDATE conversations SET status=?, updated_at=? WHERE conversation_id=?",
                        (status, now, conv["conversation_id"]),
                    )
                    write_conn.execute(
                        """
                        INSERT INTO inference_logs (
                            conversation_id, user_id, input_text, prediction_json, action, reply_text, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            conv["conversation_id"],
                            req.user_id,
                            req.text,
                            json.dumps(pred, ensure_ascii=False),
                            action,
                            ("\n".join(m["content"] for m in ai_msgs) if ai_msgs else reply_raw),
                            now,
                        ),
                    )
                    write_conn.commit()
                finally:
                    write_conn.close()

                reply_parts = [m["content"] for m in ai_msgs] if ai_msgs else []
                stored_reply = reply_parts[0] if len(reply_parts) == 1 else "\n".join(reply_parts)
                done = {
                    "type": "done",
                    "conversation_id": conv["conversation_id"],
                    "reply": stored_reply,
                    "reply_parts": reply_parts,
                    "status": status,
                }
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
                ws_ai_msgs = ai_msgs

                _schedule_ws_broadcast(
                    conv["conversation_id"],
                    {
                        "type": "conversation_update",
                        "conversation_id": conv["conversation_id"],
                        "messages": [
                            {"role": "user", "content": req.text, "ts": now},
                            *ws_ai_msgs,
                            *(
                                [{"role": "system", "content": handoff_tip, "ts": now + len(ws_ai_msgs)}]
                                if handoff_tip
                                else []
                            ),
                            *(
                                [
                                    {
                                        "role": "system",
                                        "content": hotline_tip,
                                        "ts": now + len(ws_ai_msgs) + (1 if handoff_tip else 0),
                                    }
                                ]
                                if hotline_tip
                                else []
                            ),
                        ],
                        "status": status,
                    },
                    admin_ping=True,
                )
            finally:
                pass

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise


@app.get("/chat/history/{conversation_id}")
def chat_history(conversation_id: str):
    conn = _get_conn()
    conv = _get_conversation(conn, conversation_id)
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="conversation not found")
    messages = _fetch_messages(conn, conversation_id)
    conn.close()
    conv["messages"] = messages
    return conv


@app.get("/chat/latest/{user_id}")
def chat_latest(user_id: str):
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 1",
        (uid,),
    ).fetchone()
    if not row:
        conn.close()
        return JSONResponse(
            content={"found": False},
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )
    conv = _conversation_row_to_dict(row)
    conv["messages"] = _fetch_messages(conn, conv["conversation_id"])
    conn.close()
    conv["found"] = True
    return JSONResponse(
        content=conv,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.post("/chat/clear")
async def chat_clear(req: ChatClearReq):
    uid = (req.user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")
    conn = _get_conn()
    try:
        cid_opt = (req.conversation_id or "").strip()
        if cid_opt:
            row = conn.execute(
                "SELECT conversation_id, user_id FROM conversations WHERE conversation_id=?",
                (cid_opt,),
            ).fetchone()
            if not row or row["user_id"] != uid:
                raise HTTPException(status_code=404, detail="conversation not found")
            cid = row["conversation_id"]
        else:
            row = conn.execute(
                "SELECT conversation_id FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 1",
                (uid,),
            ).fetchone()
            if not row:
                return {"ok": True, "conversation_id": None, "deleted_count": 0}
            cid = row["conversation_id"]

        cur = conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        deleted = int(cur.rowcount or 0)
        now = int(time.time())
        conn.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (now, cid))
        conn.commit()
    finally:
        conn.close()

    await ws_manager.broadcast_conversation(
        cid,
        {"type": "conversation_cleared", "conversation_id": cid},
    )
    await ws_manager.broadcast_admins({"type": "conversation_changed", "conversation_id": cid})
    return {"ok": True, "conversation_id": cid, "deleted_count": deleted}


@app.post("/chat/translate")
async def chat_translate(req: TranslateReq):
    en = await _translate_to_en(req.text)
    return {"en": en}


@app.post("/chat/user-event")
async def chat_user_event(req: UserEventReq):
    event_type = (req.event_type or "").strip().lower()
    if event_type not in {"moments", "status"}:
        raise HTTPException(status_code=400, detail="event_type must be 'moments' or 'status'")
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is empty")

    event_text = f"[{event_type}] {content}"
    now = int(time.time())
    conn = _get_conn()
    conv = _get_or_create_conv(conn, req.user_id)
    conn.execute(
        """
        INSERT INTO user_events (user_id, event_type, content, conversation_id, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (req.user_id, event_type, content, conv["conversation_id"], now),
    )
    conn.commit()
    conn.close()

    out = await chat_send(
        ChatSendReq(
            user_id=req.user_id,
            text=event_text,
            model_dir=req.model_dir,
            device=req.device,
            stream=False,
        )
    )
    out["event_type"] = event_type
    out["event_content"] = content
    return out


@app.get("/admin/llm-config")
def admin_llm_config_get():
    row = _get_llm_config_row()
    key = (row.get("api_key") or "").strip()
    t, tp, fp, pp, mt = _clamp_llm_sampling(
        float(row.get("temperature") if row.get("temperature") is not None else 0.7),
        float(row.get("top_p") if row.get("top_p") is not None else 1.0),
        float(row.get("frequency_penalty") if row.get("frequency_penalty") is not None else 0.0),
        float(row.get("presence_penalty") if row.get("presence_penalty") is not None else 0.0),
        int(row.get("max_tokens") or 0),
    )
    return {
        "api_base": row.get("api_base") or "",
        "endpoint_path": row.get("endpoint_path") or "/v1/chat/completions",
        "model": row.get("model") or "",
        "api_group": row.get("api_group") or "",
        "llm_enabled": bool(row.get("llm_enabled", 1)),
        "has_api_key": bool(key),
        "updated_at": row.get("updated_at") or 0,
        "temperature": t,
        "top_p": tp,
        "frequency_penalty": fp,
        "presence_penalty": pp,
        "max_tokens": mt,
        "nonstream_http_fallback": bool(int(row.get("nonstream_http_fallback", 1))),
        "chat_llm_http_stream": bool(int(row.get("chat_llm_http_stream", 1))),
    }


@app.get("/admin/llm-presets")
def admin_llm_presets_list():
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, name, api_base, endpoint_path, model, api_group, created_at,
               temperature, top_p, frequency_penalty, presence_penalty, max_tokens
        FROM llm_presets
        ORDER BY name COLLATE NOCASE ASC
        """
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}


@app.post("/admin/llm-presets")
def admin_llm_preset_create(body: LlmPresetCreate):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    ep = (body.endpoint_path or "").strip() or "/v1/chat/completions"
    if not ep.startswith("/"):
        ep = "/" + ep
    t, tp, fp, pp, mt = _clamp_llm_sampling(
        body.temperature,
        body.top_p,
        body.frequency_penalty,
        body.presence_penalty,
        body.max_tokens,
    )
    now = int(time.time())
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO llm_presets (
                name, api_base, endpoint_path, model, api_group, created_at,
                temperature, top_p, frequency_penalty, presence_penalty, max_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                (body.api_base or "").strip(),
                ep,
                (body.model or "").strip(),
                (body.api_group or "").strip(),
                now,
                t,
                tp,
                fp,
                pp,
                mt,
            ),
        )
        conn.commit()
        pid = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="preset name already exists")
    row = conn.execute(
        """
        SELECT id, name, api_base, endpoint_path, model, api_group, created_at,
               temperature, top_p, frequency_penalty, presence_penalty, max_tokens
        FROM llm_presets WHERE id=?
        """,
        (pid,),
    ).fetchone()
    conn.close()
    return {"ok": True, "preset": dict(row) if row else {"id": pid}}


@app.post("/admin/llm-presets/{preset_id}/apply")
def admin_llm_preset_apply(preset_id: int):
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT api_base, endpoint_path, model, api_group,
               temperature, top_p, frequency_penalty, presence_penalty, max_tokens
        FROM llm_presets WHERE id=?
        """,
        (preset_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="preset not found")
    now = int(time.time())
    conn.execute(
        """
        UPDATE llm_config
        SET api_base=?, endpoint_path=?, model=?, api_group=?,
            temperature=?, top_p=?, frequency_penalty=?, presence_penalty=?, max_tokens=?,
            updated_at=?
        WHERE id=1
        """,
        (
            row["api_base"],
            row["endpoint_path"],
            row["model"],
            row["api_group"],
            row["temperature"],
            row["top_p"],
            row["frequency_penalty"],
            row["presence_penalty"],
            row["max_tokens"],
            now,
        ),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.delete("/admin/llm-presets/{preset_id}")
def admin_llm_preset_delete(preset_id: int):
    conn = _get_conn()
    cur = conn.execute("DELETE FROM llm_presets WHERE id=?", (preset_id,))
    conn.commit()
    conn.close()
    if not cur.rowcount:
        raise HTTPException(status_code=404, detail="preset not found")
    return {"ok": True}


@app.post("/admin/llm-config")
def admin_llm_config_post(body: LlmConfigUpdate):
    conn = _get_conn()
    cur_row = conn.execute("SELECT api_key, llm_enabled FROM llm_config WHERE id=1").fetchone()
    existing_key = (cur_row["api_key"] if cur_row else "") or ""
    existing_enabled = int(cur_row["llm_enabled"]) if cur_row else 1
    new_key = existing_key if body.api_key is None else body.api_key
    new_enabled = existing_enabled if body.llm_enabled is None else (1 if body.llm_enabled else 0)
    now = int(time.time())
    api_base = (body.api_base or "").strip()
    endpoint_path = (body.endpoint_path or "").strip() or "/v1/chat/completions"
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path
    model = (body.model or "").strip()
    group = (body.api_group or "").strip()
    t, tp, fp, pp, mt = _clamp_llm_sampling(
        body.temperature,
        body.top_p,
        body.frequency_penalty,
        body.presence_penalty,
        body.max_tokens,
    )
    ns_fb = 1 if body.nonstream_http_fallback else 0
    chat_stream = 1 if body.chat_llm_http_stream else 0
    cur = conn.execute(
        """
        UPDATE llm_config
        SET api_base=?, endpoint_path=?, model=?, api_group=?, api_key=?, llm_enabled=?,
            temperature=?, top_p=?, frequency_penalty=?, presence_penalty=?, max_tokens=?,
            nonstream_http_fallback=?, chat_llm_http_stream=?, updated_at=?
        WHERE id=1
        """,
        (api_base, endpoint_path, model, group, new_key, new_enabled, t, tp, fp, pp, mt, ns_fb, chat_stream, now),
    )
    if cur.rowcount == 0:
        conn.execute(
            """
            INSERT INTO llm_config (
                id, api_base, endpoint_path, model, api_group, api_key, llm_enabled,
                temperature, top_p, frequency_penalty, presence_penalty, max_tokens,
                nonstream_http_fallback, chat_llm_http_stream, updated_at
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (api_base, endpoint_path, model, group, new_key, new_enabled, t, tp, fp, pp, mt, ns_fb, chat_stream, now),
        )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/admin/llm-ping")
async def admin_llm_ping():
    cfg = _get_llm_config_row()
    if not bool(cfg.get("llm_enabled", 1)):
        return {"ok": False, "reason": "llm_disabled", "message": "External LLM is disabled. Enable it first."}

    api_base = (cfg.get("api_base") or "").strip() or None
    endpoint_path = (cfg.get("endpoint_path") or "").strip() or "/v1/chat/completions"
    api_key = cfg.get("api_key") or ""
    model = (cfg.get("model") or "").strip() or None
    group = (cfg.get("api_group") or "").strip() or None
    samp = _llm_sampling_from_cfg(cfg)

    messages = [
        {"role": "user", "content": "请回复“pong”即可。"},
    ]
    started = time.time()
    try:
        text = await asyncio.to_thread(
            generate_reply,
            messages=messages,
            api_base=api_base,
            endpoint_path=endpoint_path,
            api_key=api_key,
            model=model,
            group=group,
            stream=False,
            **{**samp, "temperature": min(float(samp["temperature"]), 0.3)},
        )
        elapsed_ms = int((time.time() - started) * 1000)
        if text and text.strip():
            return {"ok": True, "elapsed_ms": elapsed_ms, "reply_preview": text[:120]}
        return {"ok": False, "elapsed_ms": elapsed_ms, "reason": "empty_reply", "message": "LLM returned empty content."}
    except Exception as e:
        elapsed_ms = int((time.time() - started) * 1000)
        return {"ok": False, "elapsed_ms": elapsed_ms, "reason": "request_failed", "message": str(e)[:500]}


@app.get("/admin/queue")
def admin_queue():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT conversation_id, user_id, updated_at FROM conversations WHERE status='waiting_admin' ORDER BY updated_at DESC"
    ).fetchall()
    items: List[dict] = []
    for row in rows:
        msg_row = conn.execute(
            "SELECT content FROM messages WHERE conversation_id=? AND role='user' ORDER BY id DESC LIMIT 1",
            (row["conversation_id"],),
        ).fetchone()
        items.append(
            {
                "conversation_id": row["conversation_id"],
                "user_id": row["user_id"],
                "last_user_message": msg_row["content"] if msg_row else "",
                "updated_at": row["updated_at"],
            }
        )
    conn.close()
    return {"items": items}


@app.post("/admin/takeover")
async def admin_takeover(req: AdminTakeoverReq):
    conn = _get_conn()
    conv = _get_conversation(conn, req.conversation_id)
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="conversation not found")
    now = int(time.time())
    conn.execute(
        "UPDATE conversations SET status='admin', assigned_admin_id=?, admin_lock=1, updated_at=? WHERE conversation_id=?",
        (req.admin_id, now, req.conversation_id),
    )
    conn.commit()
    conn.close()

    await ws_manager.broadcast_conversation(
        req.conversation_id,
        {
            "type": "conversation_status",
            "conversation_id": req.conversation_id,
            "status": "admin",
            "assigned_admin_id": req.admin_id,
        },
    )
    await ws_manager.broadcast_admins({"type": "queue_changed"})
    return {"ok": True, "conversation_id": req.conversation_id, "status": "admin"}


# Shown to the user when a moderator returns the chat to AI (English).
ADMIN_RELEASE_TO_AI_SYSTEM_MSG_EN = (
    "System: The moderator has released this chat. W (AI) is handling messages again."
)


@app.post("/admin/release")
async def admin_release(req: AdminReleaseReq):
    conn = _get_conn()
    conv = _get_conversation(conn, req.conversation_id)
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="conversation not found")
    assigned = conv.get("assigned_admin_id")
    if assigned and assigned != req.admin_id:
        conn.close()
        raise HTTPException(
            status_code=403,
            detail="This conversation is locked to another moderator ID.",
        )
    now = int(time.time())
    conn.execute(
        "UPDATE conversations SET status='ai', assigned_admin_id=NULL, admin_lock=0, manual_risk=?, updated_at=? WHERE conversation_id=?",
        (req.manual_risk, now, req.conversation_id),
    )
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'system', ?, ?)",
        (req.conversation_id, ADMIN_RELEASE_TO_AI_SYSTEM_MSG_EN, now),
    )
    conn.commit()
    conn.close()
    await ws_manager.broadcast_conversation(
        req.conversation_id,
        {
            "type": "conversation_update",
            "conversation_id": req.conversation_id,
            "messages": [{"role": "system", "content": ADMIN_RELEASE_TO_AI_SYSTEM_MSG_EN, "ts": now}],
            "status": "ai",
        },
    )
    await ws_manager.broadcast_conversation(
        req.conversation_id,
        {
            "type": "conversation_status",
            "conversation_id": req.conversation_id,
            "status": "ai",
            "assigned_admin_id": None,
        },
    )
    await ws_manager.broadcast_admins({"type": "queue_changed"})
    await ws_manager.broadcast_admins(
        {"type": "conversation_changed", "conversation_id": req.conversation_id}
    )
    return {"ok": True, "conversation_id": req.conversation_id, "status": "ai", "manual_risk": req.manual_risk}


@app.post("/admin/reply")
async def admin_reply(req: AdminReplyReq):
    conn = _get_conn()
    conv = _get_conversation(conn, req.conversation_id)
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="conversation not found")
    if conv.get("assigned_admin_id") != req.admin_id:
        conn.close()
        raise HTTPException(status_code=403, detail="conversation not assigned to this admin")
    now = int(time.time())
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'admin', ?, ?)",
        (req.conversation_id, req.text, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
        (now, req.conversation_id),
    )
    conn.commit()
    conn.close()

    await ws_manager.broadcast_conversation(
        req.conversation_id,
        {
            "type": "conversation_update",
            "conversation_id": req.conversation_id,
            "messages": [{"role": "admin", "content": req.text, "ts": now}],
            "status": "admin",
        },
    )
    return {"ok": True}


@app.get("/admin/conversation/{conversation_id}")
def admin_conversation(conversation_id: str):
    conn = _get_conn()
    conv = _get_conversation(conn, conversation_id)
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="conversation not found")
    conv["messages"] = _fetch_messages(conn, conversation_id)
    conn.close()
    return conv


@app.post("/admin/messages/delete")
async def admin_delete_messages(req: AdminDeleteMessagesReq):
    cid = (req.conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id required")
    try:
        ids = sorted({int(x) for x in req.message_ids})
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="message_ids must be integers")
    if not ids:
        raise HTTPException(status_code=400, detail="message_ids required")
    conn = _get_conn()
    conv = _get_conversation(conn, cid)
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="conversation not found")
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id FROM messages WHERE conversation_id=? AND id IN ({placeholders})",
        (cid, *ids),
    ).fetchall()
    found_ids = [int(r["id"]) for r in rows]
    if len(found_ids) != len(ids):
        conn.close()
        raise HTTPException(status_code=400, detail="some message ids are missing or not in this conversation")
    conn.execute(
        f"DELETE FROM messages WHERE conversation_id=? AND id IN ({placeholders})",
        (cid, *ids),
    )
    now = int(time.time())
    conn.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (now, cid))
    conn.commit()
    conn.close()

    await ws_manager.broadcast_conversation(
        cid,
        {
            "type": "messages_deleted",
            "conversation_id": cid,
            "message_ids": found_ids,
        },
    )
    await ws_manager.broadcast_admins({"type": "conversation_changed", "conversation_id": cid})
    return {"ok": True, "conversation_id": cid, "deleted_count": len(found_ids)}


@app.post("/admin/messages/clear-all")
async def admin_clear_conversation_messages(req: AdminClearConversationMessagesReq):
    cid = (req.conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id required")
    conn = _get_conn()
    conv = _get_conversation(conn, cid)
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="conversation not found")
    cur = conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    deleted = int(cur.rowcount or 0)
    now = int(time.time())
    conn.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (now, cid))
    conn.commit()
    conn.close()

    await ws_manager.broadcast_conversation(
        cid,
        {"type": "conversation_cleared", "conversation_id": cid},
    )
    await ws_manager.broadcast_admins({"type": "conversation_changed", "conversation_id": cid})
    return {"ok": True, "conversation_id": cid, "deleted_count": deleted}


@app.get("/admin/db/conversations")
def admin_db_conversations(limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 200))
    offset = max(0, min(offset, 5000))
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT conversation_id, user_id, status, assigned_admin_id, admin_lock, manual_risk, created_at, updated_at
        FROM conversations
        ORDER BY updated_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "limit": limit, "offset": offset}


@app.get("/admin/db/messages/{conversation_id}")
def admin_db_messages(conversation_id: str, limit: int = 400):
    limit = max(1, min(limit, 2000))
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, role, content, ts
        FROM messages
        WHERE conversation_id=?
        ORDER BY id ASC
        LIMIT ?
        """,
        (conversation_id, limit),
    ).fetchall()
    conn.close()
    return {"conversation_id": conversation_id, "items": [dict(r) for r in rows]}


@app.get("/admin/db/inference/{conversation_id}")
def admin_db_inference(conversation_id: str, limit: int = 50):
    limit = max(1, min(limit, 200))
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, input_text, prediction_json, action, reply_text, created_at
        FROM inference_logs
        WHERE conversation_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (conversation_id, limit),
    ).fetchall()
    conn.close()
    items: List[dict] = []
    for r in rows:
        d = dict(r)
        try:
            d["prediction"] = json.loads(d.pop("prediction_json") or "{}")
        except Exception:
            d["prediction"] = {}
        items.append(d)
    return {"conversation_id": conversation_id, "items": items}


@app.get("/admin/db/events")
def admin_db_events(user_id: str, limit: int = 50):
    limit = max(1, min(limit, 200))
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, user_id, event_type, content, conversation_id, created_at
        FROM user_events
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (uid, limit),
    ).fetchall()
    conn.close()
    return {"user_id": uid, "items": [dict(r) for r in rows]}


@app.websocket("/ws/user/{user_id}")
async def ws_user(user_id: str, websocket: WebSocket):
    await ws_manager.connect_user(user_id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw) if raw else {}
            conversation_id = data.get("conversation_id")
            if conversation_id:
                ws_manager.join_conversation(conversation_id, websocket)
                await websocket.send_text(
                    json.dumps(
                        {"type": "joined_conversation", "conversation_id": conversation_id},
                        ensure_ascii=False,
                    )
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/admin/{admin_id}")
async def ws_admin(admin_id: str, websocket: WebSocket):
    await ws_manager.connect_admin(admin_id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw) if raw else {}
            if data.get("action") == "watch_conversation" and data.get("conversation_id"):
                ws_manager.join_conversation(data["conversation_id"], websocket)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "watching_conversation",
                            "conversation_id": data["conversation_id"],
                        },
                        ensure_ascii=False,
                    )
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    def _port_available(port: int = 8010) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(("127.0.0.1", port)) != 0

    port = 8010
    if not _port_available(port):
        print("[backend_api] ERROR: port 8010 is busy. Please stop the occupying process, then retry.")
        raise SystemExit(1)

    print(f"[backend_api] API + 用户页(同源): http://127.0.0.1:{port}/")
    print(f"[backend_api] Admin 页:              http://127.0.0.1:{port}/admin.html")
    print(f"[backend_api] OpenAPI:                http://127.0.0.1:{port}/docs")
    print("[backend_api] Gradio (optional): run `python gradio.py` then open shown URL.")
    uvicorn.run("backend_api:app", host="127.0.0.1", port=port, reload=False)
