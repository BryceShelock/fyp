from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from multitask_predict import predict_text

DB_PATH = os.path.join(os.path.dirname(__file__), "chat_app.db")


class ChatSendReq(BaseModel):
    user_id: str
    text: str
    model_dir: str = "multitask_output/best_model"
    device: str = "auto"


class AdminTakeoverReq(BaseModel):
    conversation_id: str
    admin_id: str


class AdminReplyReq(BaseModel):
    conversation_id: str
    admin_id: str
    text: str


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
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
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
    conn.commit()
    conn.close()


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
        "SELECT role, content, ts FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]


def _conversation_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "conversation_id": row["conversation_id"],
        "user_id": row["user_id"],
        "status": row["status"],
        "assigned_admin_id": row["assigned_admin_id"],
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


@app.post("/chat/send")
async def chat_send(req: ChatSendReq):
    conn = _get_conn()
    conv = _get_or_create_conv(conn, req.user_id)
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
    action = pred.get("action", "ai_reply")

    if action == "handoff_to_human":
        status = "waiting_admin"
        reply = "检测到高风险内容，已为你转接人工管理员，请稍等。"
    else:
        status = "ai"
        risk_level = pred.get("风险等级", "未知")
        state = pred.get("心理状态", "未知")
        reply = f"我在这里陪你。当前识别为{risk_level}，心理状态倾向{state}。你愿意多说一点吗？"

    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, 'ai', ?, ?)",
        (conv["conversation_id"], reply, now),
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
                {"role": "ai", "content": reply, "ts": now},
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
    }


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
        "UPDATE conversations SET status='admin', assigned_admin_id=?, updated_at=? WHERE conversation_id=?",
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

    def _pick_port(preferred_port: int = 8010) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", preferred_port)) != 0:
                return preferred_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp:
            temp.bind(("127.0.0.1", 0))
            return int(temp.getsockname()[1])

    base_port = int(os.getenv("BACKEND_PORT", "8010"))
    port = _pick_port(base_port)
    if port != base_port:
        print(f"[backend_api] port {base_port} is busy, fallback to {port}")
    uvicorn.run("backend_api:app", host="127.0.0.1", port=port, reload=False)
