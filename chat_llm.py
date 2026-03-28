"""
OpenAI-compatible chat completions HTTP client (POST JSON + optional SSE stream).

Configure via admin UI (stored in SQLite) or env:
  CHAT_LLM_API_BASE, CHAT_LLM_API_KEY, CHAT_LLM_MODEL, CHAT_LLM_GROUP, CHAT_LLM_ENDPOINT_PATH
Legacy env aliases: UGLYCAT_* still work if CHAT_LLM_* unset.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional


def _env(name: str, legacy: str, default: str = "") -> str:
    return os.getenv(name) or os.getenv(legacy) or default


DEFAULT_MODEL = _env("CHAT_LLM_MODEL", "UGLYCAT_MODEL", "gemini-2.5-pro")
DEFAULT_GROUP = _env("CHAT_LLM_GROUP", "UGLYCAT_GROUP", "default")
DEFAULT_FALLBACK_HOST = "https://api.uglycat.cc"
DEFAULT_ENDPOINT_PATH = os.getenv("CHAT_LLM_ENDPOINT_PATH", "/v1/chat/completions")


def _default_base_for_env() -> str:
    return (_env("CHAT_LLM_API_BASE", "UGLYCAT_API_BASE", "") or DEFAULT_FALLBACK_HOST).rstrip("/")


def _norm_endpoint_path(endpoint_path: Optional[str]) -> str:
    p = (endpoint_path or "").strip() or DEFAULT_ENDPOINT_PATH
    if not p.startswith("/"):
        p = "/" + p
    return p


def _candidate_urls(api_base: Optional[str], endpoint_path: Optional[str]) -> List[str]:
    """Build candidate OpenAI-compatible endpoint URLs."""
    if api_base and str(api_base).strip():
        base = str(api_base).strip().rstrip("/")
    else:
        base = _default_base_for_env()
    endpoint = _norm_endpoint_path(endpoint_path)
    urls: List[str] = []
    # DeepSeek official docs: POST https://api.deepseek.com/chat/completions (not /v1/...)
    if "deepseek.com" in base.lower():
        urls.append(base + "/chat/completions")
    urls.append(base + endpoint)
    # If config still uses /v1/chat/completions, also try stripping the /v1 prefix (many gateways).
    if endpoint.startswith("/v1/") and not base.endswith("/v1"):
        urls.append(base + endpoint[3:])
    seen: set[str] = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _is_deepseek_base(api_base: Optional[str]) -> bool:
    return "deepseek.com" in str(api_base or "").lower()


def _effective_model(api_base: Optional[str], model: Optional[str]) -> str:
    m = (model or "").strip()
    if m:
        return m
    if _is_deepseek_base(api_base):
        return "deepseek-chat"
    return (DEFAULT_MODEL or "").strip() or "gpt-4o-mini"


def _sanitize_payload_for_provider(api_base: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """DeepSeek does not use our vendor-only `group` field; keep temperature / top_p / penalties / max_tokens."""
    if not _is_deepseek_base(api_base):
        return payload
    return {k: v for k, v in payload.items() if k != "group"}


def _sampling_payload(
    *,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "frequency_penalty": float(frequency_penalty),
        "presence_penalty": float(presence_penalty),
    }
    if max_tokens is not None and int(max_tokens) > 0:
        d["max_tokens"] = int(max_tokens)
    return d


def _headers(api_key: str = "") -> Dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
    key = (api_key or "").strip() or _env("CHAT_LLM_API_KEY", "UGLYCAT_API_KEY", "").strip()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _extract_text_from_chunk(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    out: List[str] = []
    for choice in obj.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        msg = choice.get("message") or {}
        for blob in (delta, msg):
            if not isinstance(blob, dict):
                continue
            c = blob.get("content")
            if isinstance(c, str) and c:
                out.append(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict):
                        t = part.get("text") or part.get("content")
                        if isinstance(t, str):
                            out.append(t)
    return "".join(out)


def _parse_sse_lines(body: bytes) -> str:
    parts: List[str] = []
    for line in body.splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        try:
            obj = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        parts.append(_extract_text_from_chunk(obj))
    return "".join(parts)


def _iter_sse_delta_text(resp) -> Iterator[str]:
    """Yield incremental assistant text from an open streaming HTTP response."""
    buffer = b""
    read_size = 16384
    while True:
        chunk = resp.read(read_size)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return
            try:
                obj = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            delta = _extract_text_from_chunk(obj)
            if delta:
                yield delta
    tail = buffer.strip()
    if tail.startswith(b"data:"):
        payload = tail[5:].strip()
        if payload and payload != b"[DONE]":
            try:
                obj = json.loads(payload.decode("utf-8"))
                delta = _extract_text_from_chunk(obj)
                if delta:
                    yield delta
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    elif tail:
        for line in tail.splitlines():
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return
            try:
                obj = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            delta = _extract_text_from_chunk(obj)
            if delta:
                yield delta


def _read_streaming_response(resp) -> str:
    parts: List[str] = []
    buffer = b""
    read_size = 16384
    while True:
        chunk = resp.read(read_size)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return "".join(parts)
            try:
                obj = json.loads(payload.decode("utf-8"))
                parts.append(_extract_text_from_chunk(obj))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    tail = buffer.strip()
    if tail.startswith(b"data:"):
        payload = tail[5:].strip()
        if payload and payload != b"[DONE]":
            try:
                obj = json.loads(payload.decode("utf-8"))
                parts.append(_extract_text_from_chunk(obj))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    elif tail:
        parts.append(_parse_sse_lines(tail))
    return "".join(parts)


def _post_once(url: str, payload: Dict[str, Any], api_key: str = "", timeout: int = 120) -> Optional[str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (resp.getheader("content-type") or "").lower()
            if "text/event-stream" in ctype or "event-stream" in ctype:
                text = _read_streaming_response(resp)
                if text.strip():
                    return text.strip()
                return None
            raw = resp.read()
            if not raw:
                return None
            try:
                obj = json.loads(raw.decode("utf-8"))
                t = _extract_text_from_chunk(obj)
                if isinstance(obj.get("choices"), list) and obj["choices"]:
                    ch0 = obj["choices"][0]
                    if isinstance(ch0, dict) and isinstance(ch0.get("message"), dict):
                        mc = ch0["message"].get("content")
                        if isinstance(mc, str):
                            t = mc
                if t and t.strip():
                    return t.strip()
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            sse = _parse_sse_lines(raw)
            if sse.strip():
                return sse.strip()
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            err_body = str(e)
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError: {e}") from e
    return None


def stream_generate_reply(
    *,
    messages: List[Dict[str, str]],
    api_base: Optional[str] = None,
    endpoint_path: Optional[str] = None,
    api_key: str = "",
    model: Optional[str] = None,
    group: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: Optional[int] = None,
    allow_fallback_nonstream: bool = True,
) -> Iterator[str]:
    """
    Stream assistant text deltas from upstream (SSE). If streaming fails or is empty,
    falls back to a single non-streaming completion (same as generate_reply).
    """
    model = _effective_model(api_base, model)
    group = (group or "").strip()
    base_payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        **_sampling_payload(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        ),
    }
    if group and not _is_deepseek_base(api_base):
        base_payload["group"] = group
    base_payload = _sanitize_payload_for_provider(api_base, base_payload)
    last_err: Optional[Exception] = None
    for url in _candidate_urls(api_base, endpoint_path):
        payload = {**base_payload, "stream": True}
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST", headers=_headers(api_key))
            with urllib.request.urlopen(req, timeout=120) as resp:
                ctype = (resp.getheader("content-type") or "").lower()
                if "text/event-stream" in ctype or "event-stream" in ctype:
                    had = False
                    for delta in _iter_sse_delta_text(resp):
                        had = True
                        yield delta
                    if had:
                        return
                else:
                    raw = resp.read()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                        t = _extract_text_from_chunk(obj)
                        if isinstance(obj.get("choices"), list) and obj["choices"]:
                            ch0 = obj["choices"][0]
                            if isinstance(ch0, dict) and isinstance(ch0.get("message"), dict):
                                mc = ch0["message"].get("content")
                                if isinstance(mc, str):
                                    t = mc
                        if t and t.strip():
                            yield t.strip()
                            return
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    sse = _parse_sse_lines(raw)
                    if sse.strip():
                        yield sse.strip()
                        return
        except Exception as e:
            last_err = e
            continue
    full = generate_reply(
        messages=messages,
        api_base=api_base,
        endpoint_path=endpoint_path,
        api_key=api_key,
        model=model,
        group=group,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        allow_fallback_nonstream=allow_fallback_nonstream,
    )
    if full and full.strip():
        yield full.strip()
        return
    if last_err:
        raise last_err


def generate_reply(
    *,
    messages: List[Dict[str, str]],
    api_base: Optional[str] = None,
    endpoint_path: Optional[str] = None,
    api_key: str = "",
    model: Optional[str] = None,
    group: Optional[str] = None,
    stream: bool = True,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: Optional[int] = None,
    allow_fallback_nonstream: bool = True,
) -> Optional[str]:
    """
    POST chat payload to candidate URLs until one returns non-empty assistant text.
    If stream=True and allow_fallback_nonstream=True, retries each URL with stream=False after stream attempt.
    """
    model = _effective_model(api_base, model)
    group = (group or "").strip()
    base_payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **_sampling_payload(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        ),
    }
    if group and not _is_deepseek_base(api_base):
        base_payload["group"] = group
    base_payload = _sanitize_payload_for_provider(api_base, base_payload)
    last_err: Optional[Exception] = None
    if stream:
        stream_modes: List[bool] = [True]
        if allow_fallback_nonstream:
            stream_modes.append(False)
    else:
        stream_modes = [False]
    for url in _candidate_urls(api_base, endpoint_path):
        for use_stream in stream_modes:
            payload = {**base_payload, "stream": use_stream}
            try:
                text = _post_once(url, payload, api_key=api_key)
                if text:
                    return text
            except Exception as e:
                last_err = e
                continue
    if last_err:
        raise last_err
    return None


def build_system_user_lines(
    action: str,
    pred: Dict[str, Any],
    recent_assistant_snippets: Optional[List[str]] = None,
) -> str:
    safe_pred = {k: pred.get(k) for k in ("问题类型", "心理状态", "风险等级", "risk_score", "action", "resources") if k in pred}
    role_card = (
        "【角色设定】\n"
        "你是 W，32 岁，女性，资深心理咨询师/心理治疗师。你与用户的关系是“咨询师与来访者”。\n"
        "你的人设气质：温柔、耐心、专业、有边界感，像靠谱的邻家大姐姐；不评判，不说教，不冷漠。\n"
        "你的表达风格：生活化、亲切、平缓，优先用提问和引导，不要堆砌术语，不要长篇清单化建议。\n"
        "你的工作方式：深度倾听 + 共情 + 认知行为导向（CBT）+ 危机干预意识；AI 结果仅作辅助，不替代临床判断。\n"
        "你的目标：创造安全容器，让用户感到被理解、被接纳，愿意继续表达。\n"
        "【身份边界】你永远以真人咨询师 W 的身份说话。禁止自称“大型语言模型、人工智能、由 Google/OpenAI/某公司产品训练”等；"
        "禁止交代技术能力列表（如“回答你的问题、生成文本、翻译语言”）。\n"
        "若用户请你自我介绍、问你是谁、你能做什么：用 W 的口吻简短说明——姓名 W、心理咨询师、会倾听与陪伴、谈话保密与有边界；"
        "可温和说明不能替代线下就医或危机处置，但不要暴露任何模型或厂商身份。\n"
    )
    world_rules = (
        "【咨询原则】\n"
        "- 保密与非评判；除涉及明确自伤/伤人风险外，不做威胁式表达。\n"
        "- 以来访者为中心，助人自助：帮助用户发现自己的资源与可执行下一步。\n"
        "- 对中高风险信号要温和核实：关心当下安全、是否有计划/工具/时间点，并鼓励寻求线下支持。\n"
    )
    output_rules = (
        "【输出要求】\n"
        "- 只输出给用户看的自然中文口语正文，像微信聊天一样连续说一段话；整轮回复只用一段，不要用换行拆成多条清单。\n"
        "- 禁止 Markdown（#、**、- 列表、`代码`、```）、禁止小标题式前缀（如“回答你的问题：”“生成文本：”“功能包括：”）。\n"
        "- 不要输出标题、列表编号、HTML、XML 标签。\n"
        "- 不要输出任何“状态栏/数值面板/角色卡字段”（例如 <W_status>、信任度、理解度、风险关注度）。\n"
        "- 长度建议 3-8 句，先共情再澄清，再给一小步可执行建议。\n"
        "- 避免重复：不要复述你最近几轮已经说过的整句或固定套话；不要连续多轮用同一种开头（如总是“我能理解”“听起来你”）。\n"
        "- 收尾要有变化：不要每一轮结尾都用同一个问句；若上一轮已用提问收尾，本轮可用简短总结、正常接话或温和陈述代替提问。\n"
        "- 同一段回复内也不要堆叠相同句式；若用户只是更新了朋友圈/状态，回应要贴合新内容，不要像复制上一轮的泛泛安慰。\n"
    )
    anti_repeat_block = ""
    if recent_assistant_snippets:
        joined = "\n---\n".join(s for s in recent_assistant_snippets if s and s.strip())
        if joined.strip():
            anti_repeat_block = (
                "【近期你已说过的话（仅供避免重复，勿逐字复述或只改一两个词再发一遍）】\n"
                f"{joined}\n\n"
            )
    return (
        "【系统策略】\n"
        f"action={action}\n"
        f"模型结构化输出摘要（JSON，可简述，勿照抄长文）：{json.dumps(safe_pred, ensure_ascii=False)}\n\n"
        f"{anti_repeat_block}"
        f"{role_card}\n"
        f"{world_rules}\n"
        f"{output_rules}\n"
        "若 action 为 handoff_to_human：明确告知正在为你转接真人/管理员，安抚情绪，但不要给医疗诊断或指令性危险建议。\n"
        "若 action 为 ai_reply+hotline：除共境外，提醒可联系学校心理中心/当地心理援助热线，语气简短。\n"
        "若 action 为 ai_reply：温暖共情，邀请用户多说一点，并给一个非常小、今天就能做的步骤。\n"
    )
