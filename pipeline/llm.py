"""DeepSeek (OpenAI-compatible) client with an on-disk cache.

Every call is keyed by (model, messages, temperature, max_tokens, json_mode, tag) and stored in
data/llm_cache/<tag>.jsonl, so reruns are free, interrupted runs resume, and every LLM output used in the
paper is kept. The API key is read from DEEPSEEK_API_KEY; in the cloud session it is injected by the network
proxy, so any placeholder value works there.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "llm_cache")

_client = None
_lock = threading.Lock()
_caches: dict[str, dict] = {}
USAGE = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0}


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", "proxy-injected"),
                         base_url="https://api.deepseek.com", timeout=300, max_retries=0)
    return _client


def _cache(tag: str) -> dict:
    if tag not in _caches:
        os.makedirs(CACHE_DIR, exist_ok=True)
        c = {}
        p = os.path.join(CACHE_DIR, f"{tag}.jsonl")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        c[r["key"]] = r
                    except json.JSONDecodeError:
                        pass  # truncated last line after a crash
        _caches[tag] = c
    return _caches[tag]


def chat(messages, tag: str, temperature=0.6, max_tokens=4096, json_mode=False, seed_key="") -> str:
    """One chat completion. `seed_key` lets callers ask for distinct samples of the same prompt."""
    key = hashlib.sha256(json.dumps([MODEL, messages, temperature, max_tokens, json_mode, seed_key],
                                    ensure_ascii=False).encode()).hexdigest()
    c = _cache(tag)
    with _lock:
        if key in c:
            USAGE["cached"] += 1
            return c[key]["output"]
    kw = dict(model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    err = None
    for attempt in range(6):
        try:
            r = client().chat.completions.create(**kw)
            out = r.choices[0].message.content or ""
            u = r.usage
            break
        except Exception as e:  # rate limit / transient
            err = e
            time.sleep(min(60, 2 ** attempt * 2))
    else:
        raise RuntimeError(f"LLM call failed after retries: {err}")
    rec = {"key": key, "model": r.model, "messages": messages, "temperature": temperature,
           "max_tokens": max_tokens, "output": out, "finish_reason": r.choices[0].finish_reason,
           "usage": {"prompt": u.prompt_tokens, "completion": u.completion_tokens,
                     "cache_hit": getattr(u, "prompt_cache_hit_tokens", 0) or 0}}
    with _lock:
        c[key] = rec
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += u.prompt_tokens
        USAGE["completion_tokens"] += u.completion_tokens
        USAGE["cache_hit_tokens"] += rec["usage"]["cache_hit"]
        with open(os.path.join(CACHE_DIR, f"{tag}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def parse_json(text: str):
    """Parse a JSON object from an LLM reply (tolerates ```json fences)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s >= 0 and e > s:
            return json.loads(t[s:e + 1])
        raise


def pmap(fn, items, workers=16, desc=""):
    """Thread-parallel map preserving order; prints coarse progress."""
    items = list(items)
    out = [None] * len(items)
    done = [0]

    def run(i):
        out[i] = fn(items[i])
        with _lock:
            done[0] += 1
            if done[0] % max(1, len(items) // 10) == 0 or done[0] == len(items):
                print(f"  [{desc}] {done[0]}/{len(items)}  usage={USAGE}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, range(len(items))))
    return out


def cost_usd(usage=USAGE, in_miss=0.27, in_hit=0.07, out=1.10):
    """Rough cost estimate (USD per 1M tokens; check current DeepSeek pricing)."""
    miss = usage["prompt_tokens"] - usage["cache_hit_tokens"]
    return (miss * in_miss + usage["cache_hit_tokens"] * in_hit + usage["completion_tokens"] * out) / 1e6
