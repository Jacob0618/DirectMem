"""Async clients for the Gemini REST API and the Jev decision model (TypeSafe System One API), with an on-disk cache.

Every request is cached by its exact content, so an interrupted run resumes without paying twice.
"""
import asyncio
import hashlib
import json
import os
import random
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import numpy as np

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
# USD per 1M tokens (input, output including thinking), list prices used in the paper.
GEN_PRICES = {"gemini-3.8-flash": (0.75, 3.75)}
EMBED_PRICE_PER_M = 0.20
JEV_PRICE_PER_M_INPUT = 0.042  # Jev bills input tokens only
# gemini-embedding-2 ignores taskType; the task is given as a text prefix.
EMBED_PREFIX = {"document": "title: none | text: {}", "query": "task: search result | query: {}"}
EMBED_BATCH = 100
RETRY_STATUS = (429, 500, 502, 503, 504, 529)


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a local .env file into os.environ (never printed)."""
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Cache:
    """sqlite key-value store: sha256(json(key)) -> json(value)."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    @staticmethod
    def _h(key: Any) -> str:
        return hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, key: Any) -> Any:
        with self.lock:
            row = self.db.execute("SELECT v FROM kv WHERE k=?", (self._h(key),)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: Any, value: Any) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (self._h(key), json.dumps(value, ensure_ascii=False)))
            self.db.commit()


@dataclass
class Usage:
    gen_input_tokens: int = 0
    gen_output_tokens: int = 0
    gen_cost: float = 0.0
    embed_chars: int = 0
    jev_input_tokens: int = 0
    jev_calls: int = 0
    by_stage: Dict[str, float] = field(default_factory=dict)

    def add_cost(self, stage: str, usd: float) -> None:
        self.by_stage[stage] = self.by_stage.get(stage, 0.0) + usd

    def summary(self) -> Dict[str, Any]:
        jev_cost = self.jev_input_tokens / 1e6 * JEV_PRICE_PER_M_INPUT
        return {"generation_usd": round(self.gen_cost, 4), "jev_usd": round(jev_cost, 4),
                "jev_calls": self.jev_calls, "embedded_chars": self.embed_chars,
                "by_stage_usd": {k: round(v, 4) for k, v in self.by_stage.items()}}


def _http2() -> bool:
    try:
        import h2  # noqa: F401
        return True
    except ImportError:
        return False


async def _backoff(attempt: int, r: Optional[httpx.Response] = None) -> None:
    ra = r.headers.get("retry-after") if r is not None else None
    delay = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(30.0, 0.5 * 2 ** attempt)
    await asyncio.sleep(delay * (0.75 + random.random() / 2))


class Gemini:
    def __init__(self, cache: Cache, usage: Usage, api_key: Optional[str] = None, embed_model: str = "gemini-embedding-2",
                 dim: int = 768, concurrency: int = 16, max_retries: int = 6):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (environment or .env)")
        self.cache, self.usage = cache, usage
        self.embed_model, self.dim, self.max_retries = embed_model, dim, max_retries
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(http2=_http2(), timeout=httpx.Timeout(180.0, connect=15.0),
                                        headers={"x-goog-api-key": key, "Content-Type": "application/json"})

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _post(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        async with self.sem:
            for attempt in range(self.max_retries + 1):
                try:
                    r = await self.client.post(url, json=body)
                except (httpx.TransportError, httpx.TimeoutException):
                    if attempt == self.max_retries:
                        raise
                    await _backoff(attempt)
                    continue
                if r.status_code == 200:
                    return r.json()
                if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                    await _backoff(attempt, r)
                    continue
                raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
        raise AssertionError("unreachable")

    async def embed(self, texts: Sequence[str], kind: str) -> np.ndarray:
        """Unit-normalised 768-d gemini-embedding-2 vectors; kind is 'document' or 'query'."""
        prefixed = [EMBED_PREFIX[kind].format(t) for t in texts]
        out: List[Optional[List[float]]] = [self.cache.get([self.embed_model, self.dim, p]) for p in prefixed]
        todo = [i for i, v in enumerate(out) if v is None]

        async def batch(idx: List[int]) -> None:
            reqs = [{"model": f"models/{self.embed_model}", "content": {"parts": [{"text": prefixed[i]}]},
                     "outputDimensionality": self.dim} for i in idx]
            d = await self._post(f"{GEMINI_URL}/models/{self.embed_model}:batchEmbedContents", {"requests": reqs})
            for i, e in zip(idx, d["embeddings"]):
                out[i] = e["values"]
                self.cache.set([self.embed_model, self.dim, prefixed[i]], e["values"])
                self.usage.embed_chars += len(prefixed[i])

        await asyncio.gather(*[batch(todo[j:j + EMBED_BATCH]) for j in range(0, len(todo), EMBED_BATCH)])
        E = np.asarray(out, dtype=np.float32).reshape(len(texts), self.dim)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    async def generate(self, model: str, prompt: str, *, schema: Optional[Dict[str, Any]] = None, thinking: str = "off",
                       max_output_tokens: Optional[int] = None, stage: str = "generation") -> Dict[str, Any]:
        """Temperature-0 generation. thinking='off' -> thinkingBudget 0; 'low' -> thinkingLevel low.
        Returns {'text', 'obj' (parsed JSON when schema is given), 'finish_reason'}."""
        gen: Dict[str, Any] = {"temperature": 0.0,
                               "thinkingConfig": {"thinkingBudget": 0} if thinking == "off" else {"thinkingLevel": thinking}}
        if max_output_tokens:
            gen["maxOutputTokens"] = max_output_tokens
        if schema is not None:
            gen["responseMimeType"] = "application/json"
            openapi = isinstance(schema.get("type"), str) and schema["type"].isupper()
            gen["responseSchema" if openapi else "responseJsonSchema"] = schema
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}
        key = [model, body]
        hit = self.cache.get(key)
        if hit is None:
            d = await self._post(f"{GEMINI_URL}/models/{model}:generateContent", body)
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts", [])
            u = d.get("usageMetadata", {})
            hit = {"text": "".join(p.get("text", "") for p in parts if not p.get("thought")),
                   "finish_reason": cand.get("finishReason"),
                   "input_tokens": u.get("promptTokenCount", 0),
                   "output_tokens": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}
            self.cache.set(key, hit)
            pin, pout = GEN_PRICES.get(model, (0.0, 0.0))
            usd = (hit["input_tokens"] * pin + hit["output_tokens"] * pout) / 1e6
            self.usage.gen_input_tokens += hit["input_tokens"]
            self.usage.gen_output_tokens += hit["output_tokens"]
            self.usage.gen_cost += usd
            self.usage.add_cost(stage, usd)
        res = dict(hit)
        if schema is not None:
            try:
                res["obj"] = json.loads(res["text"])
            except json.JSONDecodeError:
                res["obj"] = {}
        return res


class JevError(RuntimeError):
    pass


class Jev:
    """Non-generative decision model: returns a probability for every yes/no question in one request."""

    def __init__(self, cache: Cache, usage: Usage, api_key: Optional[str] = None, model: str = "jev-1.13.0",
                 concurrency: int = 16, max_retries: int = 6):
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set (environment or .env)")
        self.cache, self.usage, self.model, self.max_retries = cache, usage, model, max_retries
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(http2=_http2(), timeout=httpx.Timeout(120.0, connect=15.0),
                                        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})

    async def aclose(self) -> None:
        await self.client.aclose()

    async def ask(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        key = [self.model, state, questions]
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        body = {"state": state, "model": self.model, "questions": questions}
        async with self.sem:
            for attempt in range(self.max_retries + 1):
                try:
                    r = await self.client.post(JEV_URL, json=body)
                except (httpx.TransportError, httpx.TimeoutException):
                    if attempt == self.max_retries:
                        raise
                    await _backoff(attempt)
                    continue
                if r.status_code == 200:
                    d = r.json()
                    self.usage.jev_calls += 1
                    self.usage.jev_input_tokens += d.get("usage", {}).get("input_tokens", 0)
                    self.cache.set(key, d["answers"])
                    return d["answers"]
                if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                    await _backoff(attempt, r)
                    continue
                raise JevError(f"Jev HTTP {r.status_code}: {r.text[:300]}")
        raise AssertionError("unreachable")


def p_yes(answer: Dict[str, Any]) -> float:
    return float(answer["noul"]) if answer.get("type") == "noul" else float(answer.get("probabilities", {}).get("yes", 0.0))
