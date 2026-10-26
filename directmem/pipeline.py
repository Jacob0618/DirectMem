"""DirectMem query path: cosine top-n retrieval -> non-generative selection of k turns -> chronological reading.

Ingestion stores each turn as written together with its embedding; no generative model is called. At query time the
question embedding retrieves the n most similar turns, one Jev request scores every candidate, the k most probable
turns are put back in conversation order, and the answering LLM reads them.
"""
import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import prompts as P
from .clients import JevError, p_yes
from .data import Benchmark, Question, Turn

# Yes/no question asked for every candidate (LoCoMo, SocialMemBench).
CAND_TRUE = "states a fact, event, date or detail needed for the answer"
CAND_FALSE = "only the same topic or people, without the needed information"
# LongMemEval-S variant; each request also carries four whole-set questions that selection does not use.
MEM_TRUE = "It states a fact, event, date, or detail needed for the answer"
MEM_FALSE = "It only shares the topic or people, without the needed information"
# Seeds of the pseudo-random candidate order used in the paper.
CAND_SEED, MEM_SEED = "e4_locomo_rerank-{}", "r21a_lme_replication-{}"


def noul(instructions: str, true: Optional[str] = None, false: Optional[str] = None) -> Dict[str, Any]:
    q: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
    return q


def candidate_request(question: str, texts: List[str]):
    state = {"question": question, "candidates": {f"c{i}": t for i, t in enumerate(texts)}}
    qs = {f"relevant::c{i}": noul(f"Does `candidates.c{i}` contain specific information that helps answer `question`?",
                                  true=CAND_TRUE, false=CAND_FALSE) for i in range(len(texts))}
    return state, qs


def memory_request(question: str, now: str, texts: List[str]):
    state = {"question": question, "now": now, "memories": {f"m{i}": t for i, t in enumerate(texts)}}
    qs = {
        "sufficient": noul("Do `memories` contain the specific information needed to answer `question`?",
                           true="The answer can be given from these memories",
                           false="The needed information is missing, so the honest answer is that it is unknown"),
        "premise": noul("Is every person, item, or event that `question` refers to actually mentioned in `memories`?"),
        "attribution": noul(
            "Do `memories` show that the person `question` names is the one who did or said what it asks about?",
            true="A memory shows this same named person doing, saying, having or experiencing the thing `question` asks about.",
            false="The memories show a different person doing or saying it, or never show the named person doing or saying it."),
        "asks_fact": noul("Is `question` asking for a specific fact about the user or their past, rather than asking for help "
                          "with a task?"),
    }
    for i in range(len(texts)):
        qs[f"relevant::m{i}"] = noul(f"Does `memories.m{i}` contain specific information that helps answer `question`?",
                                     true=MEM_TRUE, false=MEM_FALSE)
    return state, qs


class DirectMem:
    def __init__(self, bench: Benchmark, gemini, jev, n: int = 100, k: int = 30,
                 answer_model: str = "gemini-3.8-flash", judge_model: str = "gemini-3.8-flash",
                 retry_truncated: bool = True):
        self.b, self.gem, self.jev, self.n, self.k = bench, gemini, jev, n, k
        self.answer_model, self.judge_model, self.retry_truncated = answer_model, judge_model, retry_truncated
        self.turns = {s: {t.id: t for t in T} for s, T in bench.stores.items()}
        self.ids: Dict[str, List[str]] = {}
        self.mat: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ ingestion: embed every stored line once
    async def index(self, log=print) -> None:
        texts = sorted({t.line for T in self.b.stores.values() for t in T})
        log(f"embedding {len(texts):,} unique stored lines")
        vecs = {}
        step = 5000
        for j in range(0, len(texts), step):
            E = await self.gem.embed(texts[j:j + step], "document")
            vecs.update(zip(texts[j:j + step], E))
            log(f"  embedded {min(j + step, len(texts)):,}/{len(texts):,}")
        for s, T in self.b.stores.items():
            self.ids[s] = [t.id for t in T]
            self.mat[s] = np.stack([vecs[t.line] for t in T]) if T else np.zeros((0, 768), np.float32)

    # ------------------------------------------------------------ query path
    def retrieve(self, q: Question, qv: np.ndarray) -> List[str]:
        sims = self.mat[q.store] @ qv
        # Sort kinds as in the paper runs (they only differ on exact ties): numpy default on LoCoMo, stable elsewhere.
        order = np.argsort(-sims) if self.b.name == "locomo" else np.argsort(-sims, kind="stable")
        return [self.ids[q.store][i] for i in order[:self.n]]

    async def _score_candidates(self, q: Question, ids: List[str], depth: int = 0) -> Dict[str, float]:
        T = self.turns[q.store]
        state, qs = candidate_request(q.question, [T[d].line for d in ids])
        try:
            ans = await self.jev.ask(state, qs)
        except JevError as e:  # a request rejected for length is split into two halves
            msg = str(e).lower()
            if depth < 2 and any(s in msg for s in ("token", "too long", "413", "exceed")):
                half = len(ids) // 2
                p: Dict[str, float] = {}
                for part in (ids[:half], ids[half:]):
                    p.update(await self._score_candidates(q, part, depth + 1))
                return p
            raise
        return {d: p_yes(ans[f"relevant::c{i}"]) for i, d in enumerate(ids)}

    async def score(self, q: Question, pool: List[str]) -> Dict[str, float]:
        perm = list(pool)
        if self.b.name == "longmemeval":
            random.Random(MEM_SEED.format(q.id)).shuffle(perm)
            m = 2 if len(perm) > 60 else 1  # 100 candidates exceed the 32k-token input limit of Jev: two blocks of 50
            bounds = [round(len(perm) * c / m) for c in range(m + 1)]
            T, p = self.turns[q.store], {}
            for c in range(m):
                chunk = perm[bounds[c]:bounds[c + 1]]
                state, qs = memory_request(q.question, q.question_date, [T[d].line for d in chunk])
                ans = await self.jev.ask(state, qs)
                p.update({d: p_yes(ans[f"relevant::m{i}"]) for i, d in enumerate(chunk)})
            return p
        random.Random(CAND_SEED.format(q.id)).shuffle(perm)
        return await self._score_candidates(q, perm)

    def select(self, q: Question, pool: List[str], p: Optional[Dict[str, float]]) -> List[str]:
        """k most probable candidates (ties keep the cosine order), returned in conversation order."""
        rank = {d: i for i, d in enumerate(pool)}
        chosen = pool[:self.k] if p is None else sorted(pool, key=lambda d: (-p[d], rank[d]))[:self.k]
        T = self.turns[q.store]
        return sorted(chosen, key=lambda d: T[d].order)

    async def answer(self, q: Question, ctx: List[str]) -> Dict[str, Any]:
        T = self.turns[q.store]
        if self.b.name == "longmemeval":
            prompt = P.lme_answer_prompt({"question": q.question, "question_date": q.question_date}, [T[d].line for d in ctx])
            r = await self.gem.generate(self.answer_model, prompt, schema=P.LME_ANSWER_SCHEMA, thinking="off", stage="answer")
            return {"answer": str(r["obj"].get("answer", "")), "finish_reason": r["finish_reason"]}
        prompt = P.u2_prompt(q.question, q.ref_date, [(T[d].iso, T[d].mem_text) for d in ctx])
        r = await self.gem.generate(self.answer_model, prompt, thinking="off", max_output_tokens=4096, stage="answer")
        if r["finish_reason"] == "MAX_TOKENS" and self.retry_truncated:  # one re-answer with a larger output budget
            r = await self.gem.generate(self.answer_model, prompt, thinking="off", max_output_tokens=16384, stage="answer")
        return {"answer": P.u2_parse(r["text"]), "finish_reason": r["finish_reason"]}

    async def judge(self, q: Question, ans: str) -> int:
        if self.b.name == "longmemeval":
            prompt = P.get_anscheck_prompt(q.category, q.question, q.gold, ans, abstention=q.abstention)
            r = await self.gem.generate(self.judge_model, prompt, schema=P.LME_JUDGE_SCHEMA, thinking="low", stage="judge")
            return int(r["obj"].get("verdict") == "yes")
        if P.is_abstain_text(ans):  # abstentions count as wrong on LoCoMo and SocialMemBench
            return 0
        prompt = P.locomo_judge_prompt({"question": q.question, "answer": q.gold}, ans)
        r = await self.gem.generate(self.judge_model, prompt, schema=P.LOCOMO_JUDGE_SCHEMA, thinking="low", stage="judge")
        return int(r["obj"].get("label") == "CORRECT")

    async def run_question(self, q: Question, qv: np.ndarray) -> Dict[str, Any]:
        pool = self.retrieve(q, qv)
        p = await self.score(q, pool) if self.k < len(pool) else None  # k = n reads all candidates without selection
        ctx = self.select(q, pool, p)
        a = await self.answer(q, ctx)
        correct = await self.judge(q, a["answer"])
        return {"id": q.id, "category": q.category, "question": q.question, "gold": q.gold, "n": self.n, "k": self.k,
                "context": ctx, "answer": a["answer"], "finish_reason": a["finish_reason"], "correct": correct}

    async def evaluate(self, out_dir: str, limit: Optional[int] = None, concurrency: int = 8, log=print) -> Dict[str, Any]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        qs = self.b.questions[:limit] if limit else self.b.questions
        await self.index(log)
        Q = await self.gem.embed([q.question for q in qs], "query")
        sem, rows, t0 = asyncio.Semaphore(concurrency), [], time.time()

        async def one(q: Question, qv: np.ndarray) -> None:
            async with sem:
                rows.append(await self.run_question(q, qv))
                if len(rows) % 50 == 0:
                    log(f"  {len(rows)}/{len(qs)} answered, accuracy so far {100 * np.mean([r['correct'] for r in rows]):.1f}%")

        await asyncio.gather(*[one(q, qv) for q, qv in zip(qs, Q)])
        order = {q.id: i for i, q in enumerate(qs)}
        rows.sort(key=lambda r: order[r["id"]])
        with open(out / "results.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r["correct"])
        summary = {"benchmark": self.b.name, "n": self.n, "k": self.k, "questions": len(rows),
                   "accuracy": round(100 * float(np.mean([r["correct"] for r in rows])), 2),
                   "accuracy_by_category": {c: round(100 * float(np.mean(v)), 2) for c, v in sorted(by_cat.items())},
                   "answer_model": self.answer_model, "judge_model": self.judge_model,
                   "decision_model": getattr(self.jev, "model", None), "seconds": round(time.time() - t0, 1)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary
