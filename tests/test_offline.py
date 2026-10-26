"""Offline test of the full DirectMem pipeline with fake API clients (no network, no spend).

Run: python tests/test_offline.py   (or pytest)
Set DIRECTMEM_DATA=<dir with locomo10.json, longmemeval_s_cleaned.json, socialmembench/> to also check the loaders
against the benchmark sizes reported in the paper.
"""
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from directmem import LOADERS, DirectMem  # noqa: E402
from directmem import prompts as P  # noqa: E402


def _vec(text, dim=768):
    v = np.zeros(dim, np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % dim] += 1.0
    return v / max(np.linalg.norm(v), 1e-9)


class FakeGemini:
    def __init__(self):
        self.prompts = []

    async def embed(self, texts, kind):
        return np.stack([_vec(t.split("| ", 1)[-1]) for t in texts]) if texts else np.zeros((0, 768), np.float32)

    async def generate(self, model, prompt, *, schema=None, thinking="off", max_output_tokens=None, stage=""):
        self.prompts.append(prompt)
        if schema is P.LOCOMO_JUDGE_SCHEMA:
            return {"text": "", "obj": {"reason": "", "label": "CORRECT" if "blue" in prompt.split("Generated answer:")[-1] else "WRONG"},
                    "finish_reason": "STOP"}
        if schema is P.LME_JUDGE_SCHEMA:
            return {"text": "", "obj": {"verdict": "yes" if "Paris" in prompt.split("Model Response:")[-1] else "no"},
                    "finish_reason": "STOP"}
        if schema is P.LME_ANSWER_SCHEMA:
            return {"text": "", "obj": {"answer": "Paris" if "Paris" in prompt else "I don't know", "unknown": False},
                    "finish_reason": "STOP"}
        return {"text": "Step 1 ...\nANSWER: blue" if "blue" in prompt else "ANSWER: not mentioned", "finish_reason": "STOP"}


class FakeJev:
    model = "fake"

    def __init__(self):
        self.requests = []

    async def ask(self, state, questions):
        self.requests.append((state, questions))
        field = "candidates" if "candidates" in state else "memories"
        qv = _vec(state["question"])
        out = {name: {"type": "noul", "noul": 0.5} for name in questions}
        for key, text in state[field].items():
            out[f"relevant::{key}"] = {"type": "noul", "noul": float(_vec(text) @ qv)}
        return out


def _locomo_fixture(d):
    conv = {"sample_id": "conv-1", "conversation": {
        "speaker_a": "Ann", "speaker_b": "Bob",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"dia_id": "D1:1", "speaker": "Ann", "text": "I bought a blue bike yesterday."},
                      {"dia_id": "D1:2", "speaker": "Bob", "text": "Nice, what about your garden?"},
                      {"dia_id": "D1:3", "speaker": "Ann", "text": "The garden is full of tomatoes.",
                       "blip_caption": "a photo of tomatoes"}],
        "session_2_date_time": "10:00 am on 9 June, 2023",
        "session_2": [{"dia_id": "D2:1", "speaker": "Bob", "text": "Did you ride the bike to work?"},
                      {"dia_id": "D2:2", "speaker": "Ann", "text": "Yes, the blue bike is fast."}]},
        "qa": [{"question": "What color is the bike Ann bought?", "answer": "blue", "category": 4, "evidence": ["D1:1"]},
               {"question": "What grows in the garden?", "answer": "tomatoes", "category": 1, "evidence": ["D1:3"]},
               {"question": "Adversarial?", "adversarial_answer": "x", "category": 5, "evidence": []}]}
    p = Path(d) / "locomo10.json"
    p.write_text(json.dumps([conv]))
    return str(p)


def _lme_fixture(d):
    q = {"question_id": "q1", "question_type": "single-session-user", "question": "Which city did I move to?",
         "answer": "Paris", "question_date": "2023/06/01 (Thu) 10:00",
         "haystack_session_ids": ["s1", "s2"], "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/05/20 (Sat) 12:00"],
         "haystack_sessions": [[{"role": "user", "content": "I just moved to Paris for work."},
                                {"role": "assistant", "content": "Congratulations on the move!"}],
                               [{"role": "user", "content": "Any tips for baking bread?"},
                                {"role": "assistant", "content": "Use a hot oven."},
                                {"role": "user", "content": "Thanks."}]]}
    p = Path(d) / "lme.json"
    p.write_text(json.dumps([q]))
    return str(p)


def test_locomo_pipeline():
    with tempfile.TemporaryDirectory() as d:
        bench = LOADERS["locomo"](_locomo_fixture(d))
        assert len(bench.questions) == 2 and len(bench.stores["conv-1"]) == 5
        t = bench.stores["conv-1"][2]
        assert t.line == "[1:56 pm on 8 May, 2023] Ann: The garden is full of tomatoes. [image: a photo of tomatoes]"
        assert bench.questions[0].ref_date == "10:00 am on 9 June, 2023"
        gem, jev = FakeGemini(), FakeJev()
        dm = DirectMem(bench, gem, jev, n=4, k=2)
        s = asyncio.run(dm.evaluate(str(Path(d) / "out"), concurrency=2, log=lambda *a: None))
        rows = [json.loads(x) for x in open(Path(d) / "out" / "results.jsonl")]
        assert s["questions"] == 2 and rows[0]["answer"] == "blue" and rows[0]["correct"] == 1
        assert all(len(r["context"]) == 2 for r in rows)
        order = {t.id: t.order for t in bench.stores["conv-1"]}
        assert all([order[c] for c in r["context"]] == sorted(order[c] for c in r["context"]) for r in rows)
        assert len(jev.requests) == 2 and len(jev.requests[0][0]["candidates"]) == 4
        assert "(2023-05-08) Ann: I bought a blue bike yesterday." in gem.prompts[0]


def test_longmemeval_pipeline():
    with tempfile.TemporaryDirectory() as d:
        bench = LOADERS["longmemeval"](_lme_fixture(d))
        R = bench.stores["q1"]
        assert [r.line for r in R][0] == "2023/05/01 (Mon) 09:00 | user: I just moved to Paris for work. | assistant: Congratulations on the move!"
        assert len(R) == 3  # the trailing user message forms a round of its own
        gem, jev = FakeGemini(), FakeJev()
        s = asyncio.run(DirectMem(bench, gem, jev, n=3, k=1).evaluate(str(Path(d) / "out"), log=lambda *a: None))
        assert s["accuracy"] == 100.0
        st, qs = jev.requests[0]
        assert st["now"] == "2023/06/01 (Thu) 10:00" and list(qs)[:4] == ["sufficient", "premise", "attribution", "asks_fact"]


def test_real_loaders():
    root = os.environ.get("DIRECTMEM_DATA")
    if not root:
        print("skip real-data loader check (set DIRECTMEM_DATA)")
        return
    root = Path(root)
    b = LOADERS["locomo"](str(root / "locomo10.json"))
    assert len(b.questions) == 1540 and sum(map(len, b.stores.values())) == 5882
    b = LOADERS["longmemeval"](str(root / "longmemeval_s_cleaned.json"))
    assert len(b.questions) == 500 and sum(map(len, b.stores.values())) == 124366
    b = LOADERS["socialmembench"](str(root / "socialmembench"))
    assert len(b.questions) == 1024 and len(b.stores) == 43 and sum(map(len, b.stores.values())) == 7355
    print("real-data loaders match the paper: 1,540 / 500 / 1,024 questions")


if __name__ == "__main__":
    test_locomo_pipeline()
    test_longmemeval_pipeline()
    test_real_loaders()
    print("all offline tests passed")
