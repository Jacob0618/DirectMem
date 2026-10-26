"""Benchmark loaders. Each benchmark becomes a set of stores (one conversation history each) and a list of questions.

A stored turn keeps the line that DirectMem embeds and passes to the decision model, plus what the answer prompt needs.
"""
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOCOMO_CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
LME_MAX_CHARS = 2000  # LongMemEval-S rounds are cut to their first 2,000 characters


@dataclass
class Turn:
    id: str
    line: str                      # stored line: embedded, scored by the decision model, read by the answering model
    order: Any                     # chronological sort key within the store
    mem_text: Optional[str] = None  # 'Speaker: text' for the Mem0-style answer prompt (LoCoMo, SocialMemBench)
    iso: Optional[str] = None       # session date as an ISO string (LoCoMo, SocialMemBench)


@dataclass
class Question:
    id: str
    store: str
    question: str
    gold: str
    category: str
    ref_date: Optional[str] = None       # LoCoMo / SocialMemBench: date of the last session
    question_date: Optional[str] = None  # LongMemEval-S
    abstention: bool = False             # LongMemEval-S unanswerable questions ('_abs')


@dataclass
class Benchmark:
    name: str
    stores: Dict[str, List[Turn]]
    questions: List[Question]


# ------------------------------------------------------------------ LoCoMo
def _locomo_dt(s: str) -> dt.datetime:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return dt.datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"unparsed LoCoMo date {s!r}")


def load_locomo(path: str) -> Benchmark:
    """locomo10.json: 10 conversations; the 1,540 questions of categories 1-4 (adversarial category 5 excluded)."""
    stores, questions = {}, []
    for conv in json.load(open(path)):
        cid, c = conv["sample_id"], conv["conversation"]
        idxs = sorted(int(k.split("_")[1]) for k in c if k.startswith("session_") and not k.endswith("date_time"))
        turns: List[Turn] = []
        for i in idxs:
            date = c[f"session_{i}_date_time"]
            iso = _locomo_dt(date).replace(tzinfo=dt.timezone.utc).isoformat()
            for t in c[f"session_{i}"]:
                cap = f" [image: {t['blip_caption']}]" if t.get("blip_caption") else ""
                turns.append(Turn(id=t["dia_id"], line=f"[{date}] {t['speaker']}: {t['text']}{cap}", order=len(turns),
                                  mem_text=f"{t['speaker']}: {t['text']}{cap}", iso=iso))
        stores[cid] = turns
        last = c[f"session_{max(idxs)}_date_time"]
        for qi, qa in enumerate(conv["qa"]):
            cat = int(qa["category"])
            if cat in LOCOMO_CATEGORY:
                questions.append(Question(id=f"{cid}::{qi}", store=cid, question=qa["question"],
                                          gold=str(qa.get("answer", "")), category=LOCOMO_CATEGORY[cat], ref_date=last))
    return Benchmark("locomo", stores, questions)


# ------------------------------------------------------------------ LongMemEval-S
def lme_rounds(q: Dict[str, Any]) -> List[Turn]:
    """A user message and the following assistant reply form one round: '<date> | user: .. | assistant: ..'."""
    R: List[Turn] = []
    for si, (sid, date, sess) in enumerate(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])):
        i = r = 0
        while i < len(sess):
            idx = [i]
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant":
                idx.append(i + 1)
            text = (f"{date} | " + " | ".join(f"{sess[t]['role']}: {sess[t]['content']}" for t in idx))[:LME_MAX_CHARS]
            R.append(Turn(id=f"{si}:{sid}#{r}", line=text, order=(date, si, r)))
            i = idx[-1] + 1
            r += 1
    return R


def load_longmemeval(path: str) -> Benchmark:
    """longmemeval_s_cleaned.json: 500 questions, each with its own history."""
    stores, questions = {}, []
    for q in json.load(open(path)):
        qid = q["question_id"]
        stores[qid] = lme_rounds(q)
        questions.append(Question(id=qid, store=qid, question=q["question"], gold=str(q["answer"]),
                                  category=q["question_type"], question_date=q["question_date"],
                                  abstention=qid.endswith("_abs")))
    return Benchmark("longmemeval", stores, questions)


# ------------------------------------------------------------------ SocialMemBench
def _turn_dt(s: str) -> dt.datetime:
    """'2025-10-06T12:08:00' -> datetime; a minute value of 60 rolls over to the next hour."""
    d, t = s.split("T")
    y, mo, da = map(int, d.split("-"))
    h, mi, se = map(int, t.split(":"))
    return dt.datetime(y, mo, da, h, 0, se) + dt.timedelta(minutes=mi)


def _locomo_style_date(d: dt.datetime) -> str:
    return f"{d.hour % 12 or 12}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'} on {d.day} {d.strftime('%B')}, {d.year}"


def _options(q: Dict[str, Any]) -> Dict[str, str]:
    o = json.loads(q["options_json"])
    if isinstance(o, list):
        o = dict((x["option"], x["name"]) if isinstance(x, dict) else re.match(r"^\s*\(?([A-Z])[).]\s*(.*)$", x, re.S).groups()
                 for x in o)
    return {k: v.strip() for k, v in o.items()}


def _question_text(q: Dict[str, Any]) -> str:
    if q["answer_format"] == "multiple_choice":
        return q["question"].strip() + "\nOptions: " + "; ".join(f"({k}) {v}" for k, v in _options(q).items())
    return q["question"].strip()


def _gold_text(q: Dict[str, Any]) -> str:
    ans = (q["answer"] or "").strip()
    if q["answer_format"] == "multiple_choice":
        opts, c = _options(q), q["correct_option"]
        return f"({c}) {opts[c]}" + (f" -- {ans}" if ans and ans != opts[c] else "")
    return ans


def load_socialmembench(directory: str) -> Benchmark:
    """conversations.parquet + qa.parquet: 43 group conversations; the 1,024 questions that passed the grounding check."""
    import pyarrow.parquet as pq
    d = Path(directory)
    C = pq.read_table(d / "conversations.parquet").to_pylist()
    Q = pq.read_table(d / "qa.parquet").to_pylist()
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for r in C:
        by_group.setdefault(r["network_id"], []).append(r)
    stores, ref_date = {}, {}
    for g, rows in sorted(by_group.items()):
        rows.sort(key=lambda r: (r["session_index"], r["message_index"], r["turn_id"]))
        T: List[Turn] = []
        latest: Tuple[str, str] = ("", "")
        for r in rows:
            t = _turn_dt(r["timestamp"])
            date, spk, msg = _locomo_style_date(t), r["speaker_display_name"], r["message"]
            iso = t.replace(tzinfo=dt.timezone.utc).isoformat()
            T.append(Turn(id=r["turn_id"], line=f"[{date}] {spk}: {msg}", order=len(T), mem_text=f"{spk}: {msg}", iso=iso))
            latest = max(latest, (iso, date))
        stores[g], ref_date[g] = T, latest[1]
    questions = [Question(id=f"{q['network_id']}|{q['qa_id']}", store=q["network_id"], question=_question_text(q),
                          gold=_gold_text(q), category=q["query_type"], ref_date=ref_date[q["network_id"]])
                 for q in Q if str(q["qc_phase2_grounded"]) == "True"]
    return Benchmark("socialmembench", stores, questions)


LOADERS = {"locomo": load_locomo, "longmemeval": load_longmemeval, "socialmembench": load_socialmembench}
