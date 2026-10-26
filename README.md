# DirectMem

Minimal code to reproduce DirectMem, a memory architecture for LLM agents that preserves raw conversation turns and
selects evidence at query time with a non-generative decision model.

- **Ingestion** stores every turn as written, serialized with its date and speaker, together with its embedding. No
  generative model is called.
- **Query path**: embedding similarity retrieves the top-`n` candidate turns, one request to the decision model (Jev)
  returns, for every candidate, the probability that it contains information needed to answer the question, the `k`
  most probable turns are put back in conversation order, and the answering LLM reads them. Only this last step
  generates text.

This repository contains only DirectMem (no baselines and no ablations).

## Setup

Python 3.9 or later.

```bash
pip install -r requirements.txt   # optionally also `pip install h2` for HTTP/2
cp .env.example .env              # then fill in the two API keys
bash scripts/download_data.sh     # LoCoMo, LongMemEval-S and SocialMemBench into data/
```

`.env` needs `GEMINI_API_KEY` (Gemini API, used for embeddings, answers and judging) and `TYPESAFE_API_KEY` (TypeSafe
System One API, which serves the Jev decision model).

## Run

```bash
# LoCoMo (1,540 questions of categories 1-4)
python run.py --benchmark locomo --data data/locomo10.json --n 100 --k 30
python run.py --benchmark locomo --data data/locomo10.json --n 100 --k 10

# LongMemEval-S (500 questions)
python run.py --benchmark longmemeval --data data/longmemeval_s_cleaned.json --n 100 --k 30

# SocialMemBench (1,024 grounded questions)
python run.py --benchmark socialmembench --data data/socialmembench --n 100 --k 30
```

Add `--limit 20` for a quick check. Each run writes `outputs/<benchmark>_n<n>_k<k>/results.jsonl` (selected turns,
answer and judgment per question) and `summary.json` (overall and per-category accuracy, new API spend). Every API call
is cached in `cache/directmem.sqlite`, so an interrupted run resumes for free, and a run with another `k` reuses the
embeddings and Jev scores of a previous run with the same `n`.

## Settings

| Component | Setting |
|---|---|
| Stored line | LoCoMo, SocialMemBench: `[<session date>] <speaker>: <text>` (plus `[image: <caption>]` for shared images); LongMemEval-S: one user message and the following reply, `<date> \| user: .. \| assistant: ..`, cut at 2,000 characters |
| Embeddings | `gemini-embedding-2`, 768 dimensions; documents as `title: none \| text: <line>`, questions as `task: search result \| query: <question>`; cosine similarity |
| Selection | `jev-1.13.0`; one yes/no question per candidate, candidates in a seeded pseudo-random order; on LongMemEval-S the 100 candidates are scored in two blocks of 50 to fit the 32k-token input limit of Jev; the top `k` by probability (ties keep the cosine order) are read in conversation order |
| Answering | `gemini-3.8-flash`, temperature 0, thinking off; LoCoMo and SocialMemBench use the Mem0 LoCoMo answer prompt with abstention allowed, LongMemEval-S a JSON answer prompt (see `directmem/prompts.py`) |
| Judging | `gemini-3.8-flash`, thinking low; correct-or-wrong prompt on LoCoMo and SocialMemBench (abstentions count as wrong), official per-type LongMemEval prompts on LongMemEval-S |
| Budgets | `n = 100`; `k = 30` and `k = 10` |

Given the same candidate pools, this code issues the same Jev requests and the same answer prompts as the runs reported
in the paper. API models can still drift over time, so reproduced accuracy may differ slightly.

## Reference results

Accuracy (%) with `n = 100`:

| | LoCoMo | LongMemEval-S | SocialMemBench |
|---|---:|---:|---:|
| DirectMem, `k = 30` | 91.0 | 91.6 | 93.3 |
| DirectMem, `k = 10` | 89.4 | 92.0 | 90.5 |

At list prices, ingestion (embeddings) costs about \$0.05 for LoCoMo and \$9.59 for LongMemEval-S. Answering costs about
\$4.84 (LoCoMo, `k = 30`), \$3.91 (LoCoMo, `k = 10`), \$11.24 (LongMemEval-S, `k = 30`) and \$5.81 (LongMemEval-S,
`k = 10`) per 1,000 questions, including the Jev request and excluding judging.

## Layout

```
run.py                   command-line entry point
directmem/data.py        benchmark loaders (stored lines, questions)
directmem/pipeline.py    retrieval, Jev selection, chronological reading, answering, judging
directmem/prompts.py     answer and judge prompts
directmem/clients.py     Gemini and Jev HTTP clients with an on-disk request cache
scripts/download_data.sh dataset download
tests/test_offline.py    end-to-end test with fake API clients (no network)
```

## Third-party material

The datasets are not redistributed: LoCoMo (CC BY-NC 4.0), LongMemEval (MIT) and SocialMemBench (CC BY 4.0) are
downloaded from their original sources. The LoCoMo answer prompt is taken from the Mem0 evaluation code
(mem0ai/memory-benchmarks, Apache License 2.0) with one sentence changed to allow abstention, and the LongMemEval judge
prompts are the official ones from the LongMemEval release (MIT).
