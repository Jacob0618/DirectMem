"""Run DirectMem on LoCoMo, LongMemEval-S or SocialMemBench.

Example:
    python run.py --benchmark locomo --data data/locomo10.json --n 100 --k 30 --out outputs/locomo_k30
"""
import argparse
import asyncio
import json
import time

from directmem import LOADERS, DirectMem
from directmem.clients import Cache, Gemini, Jev, Usage, load_dotenv


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


async def main(a: argparse.Namespace) -> None:
    bench = LOADERS[a.benchmark](a.data)
    log(f"{a.benchmark}: {len(bench.stores)} histories, {sum(map(len, bench.stores.values())):,} stored turns, "
        f"{len(bench.questions):,} questions")
    cache, usage = Cache(a.cache), Usage()
    gem = Gemini(cache, usage, concurrency=a.concurrency * 2)
    jev = Jev(cache, usage, model=a.decision_model, concurrency=a.concurrency)
    try:
        dm = DirectMem(bench, gem, jev, n=a.n, k=a.k, answer_model=a.answer_model, judge_model=a.judge_model)
        summary = await dm.evaluate(a.out, limit=a.limit, concurrency=a.concurrency, log=log)
    finally:
        await gem.aclose()
        await jev.aclose()
    summary["new_spend_this_run"] = usage.summary()
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(LOADERS))
    ap.add_argument("--data", required=True, help="locomo10.json | longmemeval_s_cleaned.json | SocialMemBench directory")
    ap.add_argument("--n", type=int, default=100, help="candidate budget: turns retrieved by embedding similarity")
    ap.add_argument("--k", type=int, default=30, help="selection budget: turns the answering model reads")
    ap.add_argument("--out", default=None, help="output directory (default: outputs/<benchmark>_n<n>_k<k>)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (quick check)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--answer-model", default="gemini-3.8-flash")
    ap.add_argument("--judge-model", default="gemini-3.8-flash")
    ap.add_argument("--decision-model", default="jev-1.13.0")
    ap.add_argument("--cache", default="cache/directmem.sqlite", help="request cache; reruns reuse every cached call")
    ap.add_argument("--env", default=".env", help="file with GEMINI_API_KEY and TYPESAFE_API_KEY")
    args = ap.parse_args()
    args.out = args.out or f"outputs/{args.benchmark}_n{args.n}_k{args.k}"
    load_dotenv(args.env)
    asyncio.run(main(args))
