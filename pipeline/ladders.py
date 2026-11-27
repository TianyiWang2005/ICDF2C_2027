"""LLM generalisation agent: build generalisation ladders for QI values that the rule hierarchies cannot handle
(events/MISC, DEM, QUANTITY, quasi PERSON/CODE, unmatched ORG, durations...).

usage: python -m pipeline.ladders --inp data/qi/sage_qi.jsonl --out data/generalisation/llm_ladders.json

Ladders are generated once per distinct (type, normalised value) and cached, so the iterative risk-graph loop
stays deterministic and reproducible: the loop decides WHICH value to raise, the ladder says WHAT it becomes.
"""
import argparse
import json
import os

from pipeline import llm
from pipeline import prompts as P
from qi_gen import core
from qi_gen.io import load_jsonl

BATCH = 40


def needed(recs):
    core.EXTRA_LADDERS.clear()
    docs = core.build_docs(recs)
    return sorted({(e.etype, e.ladder[0]) for d in docs for e in d.quasi if len(e.ladder) == 2})


def run_batch(batch, seed_key=""):
    items = "\n".join(f'- {t} "{v}"' for t, v in batch)
    out = llm.chat([{"role": "system", "content": P.LADDER_SYSTEM},
                    {"role": "user", "content": P.LADDER_PROMPT.format(items=items)}],
                   tag="ladders", temperature=0.0, max_tokens=8192, json_mode=True, seed_key=seed_key)
    try:
        lad = llm.parse_json(out).get("ladders", {})
    except Exception:
        lad = {}
    res = {}
    for t, v in batch:
        steps = lad.get(v) or lad.get(f'{t} "{v}"') or []
        steps = [str(s).strip() for s in steps if isinstance(s, str) and str(s).strip()][:3]
        res[f"{t}\t{v}"] = steps
    return res


def load(path):
    """Load ladders into qi_gen.core.EXTRA_LADDERS."""
    core.EXTRA_LADDERS.clear()
    for k, steps in json.load(open(path, encoding="utf-8")).items():
        t, v = k.split("\t", 1)
        core.EXTRA_LADDERS[(t, v)] = steps
    return core.EXTRA_LADDERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/qi/sage_qi.jsonl")
    ap.add_argument("--out", default="data/generalisation/llm_ladders.json")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    vals = needed(load_jsonl(a.inp))
    print(f"{len(vals)} distinct values need an LLM ladder")
    batches = [vals[i:i + BATCH] for i in range(0, len(vals), BATCH)]
    res = {}
    for r in llm.pmap(run_batch, batches, workers=a.workers, desc="ladders"):
        res.update(r)
    for attempt in range(1, 7):  # retry failed values in ever smaller batches (a bad reply empties a whole batch)
        miss = [tuple(k.split("\t", 1)) for k, v in res.items() if not v]
        if not miss:
            break
        size = {1: 10, 2: 10, 3: 5, 4: 5}.get(attempt, 1)
        print(f"retry {attempt}: {len(miss)} values, batch size {size}")
        small = [miss[i:i + size] for i in range(0, len(miss), size)]
        for r in llm.pmap(lambda b: run_batch(b, f"retry{attempt}"), small, workers=a.workers, desc="ladders"):
            res.update({k: v for k, v in r.items() if v})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"ladders: {sum(bool(s) for s in res.values())}/{len(res)} non-empty; usage {llm.USAGE}")


if __name__ == "__main__":
    main()
