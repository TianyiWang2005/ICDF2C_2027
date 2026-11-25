"""Pilot experiments on TAB (or on any JSONL input in the README format).

usage:
  python run_experiments.py --tab_dir path/to/text-anonymization-benchmark --out results
  python run_experiments.py --jsonl sage_with_qi.jsonl --out results_sage
"""
import argparse
import glob
import json
import os
import time

import pandas as pd

from qi_gen.core import (build_docs, evaluate, evaluate_triples, render, reset, run_method, run_random,
                         run_riskgraph, suppress_all, total_steps)
from qi_gen.io import load_jsonl, load_tab


def measure(docs, k):
    r = evaluate(docs, k_eval=k)
    r.update(evaluate_triples(docs, k_eval=k))
    r["gen_steps"] = total_steps(docs)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab_dir")
    ap.add_argument("--jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--k0", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = load_tab(sorted(glob.glob(os.path.join(a.tab_dir, "echr_*.json")))) if a.tab_dir else load_jsonl(a.jsonl)
    docs = build_docs(recs)
    print(f"{len(docs)} documents, {sum(len(d.quasi) for d in docs)} QUASI entities")

    rows, logs = [], {}

    def add(name, k0, extra=None):
        r = {"method": name, "k0": k0, **measure(docs, k0), **(extra or {})}
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "results.csv"), index=False)

    for k0 in a.k0:
        reset(docs); add("DIRECT removal only", k0)
        reset(docs); suppress_all(docs); add("Suppress all QUASI", k0)
        reset(docs); run_method(docs, k0=k0); add("Rarity only (single-value k-anon.)", k0)
        for choice, name in [("random", "Risk graph + random node"), ("rarest", "Risk graph + rarest-first (ours)"),
                             ("degree", "Risk graph + weighted degree"),
                             ("cost", "Risk graph + cost-aware degree")]:
            reset(docs); log = []; t = time.time()
            run_riskgraph(docs, k0=k0, choice=choice, log=log)
            logs[f"{choice}_k{k0}"] = log
            add(name, k0, {"rounds": len(log), "seconds": round(time.time() - t, 1)})
            if choice == "rarest":
                budget = total_steps(docs)
                with open(os.path.join(a.out, f"generalised_k{k0}.jsonl"), "w", encoding="utf-8") as f:
                    for d in docs:
                        f.write(json.dumps({"doc_id": d.doc_id, "text": render(d), "qi": [
                            {"entity_id": e.entity_id, "type": e.etype, "original": e.ladder[0], "value": e.value,
                             "level": e.level} for e in d.quasi]}, ensure_ascii=False) + "\n")
        res = []
        for s in range(a.seeds):
            reset(docs); run_random(docs, budget, seed=s); res.append(measure(docs, k0))
        rows.append({"method": "Random, same budget as ours", "k0": k0,
                     **pd.DataFrame(res).mean().to_dict()})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "results.csv"), index=False)
    json.dump(logs, open(os.path.join(a.out, "round_logs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
