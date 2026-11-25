"""Main experiment on SAGE rewrites: baselines, ablations, ours (order 2 and 3), k0 sweep.

usage: python -m pipeline.run_main --inp data/qi/sage_qi.jsonl --ladders data/generalisation/llm_ladders.json \
           --out results/sage

Writes
  results/sage/results.csv                    one row per (method, k0)
  results/sage/round_logs.json                values raised per round
  data/corpora/<method>_k<k0>.jsonl           rewritten texts (input of residual / utility evaluation)
  data/graphs/risk_graphs_k<k0>.json          per-document risk graph before generalisation
  data/graphs/cooccurrence_graphs.json        per-document QI co-occurrence graph (sentence level)
"""
import argparse
import json
import os
import time

import pandas as pd

from pipeline import ladders as L
from qi_gen.core import (build_docs, doc_frequency, pair_frequency, render, reset, risk_graph, run_method,
                         run_random, run_riskgraph, suppress_all, total_steps)
from qi_gen.io import load_jsonl
from run_experiments import measure

# corpora saved for downstream evaluation (method key -> saved?)
SAVE = {"sage", "suppress", "rarity", "ours_o2", "ours_o3", "random_budget_o2", "random_budget_o3"}


def dump_corpus(docs, path, recs_by_id):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"doc_id": d.doc_id, "split": recs_by_id[d.doc_id].get("split"), "text": render(d),
                                "qi": [{"entity_id": e.entity_id, "type": e.etype, "identifier": e.idtype,
                                        "original": e.ladder[0], "value": e.value, "level": e.level,
                                        "ladder": e.ladder} for e in d.entities]},
                               ensure_ascii=False) + "\n")


def dump_graphs(docs, k0, out_dir):
    reset(docs)
    df, pdf = doc_frequency(docs), pair_frequency(docs)
    g_all = {}
    for d in docs:
        h = risk_graph(d, df, pdf, k0)
        g_all[d.doc_id] = {
            "nodes": [{"type": t, "value": v, "df": df[(t, v)], "own_risk": h.nodes[(t, v)]["own"],
                       "risk_weighted_degree": h.nodes[(t, v)]["own"] + sum(
                           w["weight"] for _, _, w in h.edges((t, v), data=True))} for t, v in h.nodes],
            "unsafe_edges": [{"a": list(a), "b": list(b), "k": round(1 / w["weight"]), "weight": w["weight"]}
                             for a, b, w in h.edges(data=True)]}
    json.dump(g_all, open(os.path.join(out_dir, f"risk_graphs_k{k0}.json"), "w", encoding="utf-8"),
              ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/qi/sage_qi.jsonl")
    ap.add_argument("--ladders", default="data/generalisation/llm_ladders.json")
    ap.add_argument("--out", default="results/sage")
    ap.add_argument("--corpora", default="data/corpora")
    ap.add_argument("--graphs", default="data/graphs")
    ap.add_argument("--k0", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    for p in (a.out, a.corpora, a.graphs):
        os.makedirs(p, exist_ok=True)
    if a.ladders and os.path.exists(a.ladders):
        print(f"{len(L.load(a.ladders))} LLM ladders loaded")
    recs = load_jsonl(a.inp)
    by_id = {r["doc_id"]: r for r in recs}
    docs = build_docs(recs)
    nq = sum(len(d.quasi) for d in docs)
    print(f"{len(docs)} documents, {nq} QUASI entities, "
          f"{sum(len(d.entities) - len(d.quasi) for d in docs)} DIRECT entities")

    json.dump({d.doc_id: {"nodes": [{"id": e.entity_id, "type": e.etype, "value": e.ladder[0]} for e in d.quasi],
                          "edges": [{"a": u, "b": v, "sentences": w["weight"]}
                                    for u, v, w in d.graph.edges(data=True)]} for d in docs},
              open(os.path.join(a.graphs, "cooccurrence_graphs.json"), "w", encoding="utf-8"), ensure_ascii=False)

    rows, logs = [], {}

    def add(key, name, k0, extra=None):
        r = {"method": name, "key": key, "k0": k0, **measure(docs, k0), **(extra or {})}
        rows.append(r)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "results.csv"), index=False)
        if key in SAVE:
            dump_corpus(docs, os.path.join(a.corpora, f"{key}_k{k0}.jsonl"), by_id)

    for k0 in a.k0:
        dump_graphs(docs, k0, a.graphs)
        reset(docs); add("sage", "SAGE (no generalisation)", k0)
        reset(docs); suppress_all(docs); add("suppress", "SAGE + suppress all QUASI", k0)
        reset(docs); run_method(docs, k0=k0); add("rarity", "SAGE + rarity only (single-value k-anon.)", k0)
        budgets = {}
        for choice, order, key, name in [
                ("random", 2, "rg_random_o2", "SAGE + risk graph, random node"),
                ("rarest", 2, "rg_rarest_o2", "SAGE + risk graph, rarest node"),
                ("betweenness", 2, "rg_betweenness_o2", "SAGE + risk graph, betweenness"),
                ("degree", 2, "ours_o2", "SAGE + risk graph, weighted degree (ours, order 2)"),
                ("rarest", 3, "rg_rarest_o3", "SAGE + risk graph, rarest node (order 3)"),
                ("degree", 3, "ours_o3", "SAGE + risk graph, weighted degree (ours, order 3)")]:
            reset(docs); log = []; t = time.time()
            run_riskgraph(docs, k0=k0, choice=choice, order=order, log=log)
            logs[f"{key}_k{k0}"] = log
            budgets[key] = total_steps(docs)
            add(key, name, k0, {"rounds": len(log), "seconds": round(time.time() - t, 1)})
        for o in (2, 3):
            res = []
            for s in range(a.seeds):
                reset(docs); run_random(docs, budgets[f"ours_o{o}"], seed=s); res.append(measure(docs, k0))
                if s == 0:
                    dump_corpus(docs, os.path.join(a.corpora, f"random_budget_o{o}_k{k0}.jsonl"), by_id)
            rows.append({"method": f"SAGE + random, same budget as ours (order {o})", "key": f"random_budget_o{o}",
                         "k0": k0, **pd.DataFrame(res).mean().to_dict()})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "results.csv"), index=False)
    json.dump(logs, open(os.path.join(a.out, "round_logs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
