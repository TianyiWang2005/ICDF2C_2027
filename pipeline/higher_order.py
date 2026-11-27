"""Residual risk beyond the checked order (no LLM calls).

usage: QI_RUN=full python -m pipeline.higher_order  -> results/full/higher_order_risk.json

For each corpus version, the share of documents having at least one combination of m current QI values
(m = 2..5) shared by fewer than k0 documents. Combinations are sampled (400 per document), so values for
m >= 3 are lower bounds. Shows that "0% risk" holds only up to the order the method checks.
"""
import json
import random
from collections import defaultdict

import qi_gen.core as C
from pipeline import ladders as L
from pipeline import paths as PT
from pipeline.plot_doc_graph import set_levels
from qi_gen.io import load_jsonl

K0, PER_DOC = 5, 400


def risk(docs, order, k0=K0, seed=0):
    rng = random.Random(seed)
    C.doc_frequency(docs)
    inv, ks = defaultdict(set), []
    for i, d in enumerate(docs):
        k = sorted(d.qkeys)
        ks.append(k)
        for x in k:
            inv[x].add(i)
    bad = 0
    for k in ks:
        if len(k) < order:
            continue
        combos = {tuple(sorted(rng.sample(range(len(k)), order))) for _ in range(PER_DOC)}
        bad += any(len(set.intersection(*[inv[k[j]] for j in c])) < k0 for c in combos)
    return 100 * bad / len(docs)


def main():
    L.load(PT.LADDERS)
    docs = C.build_docs(load_jsonl(PT.QI))
    out = {}
    for name in (f"sage_k{K0}", f"ours_o2_k{K0}", f"ours_o3_k{K0}"):
        set_levels(docs, f"{PT.CORPORA}/{name}.jsonl")
        out[name] = {f"order_{m}_%": risk(docs, m) for m in (2, 3, 4, 5)}
        print(name, out[name])
    json.dump(out, open(f"{PT.RESULTS}/higher_order_risk.json", "w"), indent=1)


if __name__ == "__main__":
    main()
