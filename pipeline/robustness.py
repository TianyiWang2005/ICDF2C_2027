"""Robustness / subgroup analyses (CPU only, no LLM calls).

usage: python -m pipeline.robustness   -> results/sage/robustness.json

1. SAGE safe vs not-safe documents: pair risk before/after ours, residual rate of original QIs.
2. Safe-only corpus: rerun the main methods on the 222 documents SAGE marked safe.
3. Recall of the QI extractor ON SAGE TEXT, estimated from residual items: an original QI that survives in the SAGE
   text (residual judge = 1) and is no longer recoverable after "suppress all extracted QUASI" was tagged by the
   extractor; if it is still recoverable, the extractor missed it.
"""
import json
from collections import Counter

import qi_gen.core as C
from pipeline import ladders as L
from pipeline.evaluate import tab_docs
from pipeline.qi_extract import gold_entities
from qi_gen.io import load_jsonl
from run_experiments import measure

K0 = 5


def pair_risk_subset(docs, ids, k0=K0):
    df, pdf = C.doc_frequency(docs), C.pair_frequency(docs)
    n = r = 0
    for d in docs:
        if d.doc_id not in ids:
            continue
        ks = sorted(d.qkeys)
        kmin = min([df[k] for k in ks] + [pdf[(ks[a], ks[b])] for a in range(len(ks)) for b in range(a + 1, len(ks))]
                   + [10 ** 9])
        n += 1
        r += kmin < k0
    return 100 * r / n


def main():
    out = {}
    sage = {r["doc_id"]: r for r in map(json.loads, open("data/sage/sage_echr.jsonl", encoding="utf-8"))}
    safe = {i for i, r in sage.items() if r["safe"]}
    notsafe = set(sage) - safe
    L.load("data/generalisation/llm_ladders.json")
    recs = load_jsonl("data/qi/sage_qi.jsonl")
    tab = tab_docs()
    res = json.load(open("results/sage/residual_per_doc.json"))

    # 1. subgroups within the full corpus
    docs = C.build_docs(recs)
    sub = {}
    for name, run in [("sage", None), ("ours_o2", 2), ("ours_o3", 3)]:
        C.reset(docs)
        if run:
            C.run_riskgraph(docs, k0=K0, choice="rarest", order=run)
        sub[name] = {"safe": pair_risk_subset(docs, safe), "not_safe": pair_risk_subset(docs, notsafe)}
    resid = {}
    for corpus in ("sage_k5", "ours_o3_k5"):
        for g, ids in (("safe", safe), ("not_safe", notsafe)):
            items = [x for i in ids if i in res[corpus] for x in res[corpus][i]]
            resid.setdefault(corpus, {})[g] = 100 * sum(items) / len(items)
    out["subgroups_full_corpus"] = {"docs": {"safe": len(safe), "not_safe": len(notsafe)},
                                    f"pair_risk_k{K0}_%": sub, "residual_rate_%": resid}

    # 2. safe-only corpus
    docs2 = C.build_docs([r for r in recs if r["doc_id"] in safe])
    so = {}
    for name, order in [("sage", None), ("ours_o2", 2), ("ours_o3", 3)]:
        C.reset(docs2)
        if order:
            C.run_riskgraph(docs2, k0=K0, choice="rarest", order=order)
        m = measure(docs2, K0)
        so[name] = {"pair_%": m[f"docs_k<{K0}_%"], "triple_%": m[f"triple_k<{K0}_%"], "info_loss_%": m["info_loss_%"]}
    out["safe_only_corpus"] = so

    # 3. extractor recall on SAGE text via residual items
    s, sup = res["sage_k5"], res["suppress_k5"]
    n = caught = 0
    by, byc, missed = Counter(), Counter(), []
    for i in s:
        g = [(e["type"], max((m["text"] for m in e["mentions"]), key=len))
             for e in gold_entities(tab[i]) if e["identifier"] == "QUASI"]
        for j, x in enumerate(s[i]):
            if x:
                n += 1
                by[g[j][0]] += 1
                if not sup[i][j]:
                    caught += 1
                    byc[g[j][0]] += 1
                else:
                    missed.append(f"{i}: [{g[j][0]}] {g[j][1]}")
    out["extractor_recall_on_sage_text"] = {
        "residual_items": n, "tagged": caught, "recall_%": 100 * caught / n,
        "by_type": {t: {"n": by[t], "tagged": byc[t]} for t in sorted(by)}, "missed_examples": missed[:20]}
    json.dump(out, open("results/sage/robustness.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
