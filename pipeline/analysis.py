"""Descriptive analyses for the paper: QI statistics, hidden-QI evidence, per-type generalisation, SAGE safe vs
unsafe documents, and before/after text examples.

usage: python -m pipeline.analysis   (needs data/corpora from run_main)
"""
import json
from collections import Counter, defaultdict

from pipeline.qi_extract import gold_entities
from pipeline.evaluate import tab_docs
from pipeline import paths as PT

K0 = 5


def load(name):
    return {r["doc_id"]: r for r in map(json.loads, open(f"{PT.CORPORA}/{name}.jsonl", encoding="utf-8"))}


def main():
    out = {}
    tab = tab_docs(splits=PT.SPLITS)
    sage = {r["doc_id"]: r for r in map(json.loads, open(PT.SAGE, encoding="utf-8"))}
    base, ours2, ours3 = load(f"sage_k{K0}"), load(f"ours_o2_k{K0}"), load(f"ours_o3_k{K0}")

    # 1. QIs before / after SAGE
    gold_q = Counter(e["type"] for d in tab.values() for e in gold_entities(d) if e["identifier"] == "QUASI")
    sage_q = Counter(q["type"] for r in base.values() for q in r["qi"] if q["identifier"] == "QUASI")
    out["quasi_entities_original_gold"] = {"total": sum(gold_q.values()), "per_doc": sum(gold_q.values()) / len(tab),
                                           "by_type": dict(gold_q.most_common())}
    out["quasi_entities_sage_llm"] = {"total": sum(sage_q.values()), "per_doc": sum(sage_q.values()) / len(base),
                                      "by_type": dict(sage_q.most_common())}

    # 2. most frequent / rarest QI values in the SAGE corpus (what SAGE writes instead of the originals)
    df = Counter()
    for r in base.values():
        df.update({(q["type"], q["original"]) for q in r["qi"] if q["identifier"] == "QUASI"})
    out["sage_qi_values_distinct"] = len(df)
    out["sage_qi_values_df1_%"] = 100 * sum(c == 1 for c in df.values()) / len(df)
    out["sage_qi_values_most_common"] = [f"{t}: {v} ({c})" for (t, v), c in df.most_common(25)]

    # 3. per-type generalisation of ours (order 3)
    for name, corp in (("ours_o2", ours2), ("ours_o3", ours3)):
        by_t = defaultdict(lambda: [0, 0.0, 0, 0])
        for r in corp.values():
            for q in r["qi"]:
                if q["identifier"] != "QUASI":
                    continue
                top = len(q["ladder"]) - 1
                b = by_t[q["type"]]
                b[0] += 1
                b[1] += q["level"] / top
                b[2] += q["level"] == 0
                b[3] += q["level"] == top
        out[f"{name}_k{K0}_by_type"] = {t: {"n": n, "info_loss_%": 100 * l / n, "unchanged_%": 100 * u / n,
                                           "suppressed_%": 100 * s / n} for t, (n, l, u, s) in sorted(by_t.items())}

    # 4. SAGE safe vs not safe
    res = json.load(open(f"{PT.RESULTS}/residual_per_doc.json")) if __import__("os").path.exists(
        f"{PT.RESULTS}/residual_per_doc.json") else {}
    r5 = res.get(f"sage_k{K0}", {})
    grp = defaultdict(list)
    for i, r in sage.items():
        grp["safe" if r["safe"] else "not_safe"].append(i)
    out["sage_safe_vs_not"] = {
        g: {"docs": len(ids), "mean_rounds": sum(sage[i]["rounds"] for i in ids) / len(ids),
            "mean_quasi_sage": sum(sum(q["identifier"] == "QUASI" for q in base[i]["qi"]) for i in ids) / len(ids),
            "residual_rate_%": (100 * sum(sum(r5[i]) for i in ids if i in r5) /
                                max(1, sum(len(r5[i]) for i in ids if i in r5))) if r5 else None}
        for g, ids in grp.items()}

    # 5. examples: sentences whose QIs changed (first 3 docs with >= 4 changed QIs)
    ex = []
    for i in sorted(base):
        ch = [q for q in ours3[i]["qi"] if q["identifier"] == "QUASI" and q["level"] > 0]
        if len(ch) >= 4:
            ex.append({"doc_id": i, "sage_excerpt": base[i]["text"][:700], "ours_o3_excerpt": ours3[i]["text"][:700],
                       "changes": [f"{q['type']}: {q['original']} -> {q['value']}" for q in ch[:12]]})
        if len(ex) == 3:
            break
    out["examples"] = ex
    json.dump(out, open(f"{PT.RESULTS}/analysis.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
