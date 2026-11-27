"""Diagnostics of the TAB-200 utility evaluation (no LLM calls).

usage: QI_RUN=full python -m pipeline.utility_diagnostics  -> results/full/utility_tab200_diagnostics.json

1. paired bootstrap of every corpus vs SAGE (95% CI);
2. retrieval vs answering: hit rate, score when the source document is / is not retrieved;
3. per task type;
4. QI-dependence defined with the TAB gold annotations: a question is QI-dependent if its question, context or
   reference answer contains any gold-annotated mention (>= 4 chars) of its document;
5. which generalised QI values relate to the questions SAGE answered and ours (order 3, k0=5) did not.
"""
import json
import random
import re
from collections import Counter, defaultdict

from pipeline import paths as PT
from pipeline.evaluate import load_tab200, tab_docs
from pipeline.qi_extract import gold_entities

CORPORA = ["original", "sage_k5", "ours_o2_k5", "ours_o3_k5", "suppress_k5"]


def main():
    A = json.load(open(f"{PT.RESULTS}/utility_tab200_answers.json", encoding="utf-8"))
    Q = load_tab200()
    raw = {r["question_id"]: r for r in map(json.loads, open("data/utility/tab200/questions.jsonl", encoding="utf-8"))}
    tab = tab_docs(splits=("train", "dev", "test"))
    rng = random.Random(0)
    n = len(Q)

    def boot(a, b, idx, B=3000):
        d = [A[a][i]["score"] - A[b][i]["score"] for i in idx]
        m = len(d)
        bs = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(B))
        return {"diff": 100 * sum(d) / m, "ci95": [100 * bs[int(.025 * B)], 100 * bs[int(.975 * B)]], "n": m}

    def qtext(q):
        r = raw[q["question_id"]]
        return (r["question"] + " " + r["case_context"] + " " + q["answer"]).lower()

    def gold_dep(q):
        s = qtext(q)
        vals = {m["text"].strip().lower() for e in gold_entities(tab[q["doc_id"]]) for m in e["mentions"]}
        return any(len(v) >= 4 and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", s) for v in vals)

    out = {}
    allidx = list(range(n))
    out["vs_sage_all"] = {c: boot(c, "sage_k5", allidx) for c in CORPORA if c != "sage_k5"}
    out["ours_o3_vs_suppress"] = boot("ours_o3_k5", "suppress_k5", allidx)
    out["retrieval_vs_answering"] = {}
    for c in CORPORA:
        h = [r for r in A[c] if r["hit"]]
        m = [r for r in A[c] if not r["hit"]]
        out["retrieval_vs_answering"][c] = {"hit_%": 100 * len(h) / n,
                                            "score_when_hit_%": 100 * sum(r["score"] for r in h) / len(h),
                                            "score_when_missed_%": 100 * sum(r["score"] for r in m) / max(1, len(m))}
    by = defaultdict(list)
    for i, q in enumerate(Q):
        by[q["task_type"]].append(i)
    out["by_task_type"] = {t: {c: 100 * sum(A[c][i]["score"] for i in idx) / len(idx) for c in CORPORA} | {"n": len(idx)}
                           for t, idx in sorted(by.items())}
    dep = [gold_dep(q) for q in Q]
    out["gold_qi_dependence"] = {}
    for flag, lab in ((False, "qi_free"), (True, "qi_dependent")):
        idx = [i for i, d in enumerate(dep) if d == flag]
        out["gold_qi_dependence"][lab] = {
            "n": len(idx), "score_%": {c: 100 * sum(A[c][i]["score"] for i in idx) / len(idx) for c in CORPORA},
            "vs_sage": {c: boot(c, "sage_k5", idx) for c in CORPORA if c != "sage_k5"}}
    corp = {r["doc_id"]: r for r in map(json.loads, open(f"{PT.CORPORA}/ours_o3_k5.jsonl", encoding="utf-8"))}
    cnt, ex, nq = Counter(), defaultdict(list), 0
    stop = {"that", "this", "with", "from", "state", "court"}
    for i, q in enumerate(Q):
        if not (A["sage_k5"][i]["score"] >= 0.5 and A["ours_o3_k5"][i]["score"] == 0):
            continue
        nq += 1
        text, seen = qtext(q), set()
        for e in corp[q["doc_id"]]["qi"]:
            if e["identifier"] != "QUASI" or e["level"] == 0:
                continue
            words = [w for w in re.findall(r"[a-z]{4,}", e["original"]) if w not in stop]
            if words and any(w in text for w in words) and (e["type"], e["original"]) not in seen:
                seen.add((e["type"], e["original"]))
                cnt[e["type"]] += 1
                if len(ex[e["type"]]) < 8:
                    ex[e["type"]].append(f"{e['original']} -> {e['value']}")
    out["lost_answers"] = {"questions_sage_right_ours_wrong": nq, "related_generalised_qi_by_type": dict(cnt.most_common()),
                           "examples": ex}
    json.dump(out, open(f"{PT.RESULTS}/utility_tab200_diagnostics.json", "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(json.dumps({k: out[k] for k in ("vs_sage_all", "ours_o3_vs_suppress")}, indent=1))


if __name__ == "__main__":
    main()
