"""Downstream evaluation of every saved corpus version.

usage:
  python -m pipeline.evaluate residual --corpora data/corpora --out results/sage/residual.csv
  python -m pipeline.evaluate utility  --corpora data/corpora --out results/sage/utility.csv

residual: for each document, the QUASI entities of the ORIGINAL TAB text (gold, first annotator) are listed and an
          LLM judge decides, per entity, whether its value can be recovered from the rewritten text.
          Residual rate = recoverable / all original QUASI entities.
utility:  SAGE-style RAG evaluation. QA pairs about the substance of each case are generated once from the
          ORIGINAL text; for each corpus version, BM25 retrieves the top-2 documents, the LLM answers from them,
          and an LLM judge scores the answer against the reference (1 / 0.5 / 0). Also reports retrieval hit@2
          (the source document of the question is among the retrieved ones).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd
from rank_bm25 import BM25Okapi

from pipeline import llm
from pipeline import paths as PT
from pipeline import prompts as P
from pipeline.qi_extract import gold_entities

TAB_DIR = "../qi_gen_proj/tab"


def tab_docs(tab_dir=TAB_DIR, splits=None):
    splits = splits or PT.SPLITS
    out = {}
    for s in splits:
        for d in json.load(open(os.path.join(tab_dir, f"echr_{s}.json"), encoding="utf-8")):
            out[d["doc_id"]] = d
    return out


def corpora(path, only=None):
    res = {}
    for p in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        if only and not any(re.fullmatch(o, name) for o in only):
            continue
        res[name] = {r["doc_id"]: r["text"] for r in map(json.loads, open(p, encoding="utf-8"))}
    return res


# ----------------------------------------------------------------- residual QI rate
def residual_doc(args):
    text, items = args
    lst = "\n".join(f"{i + 1}. [{t}] {v}" for i, (t, v) in enumerate(items))
    out = llm.chat([{"role": "system", "content": P.RESIDUAL_SYSTEM},
                    {"role": "user", "content": P.RESIDUAL_PROMPT.format(text=text, items=lst)}],
                   tag="residual", temperature=0.0, max_tokens=2048, json_mode=True)
    try:
        rec = llm.parse_json(out).get("recoverable", {})
    except Exception:
        rec = {}
    return [int(str(rec.get(str(i + 1), 0)).strip() in ("1", "1.0", "True", "true")) for i in range(len(items))]


def cmd_residual(a):
    tab = tab_docs()
    cs = corpora(a.corpora, a.only)
    ids = set().union(*[set(t) for t in cs.values()]) if cs else set(tab)
    cs = {"original": {i: tab[i]["text"] for i in ids}, **cs} if a.with_original else cs
    gold = {}
    for i, d in tab.items():
        ents = [e for e in gold_entities(d) if e["identifier"] == "QUASI"]
        gold[i] = [(e["type"], max((m["text"] for m in e["mentions"]), key=len)) for e in ents]
    rows, per_doc = [], {}
    for name, texts in cs.items():
        ids = [i for i in texts if gold.get(i)]
        res = llm.pmap(residual_doc, [(texts[i], gold[i]) for i in ids], workers=a.workers, desc=f"residual {name}")
        tot = sum(len(r) for r in res)
        by_t = defaultdict(lambda: [0, 0])
        for i, r in zip(ids, res):
            for (t, _), x in zip(gold[i], r):
                by_t[t][0] += 1
                by_t[t][1] += x
        per_doc[name] = {i: r for i, r in zip(ids, res)}
        rows.append({"corpus": name, "docs": len(ids), "original_quasi": tot,
                     "residual_rate_%": 100 * sum(map(sum, res)) / max(1, tot),
                     "docs_with_any_residual_%": 100 * sum(any(r) for r in res) / max(1, len(ids)),
                     **{f"residual_{t}_%": 100 * k / n for t, (n, k) in sorted(by_t.items())}})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    json.dump(per_doc, open(a.out.replace(".csv", "_per_doc.json"), "w"))
    print("usage", llm.USAGE, f"${llm.cost_usd():.2f}")


# ----------------------------------------------------------------- utility (RAG QA)
_tok = re.compile(r"[a-z0-9]+")


def toks(s):
    return _tok.findall(s.lower())


def gen_qa(tab, n=2, path=None, ids=None):
    path = path or PT.QA
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))

    def one(i):
        out = llm.chat([{"role": "user", "content": P.QA_GEN_PROMPT.format(n=n, text=tab[i]["text"])}],
                       tag="qa_gen", temperature=0.0, max_tokens=1024, json_mode=True)
        try:
            return [{"doc_id": i, **q} for q in llm.parse_json(out).get("qa", [])[:n]]
        except Exception:
            return []

    qa = [q for r in llm.pmap(one, sorted(ids or tab), workers=16, desc="qa-gen") for q in r]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(qa, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return qa


def cmd_utility(a):
    tab = tab_docs()
    cs = corpora(a.corpora, a.only)
    ids = sorted(set().union(*[set(t) for t in cs.values()])) if cs else sorted(tab)
    qa = gen_qa(tab, ids=ids)
    cs = {"original": {i: tab[i]["text"] for i in ids}, **cs}
    rows, details = [], {}
    for name, texts in cs.items():
        ids = sorted(texts)
        bm = BM25Okapi([toks(texts[i]) for i in ids])

        def one(q):
            sc = bm.get_scores(toks(q["question"]))
            top = sorted(range(len(ids)), key=lambda j: -sc[j])[:2]
            ctx = "\n\n---\n\n".join(texts[ids[j]] for j in top)
            ans = llm.chat([{"role": "user", "content": P.QA_ANSWER_PROMPT.format(context=ctx,
                                                                                question=q["question"])}],
                           tag="qa_answer", temperature=0.0, max_tokens=256)
            j = llm.chat([{"role": "user", "content": P.QA_JUDGE_PROMPT.format(ref=q["answer"], cand=ans,
                                                                             question=q["question"])}],
                         tag="qa_judge", temperature=0.0, max_tokens=64, json_mode=True)
            try:
                s = float(llm.parse_json(j).get("score", 0))
            except Exception:
                s = 0.0
            return {"hit": q["doc_id"] in {ids[k] for k in top}, "score": s, "answer": ans}

        res = llm.pmap(one, qa, workers=a.workers, desc=f"utility {name}")
        details[name] = res
        rows.append({"corpus": name, "questions": len(res),
                     "qa_score_%": 100 * sum(r["score"] for r in res) / len(res),
                     "retrieval_hit@2_%": 100 * sum(r["hit"] for r in res) / len(res)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    json.dump(details, open(a.out.replace(".csv", "_answers.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("usage", llm.USAGE, f"${llm.cost_usd():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["residual", "utility"])
    ap.add_argument("--corpora", default=PT.CORPORA)
    ap.add_argument("--only", nargs="*", help="regexes on corpus names, e.g. 'sage_k5' 'ours_o.*'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with_original", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cmd_residual(a) if a.cmd == "residual" else cmd_utility(a)


if __name__ == "__main__":
    main()
