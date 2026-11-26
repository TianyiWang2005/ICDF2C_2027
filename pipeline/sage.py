"""SAGE re-implementation for ECHR (TAB dev+test).

usage: python -m pipeline.sage --tab_dir ../qi_gen_proj/tab --out data/sage/sage_echr.jsonl [--limit 5]

Stage 1: attribute extraction -> generation from attributes.
Stage 2: privacy agent evaluates (TRUE DATA vs GENERATED DATA); if not THISISSAFE, the rewriting agent applies the
suggestions; at most MAX_ROUNDS evaluations (official code: initiate_chat(max_turns=5)). If never safe, the last
rewrite is kept and safe=False (the official code does the same).
"""
import argparse
import json
import os
import re

from pipeline import llm
from pipeline import prompts as P

MAX_ROUNDS = 5
TEMP = 0.6  # official setting


def load_tab_split(tab_dir, splits=("dev", "test"), exclude=()):
    docs = []
    for s in splits:
        for d in json.load(open(os.path.join(tab_dir, f"echr_{s}.json"), encoding="utf-8")):
            if d["doc_id"] in exclude:
                continue
            docs.append({"doc_id": d["doc_id"], "split": s, "text": d["text"],
                         "respondent": d["meta"]["countries"].split(";")[0]})
    return docs


def _strip_generated(text):
    m = re.search(r"GENERATED DATA:\s*(.*)", text, re.S)
    return (m.group(1) if m else text).strip()


def run_doc(doc):
    attrs = llm.chat([{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": P.SAGE_ATTR.format(text=doc["text"])}],
                     tag="sage_attr", temperature=TEMP, max_tokens=2048)
    syn0 = llm.chat([{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": P.SAGE_SYNTH.format(attributes=attrs)}],
                    tag="sage_synth", temperature=TEMP, max_tokens=4096).strip()
    syn, history, safe = syn0, [], False
    # the evaluator keeps its own conversation (as in the autogen two-agent chat)
    ev_msgs = [{"role": "system", "content": P.SAGE_PRIVACY_AGENT},
               {"role": "user", "content": P.SAGE_FIRST_MSG.format(true=doc["text"], syn=syn)}]
    for rnd in range(1, MAX_ROUNDS + 1):
        fb = llm.chat(ev_msgs, tag="sage_eval", temperature=TEMP, max_tokens=2048)
        ev_msgs.append({"role": "assistant", "content": fb})
        if "thisissafe" in fb.lower():
            safe = True
            history.append({"round": rnd, "feedback": fb, "safe": True})
            break
        new = llm.chat([{"role": "system", "content": P.SAGE_REWRITER_AGENT},
                        {"role": "user", "content": P.SAGE_REWRITE_MSG.format(syn=syn, feedback=fb)}],
                       tag="sage_rewrite", temperature=TEMP, max_tokens=4096)
        syn = _strip_generated(new)
        history.append({"round": rnd, "feedback": fb, "safe": False, "rewrite": syn})
        ev_msgs.append({"role": "user", "content": P.SAGE_REEVAL_MSG.format(syn=syn)})
    return {**doc, "original_text": doc["text"], "text": syn, "attributes": attrs, "stage1_text": syn0,
            "safe": safe, "rounds": len(history), "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab_dir", required=True)
    ap.add_argument("--out", default="data/sage/sage_echr.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    a = ap.parse_args()
    from pipeline.qi_extract import few_shot  # the few-shot examples of the QI extractor are never rewritten
    _, shot_ids = few_shot(a.tab_dir)
    docs = load_tab_split(a.tab_dir, a.splits, exclude=set(shot_ids))[: a.limit]
    print(f"{len(docs)} documents (excluded few-shot examples {shot_ids})")
    res = llm.pmap(run_doc, docs, workers=a.workers, desc="sage")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(res)
    print(f"{n} docs, safe={sum(r['safe'] for r in res)}, mean rounds={sum(r['rounds'] for r in res) / n:.2f}, "
          f"len orig->syn {sum(len(r['original_text']) for r in res) // n} -> {sum(len(r['text']) for r in res) // n}")
    print("usage", llm.USAGE, f"est. cost ${llm.cost_usd():.3f}")


if __name__ == "__main__":
    main()
