"""LLM-based QI extraction (replaces the fine-tuned tagger) + validation against TAB gold annotations.

usage:
  # validate on the ORIGINAL TAB test texts (gold annotations available)
  python -m pipeline.qi_extract validate --tab_dir ../qi_gen_proj/tab --out data/qi/validation
  # tag the SAGE rewrites (input for the generalisation step, README JSONL format)
  python -m pipeline.qi_extract tag --inp data/sage/sage_echr.jsonl --out data/qi/sage_qi.jsonl

Few-shot examples come from the TAB train split only (no overlap with dev/test).
"""
import argparse
import json
import os
import re
from collections import defaultdict

from pipeline import llm
from pipeline import prompts as P

TYPES = {"PERSON", "CODE", "LOC", "ORG", "DEM", "DATETIME", "QUANTITY", "MISC"}
N_SHOT = 2


# ----------------------------------------------------------------- few-shot examples from train
def gold_entities(doc, annotator=None):
    """TAB doc -> list of {type, identifier, mentions:[{start,end,text}]} for DIRECT/QUASI entities."""
    ann = doc["annotations"]
    name = annotator or sorted(ann)[0]
    ents = defaultdict(list)
    for m in ann[name]["entity_mentions"]:
        ents[m["entity_id"]].append(m)
    out = []
    for eid, ms in ents.items():
        t = {m["identifier_type"] for m in ms}
        idt = "DIRECT" if "DIRECT" in t else "QUASI" if "QUASI" in t else None
        if idt:
            out.append({"entity_id": eid, "type": ms[0]["entity_type"], "identifier": idt,
                        "mentions": [{"start": m["start_offset"], "end": m["end_offset"], "text": m["span_text"]}
                                     for m in ms]})
    return out


def few_shot(tab_dir, n=N_SHOT):
    train = json.load(open(os.path.join(tab_dir, "echr_train.json"), encoding="utf-8"))
    train = sorted(train, key=lambda d: (len(d["text"]), d["doc_id"]))
    # shortest train docs that contain at least 8 QUASI entities (deterministic choice)
    picked = [d for d in train if 1500 < len(d["text"]) and
              sum(e["identifier"] == "QUASI" for e in gold_entities(d)) >= 8][:n]
    s = ""
    for d in picked:
        ents = [{"type": e["type"], "identifier": e["identifier"],
                 "mentions": list(dict.fromkeys(m["text"] for m in e["mentions"]))} for e in gold_entities(d)]
        s += P.QI_EXAMPLE.format(text=d["text"], output=json.dumps({"entities": ents}, ensure_ascii=False))
    return s, [d["doc_id"] for d in picked]


# ----------------------------------------------------------------- tagging + alignment
def align(text, entities):
    """Map mention strings to character offsets (all exact occurrences, then case-insensitive).
    Overlapping spans: longer span wins. Returns README-format entities."""
    cands = []
    for i, e in enumerate(entities):
        t = str(e.get("type", "MISC")).upper()
        t = t if t in TYPES else "MISC"
        idt = str(e.get("identifier", "QUASI")).upper()
        idt = "DIRECT" if idt.startswith("DIRECT") else "QUASI"
        for s in dict.fromkeys(m.strip() for m in e.get("mentions", []) if isinstance(m, str) and m.strip()):
            hits = [m.span() for m in re.finditer(re.escape(s), text)]
            if not hits:
                hits = [m.span() for m in re.finditer(re.escape(s), text, re.I)]
            for a, b in hits:
                cands.append((a, b, i, t, idt))
    cands.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    taken, kept = [], []
    for c in cands:
        if any(c[0] < b and a < c[1] for a, b in taken):
            continue
        taken.append((c[0], c[1]))
        kept.append(c)
    by = defaultdict(list)
    meta = {}
    for a, b, i, t, idt in sorted(kept):
        by[i].append({"start": a, "end": b, "text": text[a:b]})
        meta[i] = (t, idt)
    return [{"entity_id": f"e{i}", "entity_type": meta[i][0], "identifier_type": meta[i][1], "mentions": ms}
            for i, ms in sorted(by.items())]


def tag(text, examples):
    out = llm.chat([{"role": "system", "content": P.QI_SYSTEM},
                    {"role": "user", "content": P.QI_PROMPT.format(examples=examples, text=text)}],
                   tag="qi_extract", temperature=0.0, max_tokens=8192, json_mode=True)
    try:
        raw = llm.parse_json(out).get("entities", [])
    except Exception:
        raw = []
    return align(text, raw), raw


# ----------------------------------------------------------------- validation metrics (TAB-style)
def _covered(span, masks):
    a, b = span
    cov = sum(max(0, min(b, y) - max(a, x)) for x, y in masks)
    return cov / max(1, b - a)


def validate_doc(doc, pred):
    """Per-annotator: entity-level recall (an entity counts as protected if ALL its mentions are >=50% covered by
    predicted DIRECT/QUASI spans, as in TAB's ER), mention recall, and character-level precision against the union
    of DIRECT/QUASI spans of that annotator."""
    masks = [(m["start"], m["end"]) for e in pred for m in e["mentions"]]
    res = []
    for ann in sorted(doc["annotations"]):
        gold = gold_entities(doc, ann)
        g_spans = [(m["start"], m["end"]) for e in gold for m in e["mentions"]]
        r = {"annotator": ann}
        for idt in ("DIRECT", "QUASI"):
            es = [e for e in gold if e["identifier"] == idt]
            r[f"{idt}_entities"] = len(es)
            r[f"{idt}_entity_recalled"] = sum(all(_covered((m["start"], m["end"]), masks) >= 0.5
                                                   for m in e["mentions"]) for e in es)
            ms = [(m["start"], m["end"]) for e in es for m in e["mentions"]]
            r[f"{idt}_mentions"] = len(ms)
            r[f"{idt}_mention_recalled"] = sum(_covered(s, masks) >= 0.5 for s in ms)
        pchars = set(i for a, b in masks for i in range(a, b))
        gchars = set(i for a, b in g_spans for i in range(a, b))
        r["pred_chars"], r["gold_chars"], r["overlap_chars"] = len(pchars), len(gchars), len(pchars & gchars)
        per_type = defaultdict(lambda: [0, 0])
        for e in gold:
            if e["identifier"] == "QUASI":
                ok = all(_covered((m["start"], m["end"]), masks) >= 0.5 for m in e["mentions"])
                per_type[e["type"]][0] += 1
                per_type[e["type"]][1] += ok
        r["quasi_by_type"] = dict(per_type)
        res.append(r)
    return res


def cmd_validate(a):
    examples, shot_ids = few_shot(a.tab_dir, a.shots)
    test = json.load(open(os.path.join(a.tab_dir, "echr_test.json"), encoding="utf-8"))[: a.limit]
    outs = llm.pmap(lambda d: tag(d["text"], examples), test, workers=a.workers, desc="qi-validate")
    os.makedirs(a.out, exist_ok=True)
    rows, type_tot = [], defaultdict(lambda: [0, 0])
    with open(os.path.join(a.out, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
        for d, (pred, raw) in zip(test, outs):
            f.write(json.dumps({"doc_id": d["doc_id"], "entities": pred}, ensure_ascii=False) + "\n")
            for r in validate_doc(d, pred):
                for t, (n, k) in r.pop("quasi_by_type").items():
                    type_tot[t][0] += n
                    type_tot[t][1] += k
                rows.append({"doc_id": d["doc_id"], **r})
    S = lambda k: sum(r[k] for r in rows)
    summary = {
        "few_shot_train_docs": shot_ids, "test_docs": len(test), "doc_annotator_pairs": len(rows),
        "DIRECT_entity_recall": S("DIRECT_entity_recalled") / max(1, S("DIRECT_entities")),
        "QUASI_entity_recall": S("QUASI_entity_recalled") / max(1, S("QUASI_entities")),
        "DIRECT_mention_recall": S("DIRECT_mention_recalled") / max(1, S("DIRECT_mentions")),
        "QUASI_mention_recall": S("QUASI_mention_recalled") / max(1, S("QUASI_mentions")),
        "char_precision": S("overlap_chars") / max(1, S("pred_chars")),
        "char_recall": S("overlap_chars") / max(1, S("gold_chars")),
        "QUASI_entity_recall_by_type": {t: {"n": n, "recall": k / n} for t, (n, k) in sorted(type_tot.items())},
        "llm_usage": dict(llm.USAGE), "est_cost_usd": llm.cost_usd(),
    }
    p, r = summary["char_precision"], summary["char_recall"]
    summary["char_f1"] = 2 * p * r / (p + r) if p + r else 0
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(a.out, "per_doc_annotator.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


def cmd_tag(a):
    examples, _ = few_shot(a.tab_dir, a.shots)
    recs = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    outs = llm.pmap(lambda r: tag(r["text"], examples), recs, workers=a.workers, desc="qi-tag")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, (ents, raw) in zip(recs, outs):
            f.write(json.dumps({"doc_id": r["doc_id"], "split": r.get("split"), "respondent": r.get("respondent"),
                                "text": r["text"], "entities": ents}, ensure_ascii=False) + "\n")
    print(f"tagged {len(recs)} docs, {sum(len(o[0]) for o in outs)} entities; usage {llm.USAGE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "tag"])
    ap.add_argument("--tab_dir", default="../qi_gen_proj/tab")
    ap.add_argument("--inp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shots", type=int, default=N_SHOT, help="number of few-shot examples from TAB train")
    a = ap.parse_args()
    cmd_validate(a) if a.cmd == "validate" else cmd_tag(a)


if __name__ == "__main__":
    main()
