"""Rule-based post-filter of extracted QIs, following the TAB annotation guidelines (no LLM calls).

An extracted entity is re-labelled NO_MASK when it is a *generic* mention (no proper name, no number) of
  - a public institution acting in its official capacity (ORG): courts, ministries, authorities, police ...;
  - a legal procedure, remedy, benefit or legal instrument (MISC);
  - an official role (PERSON): judge, prosecutor, Secretary of State ...
Mentions that carry a proper name or a number (e.g. "Wodzislaw Slaski District Court", "Law no. 5683") are kept.
Rationale (measured on the TAB gold annotations of all 1,268 documents, see `python -m pipeline.qi_filter stats`):
TAB annotators mask generic institution mentions in only ~4% of cases, versus ~44% when a proper name is present.

usage:
  python -m pipeline.qi_filter stats                                  # evidence from TAB gold annotations
  python -m pipeline.qi_filter validate                               # P/R on TAB test before/after filtering
  python -m pipeline.qi_filter apply --inp data/qi/sage_qi_full.jsonl --out data/qi/sage_qi_full_filtered.jsonl
"""
import argparse
import gzip
import json
import os
import re
from collections import Counter, defaultdict

CATEGORIES = {
    "ORG": r"\b(court|courts|tribunal|cassation|prosecutor|prosecutors|ministry|minister|government|authority|"
           r"authorities|agency|revenue|council|commission|office|department|administration|parliament|ombudsman|"
           r"registry|service|police|gendarmerie|directorate|forces|army|board|inspectorate|bar)\b",
    "MISC": r"\b(proceedings?|benefits?|allowances?|pensions?|appeals?|claims?|compensation|remed(y|ies)|act|law|"
            r"code|article|legislation|order|decision|judgment|costs|damages|licen[cs]e|sentence|charges?|"
            r"indictment|investigation|application|complaint|petition|hearing|trial|regulations?|rules?|"
            r"provisions?|scheme|procedure|review|appeal)\b",
    "PERSON": r"\b(secretary of state|minister|judge|judges|prosecutor|president|governor|mayor|chief justice|"
              r"ombudsman|registrar|attorney general|government agent|agent|magistrate|inspector|officials?|"
              r"officers?|authorities)\b",
}
GENERIC = set("""the a an of and for on in at to by with under its his her their that this these those which
court courts tribunal cassation prosecutor prosecutors public ministry minister government authority authorities
agency council commission office department administration parliament ombudsman registry service state security
assize criminal civil administrative supreme constitutional high district regional appeal appeals first instance
second third chamber labour military national federal general higher lower county magistrates magistrate police
gendarmerie directorate forces army justice interior finance social health affairs inland revenue benefits board
committee division section branch unit headquarters station central local competent relevant domestic respondent
proceedings proceeding benefit allowance pension claim claims compensation remedy remedies act law code article
legislation order decision judgment costs damages licence license sentence charge charges indictment investigation
application complaint petition hearing trial regulation regulations rule rules provision provisions scheme
procedure review secretary president governor mayor chief judge judges registrar attorney agent inspector official
officials officer officers lord lady widow widows widowed bereavement survivors survivor child children family
legal aid tax income insurance welfare care disability housing employment unemployment immigration asylum
european human rights convention protocol""".split())

_TYPES = set(CATEGORIES)


def is_generic(text: str) -> bool:
    """True if the mention has no number and no capitalised word outside the generic vocabulary."""
    if re.search(r"\d", text):
        return False
    for w in re.findall(r"[A-Za-zÀ-ž]+", text):
        if w[0].isupper() and w.lower() not in GENERIC:
            return False
    return True


def non_identifying(etype: str, mentions: list) -> bool:
    pat = CATEGORIES.get(etype)
    if not pat:
        return False
    texts = [m if isinstance(m, str) else m["text"] for m in mentions]
    return all(re.search(pat, t, re.I) and is_generic(t) for t in texts)


def filter_entities(ents):
    """README-format entities -> same list with generic institutional/procedural mentions set to NO_MASK."""
    out = []
    for e in ents:
        if e["identifier_type"] == "QUASI" and non_identifying(e["entity_type"], e["mentions"]):
            e = {**e, "identifier_type": "NO_MASK", "filtered": True}
        out.append(e)
    return out


# ----------------------------------------------------------------- commands
def cmd_stats(_):
    from pipeline.evaluate import tab_docs
    tab = tab_docs(splits=("train", "dev", "test"))
    res = defaultdict(Counter)
    for d in tab.values():
        ann = d["annotations"][sorted(d["annotations"])[0]]
        for m in ann["entity_mentions"]:
            t = m["entity_type"]
            if t in _TYPES and re.search(CATEGORIES[t], m["span_text"], re.I):
                g = "generic" if is_generic(m["span_text"]) else "specific"
                res[f"{t}/{g}"][m["identifier_type"]] += 1
    out = {k: {"n": sum(v.values()), "masked_%": 100 * (v["QUASI"] + v["DIRECT"]) / sum(v.values()), **v}
           for k, v in sorted(res.items())}
    print(json.dumps(out, indent=1))
    return out


def cmd_validate(_):
    from pipeline.evaluate import tab_docs
    from pipeline.qi_extract import validate_doc
    tab = tab_docs(splits=("test",))
    preds = [json.loads(l) for l in gzip.open("data/qi/validation/test_predictions.jsonl.gz", "rt", encoding="utf-8")]
    out = {}
    for name, f in (("before", lambda e: e), ("after_filter", filter_entities)):
        rows = []
        for p in preds:
            ents = [e for e in f(p["entities"]) if e["identifier_type"] in ("DIRECT", "QUASI")]
            rows += validate_doc(tab[p["doc_id"]], ents)
        S = lambda k: sum(r[k] for r in rows)
        pr, rc = S("overlap_chars") / S("pred_chars"), S("overlap_chars") / S("gold_chars")
        out[name] = {"DIRECT_entity_recall": S("DIRECT_entity_recalled") / S("DIRECT_entities"),
                     "QUASI_entity_recall": S("QUASI_entity_recalled") / S("QUASI_entities"),
                     "char_precision": pr, "char_recall": rc, "char_f1": 2 * pr * rc / (pr + rc),
                     "predicted_entities": sum(len([e for e in f(p["entities"])
                                                    if e["identifier_type"] in ("DIRECT", "QUASI")]) for p in preds)}
    print(json.dumps(out, indent=1))
    return out


def cmd_apply(a):
    recs = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    n0 = n1 = 0
    by = Counter()
    for r in recs:
        n0 += sum(e["identifier_type"] == "QUASI" for e in r["entities"])
        r["entities"] = filter_entities(r["entities"])
        for e in r["entities"]:
            if e.get("filtered"):
                by[e["entity_type"]] += 1
        n1 += sum(e["identifier_type"] == "QUASI" for e in r["entities"])
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"QUASI entities {n0} -> {n1} (filtered {n0 - n1}: {dict(by)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "validate", "apply"])
    ap.add_argument("--inp")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {"stats": cmd_stats, "validate": cmd_validate, "apply": cmd_apply}[a.cmd](a)
    if a.cmd in ("stats", "validate"):
        os.makedirs("results/qi_filter", exist_ok=True)
        json.dump(res, open(f"results/qi_filter/{a.cmd}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
