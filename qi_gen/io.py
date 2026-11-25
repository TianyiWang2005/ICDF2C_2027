"""Loaders: TAB (original ECHR texts with gold annotations) and generic JSONL (e.g. SAGE rewrites + QI extractor)."""
import json
from collections import defaultdict


def load_tab(paths, annotator="first"):
    """Convert TAB standoff JSON into the common record format.

    An entity's identifier type is DIRECT if any mention is DIRECT, else QUASI if any mention is QUASI, else NO_MASK.
    annotator="first" uses the alphabetically first annotator of each document.
    """
    recs = []
    for p in paths:
        for d in json.load(open(p, encoding="utf-8")):
            ann = d["annotations"]
            name = sorted(ann)[0] if annotator == "first" else annotator
            ents = defaultdict(list)
            for m in ann[name]["entity_mentions"]:
                ents[m["entity_id"]].append(m)
            entities = []
            for eid, ms in ents.items():
                types = {m["identifier_type"] for m in ms}
                idt = "DIRECT" if "DIRECT" in types else "QUASI" if "QUASI" in types else "NO_MASK"
                entities.append({
                    "entity_id": eid,
                    "entity_type": ms[0]["entity_type"],
                    "identifier_type": idt,
                    "mentions": [{"start": m["start_offset"], "end": m["end_offset"], "text": m["span_text"]}
                                 for m in ms],
                })
            recs.append({"doc_id": d["doc_id"], "text": d["text"],
                         "respondent": d["meta"]["countries"].split(";")[0],
                         "split": d["dataset_type"], "entities": entities})
    return recs


def load_jsonl(path):
    """Generic input: one JSON record per line in the format documented in README (e.g. SAGE output + QI tags)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
