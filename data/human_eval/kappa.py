"""Cohen's kappa between two annotators and between each annotator and the LLM judge.
usage: python data/human_eval/kappa.py sheet_tianyi.csv sheet_colleague.csv"""
import csv, json, sys

def labels(p):
    with open(p, encoding="utf-8") as f:
        return {f"{r['doc_id']}#{r['item']}": int(list(r.values())[-1]) for r in csv.DictReader(f)}

def kappa(a, b):
    ks = sorted(set(a) & set(b)); n = len(ks)
    po = sum(a[k] == b[k] for k in ks) / n
    pa, pb = sum(a[k] for k in ks) / n, sum(b[k] for k in ks) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po, n

llm = json.load(open(__file__.replace("kappa.py", "llm_labels.json")))
anns = [labels(p) for p in sys.argv[1:]]
for i, a in enumerate(anns):
    print(f"annotator {i + 1} vs LLM: kappa=%.3f agreement=%.3f n=%d" % kappa(a, llm))
if len(anns) == 2:
    print("annotator 1 vs 2: kappa=%.3f agreement=%.3f n=%d" % kappa(*anns))
