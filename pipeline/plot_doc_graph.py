"""Before/after risk graph of one document (paper figure).

usage: python -m pipeline.plot_doc_graph --doc 001-83927 --k0 5 --method ours_o2

Left: the SAGE output; right: after the method. Nodes = current QI values of the document (grey = raised to the
type placeholder, no longer a QI), edges = unsafe pairs (shared by < k0 documents of the corpus), edge width ~ 1/k.
Node colour = risk-weighted degree (own risk 1/df if df < k0, plus the weights of incident unsafe edges).
The central node is the document; its number is 1/k_min (k_min = size of the smallest group the document falls in).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from pipeline import ladders as L
from qi_gen.core import build_docs, doc_frequency, pair_frequency, risk_graph
from qi_gen.io import load_jsonl


def set_levels(docs, corpus_path):
    lv = {}
    for r in map(json.loads, open(corpus_path, encoding="utf-8")):
        for q in r["qi"]:
            lv[(r["doc_id"], q["entity_id"])] = q["level"]
    for d in docs:
        for e in d.entities:
            e.level = lv[(d.doc_id, e.entity_id)]


def panel(ax, docs, doc, k0, title):
    df, pdf = doc_frequency(docs), pair_frequency(docs)
    h = risk_graph(doc, df, pdf, k0)
    rwd = {n: h.nodes[n]["own"] + sum(w["weight"] for _, _, w in h.edges(n, data=True)) for n in h}
    kmin = min([df[k] for k in doc.qkeys] + [pdf[(a, b)] for a, b in h.edges] + [len(docs)])
    g = nx.Graph()
    case = ("CASE", doc.doc_id)
    g.add_node(case)
    for n in h:
        g.add_node(n)
    for a, b in h.edges:
        g.add_edge(a, b, kind="unsafe", w=h.edges[a, b]["weight"])
    placeholders = sorted({(e.etype, e.value) for e in doc.quasi if e.level == e.top})
    for p in placeholders:
        g.add_node(("PH",) + p)
    for n in list(g):
        if n != case:
            g.add_edge(case, n, kind="member", w=0)
    pos = nx.spring_layout(g, seed=3, k=1.2 / max(1, len(g)) ** 0.5, weight=None)
    pos[case] = (0, 0)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[e for e in g.edges if g.edges[e]["kind"] == "member"],
                           edge_color="#e5e7eb", width=.6)
    ue = [e for e in g.edges if g.edges[e]["kind"] == "unsafe"]
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=ue, edge_color="#ef4444", alpha=.55,
                           width=[0.6 + 3 * g.edges[e]["w"] for e in ue])
    qn = list(h.nodes)
    mx = max([rwd[n] for n in qn] + [1e-9])
    cmap = plt.get_cmap("YlOrRd")
    nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=qn, node_size=[260 + 500 * rwd[n] / mx for n in qn],
                           node_color=[cmap(0.15 + 0.85 * rwd[n] / mx) for n in qn], edgecolors="k", linewidths=.4)
    ph = [n for n in g if n[0] == "PH"]
    nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=ph, node_size=200, node_color="#d1d5db", edgecolors="#9ca3af",
                           linewidths=.4)
    nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=[case], node_size=2600, node_color="#111827")
    labels = {n: f"{n[1][:26]}\n[{n[0]}] df={df[n]}" for n in qn}
    labels.update({n: n[2] for n in ph})
    labels[case] = f"CASE\n{doc.doc_id}\n1/k_min={1 / kmin:.2f}"
    for n, t in labels.items():
        x, y = pos[n]
        ax.text(x, y, t, fontsize=6 if n != case else 8, ha="center", va="center",
                color="white" if n == case else "black")
    ax.set_title(f"{title}\n{len(ue)} unsafe pairs (k < {k0}), {len(qn)} QI values, "
                 f"{len(ph)} placeholders", fontsize=9)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="001-83927")
    ap.add_argument("--k0", type=int, default=5)
    ap.add_argument("--method", default="ours_o2")
    ap.add_argument("--inp", default="data/qi/sage_qi.jsonl")
    ap.add_argument("--ladders", default="data/generalisation/llm_ladders.json")
    a = ap.parse_args()
    L.load(a.ladders)
    docs = build_docs(load_jsonl(a.inp))
    doc = next(d for d in docs if d.doc_id == a.doc)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    set_levels(docs, f"data/corpora/sage_k{a.k0}.jsonl")
    panel(axes[0], docs, doc, a.k0, "SAGE output (before)")
    set_levels(docs, f"data/corpora/{a.method}_k{a.k0}.jsonl")
    panel(axes[1], docs, doc, a.k0, f"after {a.method} (k0={a.k0})")
    fig.suptitle("Risk graph: node = QI value (colour/size = risk-weighted degree), red edge = QI pair shared by "
                 f"fewer than k0={a.k0} documents of the corpus", fontsize=10)
    fig.tight_layout()
    out = f"results/figures/risk_graph_{a.doc}_{a.method}_k{a.k0}.png"
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
