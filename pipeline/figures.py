"""Figures and summary tables for the paper.

usage: python -m pipeline.figures
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

OUT = "results/figures"
COL = {"ours_o2": "#1f6feb", "ours_o3": "#0b3d91", "rg_degree_o2": "#d97706", "rg_degree_o3": "#92400e",
       "rg_cost_o2": "#f59e0b", "rg_cost_o3": "#b45309", "rg_lookahead_o2": "#eab308",
       "rg_random_o2": "#6b7280", "rg_betweenness_o2": "#7c3aed", "rarity": "#16a34a",
       "random_budget_o2": "#dc2626", "random_budget_o3": "#991b1b", "suppress": "#000000", "sage": "#9ca3af"}
LABEL = {"sage": "SAGE", "suppress": "Suppress all QI", "rarity": "Rarity only (single values)",
         "rg_random_o2": "RG + random node", "rg_degree_o2": "RG + weighted degree", "rg_cost_o2": "RG + cost-aware",
         "rg_lookahead_o2": "RG + look-ahead", "rg_betweenness_o2": "RG + betweenness",
         "rg_degree_o3": "RG + weighted degree (o3)", "rg_cost_o3": "RG + cost-aware (o3)",
         "ours_o2": "Ours: RG + rarest-first (o2)", "ours_o3": "Ours: RG + rarest-first (o3)",
         "random_budget_o2": "Random, same budget (o2)", "random_budget_o3": "Random, same budget (o3)"}


def tradeoff(res):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], "docs_k<{k}_%", "Docs with a pair shared by < k0 docs (%)"),
                              (axes[1], "triple_k<{k}_%", "Docs with a triple shared by < k0 docs (%)")]:
        keep = ["rarity", "rg_random_o2", "rg_degree_o2", "rg_degree_o3", "ours_o2", "ours_o3",
                "random_budget_o2", "random_budget_o3"]
        for key, g in res[res.key.isin(keep)].groupby("key"):
            ys = [g[g.k0 == k][metric.format(k=k)].values[0] for k in g.k0]
            ax.plot(g["info_loss_%"], ys, "o-", color=COL.get(key, "k"), label=LABEL.get(key, key), ms=4)
            for x, y, k in zip(g["info_loss_%"], ys, g.k0):
                if key in ("ours_o2", "ours_o3"):
                    ax.annotate(f"k0={k}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Information loss (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.3)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/privacy_utility_tradeoff.png", dpi=200)
    plt.close(fig)


def privacy_utility_scatter(res, util, resid):
    """x = information loss, y = RAG QA score; marker label = residual rate of original QIs."""
    m = util.merge(resid[["corpus", "residual_rate_%"]], on="corpus", how="left")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in m.iterrows():
        name = r["corpus"]
        if name == "original":
            x = 0.0
        else:
            key, k0 = name.rsplit("_k", 1)
            row = res[(res.key == key) & (res.k0 == int(k0))]
            x = float(row["info_loss_%"].values[0]) if len(row) else float("nan")
            name = key
        ax.scatter(x, r["qa_score_%"], color=COL.get(name, "#111827"), s=40)
        ax.annotate(r["corpus"], (x, r["qa_score_%"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Information loss of QIs (%)")
    ax.set_ylabel("RAG QA score (%)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/utility_vs_loss.png", dpi=200)
    plt.close(fig)


def utility_plot(util):
    if util is None:
        return
    u = util.copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(u["corpus"], u["qa_score_%"], color="#1f6feb")
    ax.set_xlabel("RAG QA score (%)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/utility_qa.png", dpi=200)
    plt.close(fig)


def risk_graph_example(k0=5):
    p = f"data/graphs/risk_graphs_k{k0}.json"
    if not os.path.exists(p):
        return
    g_all = json.load(open(p, encoding="utf-8"))
    # a mid-sized example: the document whose unsafe-edge count is closest to 25
    doc_id = min(g_all, key=lambda d: (abs(len(g_all[d]["unsafe_edges"]) - 25), d))
    g = g_all[doc_id]
    h = nx.Graph()
    for n in g["nodes"]:
        h.add_node((n["type"], n["value"]), **n)
    for e in g["unsafe_edges"]:
        h.add_edge(tuple(e["a"]), tuple(e["b"]), k=e["k"])
    h.remove_nodes_from([n for n in list(h) if h.degree(n) == 0 and not h.nodes[n]["own_risk"]])
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(h, seed=1, k=0.9)
    deg = [h.nodes[n]["risk_weighted_degree"] for n in h]
    nx.draw_networkx_edges(h, pos, ax=ax, alpha=.4, width=[1.5 / h.edges[e]["k"] + .3 for e in h.edges])
    nx.draw_networkx_nodes(h, pos, ax=ax, node_size=[120 + 300 * d for d in deg], node_color=deg, cmap="Reds",
                           edgecolors="k", linewidths=.5)
    nx.draw_networkx_labels(h, pos, ax=ax, font_size=6,
                            labels={n: f"{n[1][:28]}\n[{n[0]}] df={h.nodes[n]['df']}" for n in h})
    ax.set_title(f"Risk graph of {doc_id} (k0={k0}); node size/colour = risk-weighted degree", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/risk_graph_example_{doc_id}.png", dpi=200)
    plt.close(fig)


def sage_stats():
    p = "data/sage/sage_echr.jsonl"
    if not os.path.exists(p):
        return
    rs = [json.loads(l) for l in open(p, encoding="utf-8")]
    s = {"docs": len(rs), "safe": sum(r["safe"] for r in rs),
         "not_safe_after_max_rounds": sum(not r["safe"] for r in rs),
         "mean_rounds": sum(r["rounds"] for r in rs) / len(rs),
         "mean_chars_original": sum(len(r["original_text"]) for r in rs) / len(rs),
         "mean_chars_sage": sum(len(r["text"]) for r in rs) / len(rs),
         "docs_longer_after_sage": sum(len(r["text"]) > len(r["original_text"]) for r in rs)}
    json.dump(s, open("results/sage/sage_stats.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter([len(r["original_text"]) for r in rs], [len(r["text"]) for r in rs], s=8,
               c=["#16a34a" if r["safe"] else "#dc2626" for r in rs])
    m = max(len(r["original_text"]) for r in rs)
    ax.plot([0, m], [0, m], "k--", lw=.8)
    ax.set_xlabel("original length (chars)")
    ax.set_ylabel("SAGE length (chars)")
    ax.set_title("SAGE rewrites (green = THISISSAFE)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sage_lengths.png", dpi=200)
    plt.close(fig)
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    res = pd.read_csv("results/sage/results.csv")
    tradeoff(res)
    util = pd.read_csv("results/sage/utility.csv") if os.path.exists("results/sage/utility.csv") else None
    res_ = pd.read_csv("results/sage/residual.csv") if os.path.exists("results/sage/residual.csv") else None
    utility_plot(util)
    risk_graph_example()
    if util is not None and res_ is not None:
        privacy_utility_scatter(res, util, res_)
    print(sage_stats())


if __name__ == "__main__":
    main()
