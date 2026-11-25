"""Rarity- and graph-guided iterative generalisation of quasi-identifiers (QIs).

Data model (one record per document, see README "Input format"):
    {"doc_id": str, "text": str, "respondent": "TUR" (optional),
     "entities": [{"entity_id": str, "entity_type": "DATETIME|LOC|ORG|DEM|QUANTITY|MISC|CODE|PERSON",
                   "identifier_type": "DIRECT|QUASI|NO_MASK",
                   "mentions": [{"start": int, "end": int, "text": str}]}]}
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import networkx as nx

# ---------------------------------------------------------------- normalisation
_TITLES = re.compile(r"^(mr|mrs|ms|dr|mme|m|messrs)\.?\s+", re.I)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalise(text: str, etype: str) -> str:
    s = strip_accents(text).lower().strip()
    s = s.replace("’", "'").replace("‑", "-")
    s = _TITLES.sub("", s)
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"[“”\"(),;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .-")
    return s


# ---------------------------------------------------------------- generalisation hierarchies
MONTHS = ("january february march april may june july august september october november december").split()
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
_MONTH_YEAR = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(1[89]\d\d|20\d\d)\b")
COUNTRIES = {w for w in """albania andorra armenia austria azerbaijan belgium bulgaria croatia cyprus denmark estonia
finland france georgia germany greece hungary iceland ireland italy latvia liechtenstein lithuania luxembourg malta
moldova monaco montenegro netherlands norway poland portugal romania russia serbia slovakia slovenia spain sweden
switzerland turkey ukraine iraq iran syria algeria morocco tunisia nigeria afghanistan pakistan india china israel
lebanon egypt libya somalia sudan congo chechnya kosovo macedonia canada""".split()}
COUNTRIES |= {"czech republic", "russian federation", "san marino", "united kingdom", "united states",
              "sri lanka", "republic of turkey", "northern cyprus", "england", "scotland", "wales"}

ORG_RULES = [
    (r"security court", "a state security court"),
    (r"assize", "an assize court"),
    (r"cassation|supreme|constitutional|high court|court of appeal|appeal court|appellate", "a higher court"),
    (r"court|tribunal", "a court"),
    (r"prison|penitentiar|detention|remand|jail|gaol", "a detention facility"),
    (r"police|gendarmerie|security directorate|anti-terror|constabulary", "a law-enforcement body"),
    (r"prosecut", "a prosecutor's office"),
    (r"ministry|minister|government|governor|office|council|municipal|commission|agency|authority|department",
     "a public authority"),
    (r"hospital|clinic|medical|health", "a health institution"),
    (r"university|school|college|academy|institute", "an educational institution"),
    (r"party|front|movement|union|organisation|organization|association|pkk|dhkp|tkp|tdkp|eta|ira",
     "an association or political group"),
    (r"bank|company|ltd|gmbh|s\.a\.|plc|corporation|enterprise|newspaper|bbc|television|radio", "a company"),
]


def ladder(value: str, etype: str, respondent: str | None) -> list[str]:
    """Return the generalisation ladder [level0, level1, ..., top]. Top is always a type placeholder."""
    top = f"[{etype}]"
    steps = [value]
    if etype == "DATETIME":
        my = _MONTH_YEAR.findall(value)
        years = _YEAR.findall(value)
        if my and not re.fullmatch(r"(" + "|".join(MONTHS) + r")\s+\d{4}", value):
            steps.append(" / ".join(f"{m} {y}" for m, y in my))
        if years:
            steps.append(" / ".join(dict.fromkeys(years)))
            steps.append(" / ".join(dict.fromkeys(f"{y[:3]}0s" for y in years)))
    elif etype == "LOC":
        if value in COUNTRIES:
            steps.append("a country")
        else:
            steps.append(f"a place in {respondent or 'the respondent state'}")
    elif etype == "ORG":
        for pat, cat in ORG_RULES:
            if re.search(pat, value):
                steps.append(cat)
                break
    steps.append(top)
    out = []
    for s in steps:  # remove consecutive duplicates
        if not out or out[-1] != s:
            out.append(s)
    return out


# ---------------------------------------------------------------- data structures
@dataclass
class Entity:
    doc_id: str
    entity_id: str
    etype: str
    idtype: str
    mentions: list
    ladder: list
    level: int = 0

    @property
    def value(self) -> str:
        return self.ladder[self.level]

    @property
    def key(self) -> tuple:
        return (self.etype, self.value)

    @property
    def top(self) -> int:
        return len(self.ladder) - 1


@dataclass
class Doc:
    doc_id: str
    text: str
    respondent: str | None
    entities: list = field(default_factory=list)
    graph: nx.Graph | None = None

    @property
    def quasi(self):
        return [e for e in self.entities if e.idtype == "QUASI"]


def build_docs(records: list[dict]) -> list[Doc]:
    docs = []
    for r in records:
        d = Doc(r["doc_id"], r["text"], r.get("respondent"))
        for e in r["entities"]:
            norms = [normalise(m["text"], e["entity_type"]) for m in e["mentions"]]
            norms = [n for n in norms if n] or [e["mentions"][0]["text"].lower()]
            c = Counter(norms)
            canon = max(c, key=lambda n: (c[n], len(n)))
            ent = Entity(d.doc_id, e["entity_id"], e["entity_type"], e["identifier_type"], e["mentions"],
                         ladder(canon, e["entity_type"], d.respondent))
            if ent.idtype == "DIRECT":
                ent.level = ent.top  # direct identifiers are always suppressed
            d.entities.append(ent)
        d.graph = cooccurrence_graph(d)
        docs.append(d)
    return docs


# ---------------------------------------------------------------- graph
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])|\n+")


def cooccurrence_graph(doc: Doc) -> nx.Graph:
    """Nodes = QUASI entities; edge weight = number of sentences in which both are mentioned."""
    bounds, pos = [], 0
    for m in _SENT.finditer(doc.text):
        bounds.append((pos, m.start()))
        pos = m.end()
    bounds.append((pos, len(doc.text)))
    starts = [b[0] for b in bounds]

    def sent_of(off):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo

    g = nx.Graph()
    by_sent = defaultdict(set)
    for e in doc.quasi:
        g.add_node(e.entity_id)
        for m in e.mentions:
            by_sent[sent_of(m["start"])].add(e.entity_id)
    for ids in by_sent.values():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                w = g.get_edge_data(ids[i], ids[j], {"weight": 0})["weight"]
                g.add_edge(ids[i], ids[j], weight=w + 1)
    return g


def centrality(doc: Doc, kind: str) -> dict:
    g = doc.graph
    if kind == "none" or g.number_of_nodes() == 0:
        return {n: 0.0 for n in g.nodes}
    if kind == "degree":
        c = dict(g.degree())
    elif kind == "betweenness":
        c = nx.betweenness_centrality(g)
    else:
        raise ValueError(kind)
    mx = max(c.values()) or 1.0
    return {n: v / mx for n, v in c.items()}


# ---------------------------------------------------------------- scoring
def doc_frequency(docs: list[Doc]) -> Counter:
    """df[key] = number of documents containing an entity (any identifier type) with this current value."""
    df = Counter()
    for d in docs:
        df.update({e.key for e in d.entities})
    return df


def scores(docs, df, lam: float, cent_kind: str) -> dict:
    """score(v) = rarity(v) * (1 + lam * centrality(v)), rarity = -log(df/N)."""
    N = len(docs)
    out = {}
    for d in docs:
        cent = centrality(d, cent_kind) if lam > 0 else {}
        for e in d.quasi:
            r = -math.log(df[e.key] / N)
            out[(d.doc_id, e.entity_id)] = r * (1 + lam * cent.get(e.entity_id, 0.0))
    return out


def run_method(docs, k0=5, lam=0.0, cent_kind="degree", max_rounds=8, log=None):
    """Iteratively raise by one level every QUASI entity whose score >= tau = -log(k0/N)."""
    N = len(docs)
    tau = -math.log(k0 / N)
    for rnd in range(1, max_rounds + 1):
        df = doc_frequency(docs)
        sc = scores(docs, df, lam, cent_kind)
        changed = 0
        for d in docs:
            for e in d.quasi:
                if e.level < e.top and sc[(d.doc_id, e.entity_id)] >= tau:
                    e.level += 1
                    changed += 1
        if log is not None:
            log.append({"round": rnd, "raised": changed})
        if changed == 0:
            break
    return docs


def pair_frequency(docs) -> Counter:
    """pair_df[(a, b)] = number of documents whose QUASI values contain both a and b."""
    pdf = Counter()
    for d in docs:
        ks = sorted({e.key for e in d.quasi})
        pdf.update((ks[i], ks[j]) for i in range(len(ks)) for j in range(i + 1, len(ks)))
    return pdf


def triple_frequency(docs) -> Counter:
    tdf = Counter()
    for d in docs:
        ks = sorted({e.key for e in d.quasi})
        n = len(ks)
        tdf.update((ks[i], ks[j], ks[l]) for i in range(n) for j in range(i + 1, n) for l in range(j + 1, n))
    return tdf


def risk_hyperedges(keys, df, pdf, tdf, k0):
    """Unsafe units of one document: single values, pairs and (if tdf given) triples shared by < k0 documents.
    Returns a list of (frozenset_of_values, weight=1/k)."""
    ks = sorted(keys)
    n = len(ks)
    out = [(frozenset([k]), 1 / df[k]) for k in ks if df[k] < k0]
    for i in range(n):
        for j in range(i + 1, n):
            c = pdf[(ks[i], ks[j])]
            if c < k0:
                out.append((frozenset((ks[i], ks[j])), 1 / c))
    if tdf is not None:
        for i in range(n):
            for j in range(i + 1, n):
                for l in range(j + 1, n):
                    c = tdf[(ks[i], ks[j], ks[l])]
                    if c < k0:
                        out.append((frozenset((ks[i], ks[j], ks[l])), 1 / c))
    return out


def risk_graph(doc: Doc, df, pdf, k0: int) -> nx.Graph:
    """Pairwise risk graph of one document (for visualisation / betweenness): nodes = current QUASI values,
    node attr "own" = 1/df if df < k0, edge (a, b) with weight 1/k if {a, b} occurs in fewer than k0 documents."""
    h = nx.Graph()
    ks = sorted({e.key for e in doc.quasi})
    for k in ks:
        h.add_node(k, own=(1 / df[k]) if df[k] < k0 else 0.0)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            c = pdf[(ks[i], ks[j])]
            if c < k0:
                h.add_edge(ks[i], ks[j], weight=1 / c)
    return h


def greedy_cover(edges, choice, df, rng, frozen):
    """Pick values until every unsafe unit contains a picked value. `frozen` values (already at top) are skipped."""
    edges = [(set(u) - frozen, w) for u, w in edges]
    edges = [(u, w) for u, w in edges if u]
    picked = []
    while edges:
        score = defaultdict(float)
        for u, w in edges:
            for v in u:
                score[v] += w
        if choice == "degree":  # risk-weighted degree in the (hyper)graph
            v = max(score, key=lambda x: (score[x], -df[x]))
        elif choice == "rarest":
            v = min(score, key=lambda x: (df[x], -score[x]))
        elif choice == "random":
            v = rng.choice(sorted(score))
        elif choice == "betweenness":
            g = nx.Graph()
            g.add_nodes_from(score)
            for u, w in edges:
                if len(u) == 2:
                    g.add_edge(*u)
            bc = nx.betweenness_centrality(g)
            v = max(score, key=lambda x: (bc.get(x, 0.0), score[x]))
        else:
            raise ValueError(choice)
        picked.append(v)
        edges = [(u, w) for u, w in edges if v not in u]
    return picked


def run_riskgraph(docs, k0=5, choice="degree", order=2, max_rounds=30, seed=0, log=None):
    """Main method. Each round: recompute corpus statistics, list each document's unsafe values / combinations
    (up to `order` values) as a risk (hyper)graph, greedily pick values by `choice` until every unsafe unit is
    covered, and raise the picked values by one level. Repeat until nothing is unsafe (or max_rounds).
    order=3 first runs the pairwise stage to convergence (cheap), then adds triples."""
    if order == 3:
        run_riskgraph(docs, k0, choice, 2, max_rounds, seed, log)
    rng = random.Random(seed)
    for rnd in range(1, max_rounds + 1):
        df, pdf = doc_frequency(docs), pair_frequency(docs)
        tdf = triple_frequency(docs) if order == 3 else None
        raised = 0
        for d in docs:
            by_key = defaultdict(list)
            for e in d.quasi:
                by_key[e.key].append(e)
            frozen = {k for k, es in by_key.items() if all(e.level == e.top for e in es)}
            edges = risk_hyperedges(by_key.keys(), df, pdf, tdf, k0)
            for v in greedy_cover(edges, choice, df, rng, frozen):
                for e in by_key[v]:
                    if e.level < e.top:
                        e.level += 1
                        raised += 1
        if log is not None:
            log.append({"order": order, "round": rnd, "raised": raised})
        if raised == 0:
            break
    return docs


def run_random(docs, n_steps: int, seed: int):
    """Spend the same number of one-level generalisation steps as a reference method, uniformly at random."""
    rng = random.Random(seed)
    pool = [e for d in docs for e in d.quasi]
    for _ in range(n_steps):
        cand = [e for e in rng.sample(pool, min(len(pool), 50)) if e.level < e.top]
        if not cand:
            cand = [e for e in pool if e.level < e.top]
            if not cand:
                break
        rng.choice(cand).level += 1
    return docs


def suppress_all(docs):
    for d in docs:
        for e in d.quasi:
            e.level = e.top
    return docs


def reset(docs):
    for d in docs:
        for e in d.entities:
            e.level = e.top if e.idtype == "DIRECT" else 0
    return docs


def total_steps(docs) -> int:
    return sum(e.level for d in docs for e in d.quasi)


# ---------------------------------------------------------------- evaluation
def evaluate(docs, k_eval=5) -> dict:
    N = len(docs)
    df = doc_frequency(docs)
    q = [e for d in docs for e in d.quasi]
    rare = sum(df[e.key] < k_eval for e in q) / len(q)
    # pair-level k-anonymity over the set of QUASI values of each document
    keysets = [sorted({e.key for e in d.quasi}) for d in docs]
    pair_df = Counter()
    for ks in keysets:
        pair_df.update((ks[i], ks[j]) for i in range(len(ks)) for j in range(i + 1, len(ks)))
    uniq_pair_docs = risky_docs = 0
    risks = []
    for ks in keysets:
        kmin = min([df[k] for k in ks] + [N])
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                c = pair_df[(ks[i], ks[j])]
                if c < kmin:
                    kmin = c
        uniq_pair_docs += kmin == 1
        risky_docs += kmin < k_eval
        risks.append(1 / kmin)
    il = sum(e.level / e.top for e in q) / len(q)
    return {
        "rare_QI_%": 100 * rare,
        "docs_unique_%": 100 * uniq_pair_docs / N,
        f"docs_k<{k_eval}_%": 100 * risky_docs / N,
        "mean_R": sum(risks) / N,
        "info_loss_%": 100 * il,
        "QI_changed_%": 100 * sum(e.level > 0 for e in q) / len(q),
        "QI_suppressed_%": 100 * sum(e.level == e.top for e in q) / len(q),
    }


def evaluate_triples(docs, k_eval=5, per_doc=300, seed=0) -> dict:
    """Held-out check (the method only looks at single values and pairs): estimated share of documents having
    at least one QUASI-value triple shared by fewer than k_eval documents. Triples are sampled (per_doc per
    document), so the result is a lower bound on the true share."""
    rng = random.Random(seed)
    inv = defaultdict(set)
    keysets = []
    for i, d in enumerate(docs):
        ks = sorted({e.key for e in d.quasi})
        keysets.append(ks)
        for k in ks:
            inv[k].add(i)
    uniq = risky = 0
    for ks in keysets:
        n = len(ks)
        if n < 3:
            continue
        total = n * (n - 1) * (n - 2) // 6
        if total <= per_doc:
            trip = [(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)]
        else:
            trip = {tuple(sorted(rng.sample(range(n), 3))) for _ in range(per_doc)}
        kmin = len(docs)
        for a, b, c in trip:
            s = inv[ks[a]] & inv[ks[b]]
            if len(s) >= kmin:
                s = s & inv[ks[c]]
                kmin = min(kmin, len(s))
            else:
                kmin = min(kmin, len(s & inv[ks[c]]))
        uniq += kmin == 1
        risky += kmin < k_eval
    N = len(docs)
    return {"triple_unique_%": 100 * uniq / N, f"triple_k<{k_eval}_%": 100 * risky / N}


# ---------------------------------------------------------------- text rendering
def render(doc: Doc) -> str:
    """Replace every mention of a generalised entity by its current value (placeholders keep brackets)."""
    spans = []
    for e in doc.entities:
        if e.level == 0:
            continue
        for m in e.mentions:
            spans.append((m["start"], m["end"], e.value))
    spans.sort()
    out, pos = [], 0
    for s, t, v in spans:
        if s < pos:
            continue
        out.append(doc.text[pos:s])
        out.append(v)
        pos = t
    out.append(doc.text[pos:])
    return "".join(out)
