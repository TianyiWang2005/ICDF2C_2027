"""Extra runs not covered by run_experiments.py: betweenness choice and order=3 (pairs then triples).

usage: python run_extra.py --tab_dir ../tab --out results/archive/tab_gold_v2 --k0 5
"""
import argparse
import glob
import os
import time

import pandas as pd

from qi_gen.core import build_docs, reset, run_riskgraph
from qi_gen.io import load_tab
from run_experiments import measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab_dir", required=True)
    ap.add_argument("--out", default="results/archive/tab_gold_v2")
    ap.add_argument("--k0", type=int, nargs="+", default=[5])
    ap.add_argument("--configs", nargs="+", default=["betweenness:2", "lookahead:2", "rarest:3", "degree:3", "cost:3"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    docs = build_docs(load_tab(sorted(glob.glob(os.path.join(a.tab_dir, "echr_*.json")))))
    rows = []
    for k0 in a.k0:
        for cfg in a.configs:
            choice, order = cfg.split(":")
            reset(docs); log = []; t = time.time()
            run_riskgraph(docs, k0=k0, choice=choice, order=int(order), log=log)
            r = {"method": f"Risk graph + {choice}, order={order}", "k0": k0, **measure(docs, k0),
                 "rounds": len(log), "seconds": round(time.time() - t, 1)}
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "results_extra.csv"), index=False)


if __name__ == "__main__":
    main()
