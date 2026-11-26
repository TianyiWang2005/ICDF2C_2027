"""Paths of one experimental run. QI_RUN unset = the 254-doc feasibility run (TAB dev+test);
QI_RUN=full = all 1266 TAB documents (minus the 2 few-shot examples)."""
import os

RUN = os.environ.get("QI_RUN", "")
SUF = f"_{RUN}" if RUN else ""
RESULTS = f"results/{RUN}" if RUN else "results/sage"
FIGURES = f"{RESULTS}/figures" if RUN else "results/figures"
SAGE = f"data/sage/sage_echr{SUF}.jsonl"
QI = f"data/qi/sage_qi{SUF}.jsonl"
LADDERS = f"data/generalisation/llm_ladders{SUF}.json"
CORPORA = f"data/corpora{SUF}"
GRAPHS = f"data/graphs{SUF}"
QA = f"data/utility/qa{SUF}.json"
SPLITS = ("train", "dev", "test") if RUN == "full" else ("dev", "test")
