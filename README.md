# Hidden quasi-identifiers in SAGE synthetic RAG data

Code and data for the ICDF2C 2027 paper. SAGE (Zeng et al., EMNLP 2025) rewrites a RAG retrieval corpus into
synthetic text. We show that the rewritten corpus still contains **hidden quasi-identifiers (QIs)**: values that
are harmless alone but unique in combination, and we remove them with an iterative, risk-graph-guided
generalisation.

| Where | What |
|---|---|
| [`DIRECTORY.md`](DIRECTORY.md) | what every directory and file in this repository is |
| [`docs/framework.md`](docs/framework.md) | method, flowchart, progress |
| [`docs/results_analysis.md`](docs/results_analysis.md) | results and analysis |
| `results/full/` | main results (1,266 TAB documents) |
| `run_all.sh` | reproduce everything |

## Pipeline

1. **SAGE rewriting** of TAB (ECHR judgments), re-implemented after the official code, LLM = DeepSeek (`pipeline/sage.py`).
2. **QI extraction** with a 2-shot LLM tagger, validated against the TAB gold annotations (`pipeline/qi_extract.py`).
3. **Risk graph**: per document, nodes = QI values, edges = value pairs shared by fewer than k₀ documents of the corpus (`qi_gen/core.py`).
4. **Iterative generalisation**: greedily cover every unsafe pair (and triple, order 3), raise the chosen values one level on a rule- or LLM-built generalisation ladder, recount, repeat until nothing is unsafe (`qi_gen/core.py`, `pipeline/ladders.py`, `pipeline/run_main.py`).
5. **Evaluation**: pair/triple k-anonymity, information loss, residual original QIs (LLM judge), RAG QA utility (`pipeline/evaluate.py`).

## Reproduce

```bash
pip install networkx pandas openai rank_bm25 matplotlib
git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark.git ../tab
gunzip -k data/llm_cache/*.gz           # cached LLM outputs: reruns make no API calls
bash run_all.sh                         # 254-document feasibility run
QI_RUN=full bash run_all.sh             # full run, 1,266 documents
```

The TAB-gold pilot alone (CPU only): `python run_experiments.py --tab_dir ../tab --out results/tab_gold --k0 2 5 10`.
