#!/usr/bin/env bash
# Reproduce every experiment in this repository.
#   bash run_all.sh            -> 254-document feasibility run (TAB dev+test)   -> results/sage/
#   QI_RUN=full bash run_all.sh -> full run, 1266 documents (all TAB minus 2 few-shot examples) -> results/full/
# Needs: TAB checkout at $TAB (default ../tab) and DeepSeek API access (DEEPSEEK_API_KEY, or the session proxy).
# All LLM outputs are cached in data/llm_cache/ (*.jsonl.gz in git: run `gunzip -k data/llm_cache/*.gz` first),
# so a rerun reproduces the same numbers without new API calls.
set -euo pipefail
TAB=${TAB:-../tab}
RUN=${QI_RUN:-}
SUF=${RUN:+_$RUN}
OUT=results/${RUN:-sage}
SPLITS="dev test"; [ "$RUN" = "full" ] && SPLITS="train dev test"

# 0. pilot on TAB gold annotations (CPU only, run once)
if [ -z "$RUN" ]; then
  python run_experiments.py --tab_dir "$TAB" --out results/tab_gold --k0 2 5 10
  python run_extra.py       --tab_dir "$TAB" --out results/tab_gold --k0 2 5 10
  python -m pipeline.qi_extract validate --tab_dir "$TAB" --out data/qi/validation          # QI extractor vs gold
fi

# 1. SAGE rewriting
python -m pipeline.sage --tab_dir "$TAB" --splits $SPLITS --out data/sage/sage_echr$SUF.jsonl
# 2. LLM QI extraction on the SAGE rewrites
python -m pipeline.qi_extract tag --tab_dir "$TAB" --inp data/sage/sage_echr$SUF.jsonl --out data/qi/sage_qi$SUF.jsonl
# 3. LLM generalisation ladders for values without a rule hierarchy
python -m pipeline.ladders --inp data/qi/sage_qi$SUF.jsonl --out data/generalisation/llm_ladders$SUF.json
# 4. risk-graph generalisation: all methods, k0 = 2/5/10
python -m pipeline.run_main --inp data/qi/sage_qi$SUF.jsonl --ladders data/generalisation/llm_ladders$SUF.json \
  --out "$OUT" --corpora data/corpora$SUF --graphs data/graphs$SUF
# 5. evaluation
python -m pipeline.evaluate residual --out "$OUT/residual.csv" \
  --only "sage_k5" "suppress_k5" "rarity_k5" "ours_o2_k5" "ours_o3_k(2|5|10)" "random_budget_o3_k5" "rg_degree_o3_k5"
if [ "$RUN" = "full" ]; then   # external 200x5 question set (its 200 cases include 33 train cases)
  python -m pipeline.evaluate utility --qa tab200 --out "$OUT/utility_tab200.csv" \
    --only "sage_k5" "suppress_k5" "ours_o2_k5" "ours_o3_k5"
else                           # 254-doc run: our 2 generated questions per document
  python -m pipeline.evaluate utility --out "$OUT/utility.csv" \
    --only "sage_k5" "suppress_k5" "rarity_k(2|5|10)" "ours_o2_k(2|5|10)" "ours_o3_k(2|5|10)" "random_budget_o3_k(2|5|10)" "rg_degree_o3_k5"
fi
# 6. subgroup / robustness analyses, figures, prompts
python -m pipeline.robustness
python -m pipeline.analysis
python -m pipeline.figures
python -m pipeline.plot_doc_graph --method ours_o3 --k0 5
python -m pipeline.export_prompts
