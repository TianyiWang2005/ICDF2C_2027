#!/usr/bin/env bash
# Full pipeline. Needs: TAB checkout at $TAB (default ../tab), DeepSeek API access (DEEPSEEK_API_KEY or proxy).
# All LLM outputs are cached under data/llm_cache/, so rerunning reproduces the same results at no cost.
set -euo pipefail
TAB=${TAB:-../tab}

# 0. pilot on TAB gold annotations (CPU only)
python run_experiments.py --tab_dir "$TAB" --out results_tab_v3 --k0 2 5 10
python run_extra.py       --tab_dir "$TAB" --out results_tab_v3 --k0 2 5 10

# 1. SAGE rewriting of TAB dev+test (254 docs)
python -m pipeline.sage --tab_dir "$TAB" --out data/sage/sage_echr.jsonl

# 2. LLM QI extraction: validation on TAB test gold, then tagging of the SAGE rewrites
python -m pipeline.qi_extract validate --tab_dir "$TAB" --out data/qi/validation
python -m pipeline.qi_extract tag      --tab_dir "$TAB" --inp data/sage/sage_echr.jsonl --out data/qi/sage_qi.jsonl

# 3. LLM generalisation ladders for values without a rule hierarchy
python -m pipeline.ladders --inp data/qi/sage_qi.jsonl --out data/generalisation/llm_ladders.json

# 4. risk-graph scoring + iterative generalisation, all baselines/ablations, k0 sweep
python -m pipeline.run_main --inp data/qi/sage_qi.jsonl --ladders data/generalisation/llm_ladders.json --out results/sage

# 5. evaluation: residual original QIs (LLM judge) and RAG utility
python -m pipeline.evaluate residual --with_original --out results/sage/residual.csv
python -m pipeline.evaluate utility  --out results/sage/utility.csv

# 6. figures, prompts
python -m pipeline.figures
python -m pipeline.export_prompts
