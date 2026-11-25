# qi_gen: risk-graph-guided generalisation of hidden quasi-identifiers

Pilot code for the ICDF2C 2027 paper. Project context: see `CLAUDE.md`.

## Install
```
pip install networkx pandas
git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark.git tab
```

## Run
```
python run_experiments.py --tab_dir tab --out results_tab --k0 2 5 10
python run_experiments.py --jsonl sage_with_qi.jsonl --out results_sage --k0 2 5 10
```
Outputs: `results.csv` (one row per method and k0), `generalised_k{k0}.jsonl` (rewritten texts and the level of
every QI), `round_logs.json` (values raised per round).

## Input format for SAGE outputs (one JSON object per line)
```json
{"doc_id": "001-100188",
 "text": "<SAGE final_context>",
 "respondent": "TUR",
 "entities": [
   {"entity_id": "e1", "entity_type": "DATETIME", "identifier_type": "QUASI",
    "mentions": [{"start": 120, "end": 134, "text": "6 February 1993"}]}
 ]}
```
`entity_type` follows TAB (PERSON, CODE, LOC, ORG, DEM, DATETIME, QUANTITY, MISC); `identifier_type` is DIRECT,
QUASI or NO_MASK; offsets are character offsets into `text`; mentions of the same entity share one `entity_id`.
`respondent` (ISO3 code of the respondent state) is optional and only used to generalise locations.

## Method (qi_gen/core.py: run_riskgraph)
Each round: count, over the whole corpus, how many documents contain each current QI value (df) and each pair of
values (pair df). In every document, a value with df < k0 or a pair with pair df < k0 is unsafe. Unsafe pairs form
the document's risk graph (edge weight 1/k). Values are picked greedily by risk-weighted degree until every unsafe
unit is covered, and each picked value is raised one level in its generalisation ladder. Rounds repeat until no
document has an unsafe value or pair. `order=3` adds triples after the pairwise stage (not yet validated).
`choice` = degree (ours) | rarest | random | betweenness, for ablations.
