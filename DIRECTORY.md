# 仓库目录说明

> ICDF2C 2027 论文：在 SAGE 改写后的合成 RAG 语料中发现并泛化**隐藏准标识符（QI）**。
> 本文件说明仓库每个部分是什么、哪些是最终结果、哪些是归档。

**第一次阅读建议的顺序**
1. [`docs/framework.md`](docs/framework.md)：方法框架、流程图、当前进度
2. [`docs/results_full.md`](docs/results_full.md)：**全数据集主结果**（1,266 篇）
3. [`docs/results_analysis.md`](docs/results_analysis.md)：254 篇可行性验证的结果与各指标的详细解释
4. 本文件：需要找某个具体文件时查阅

---

## 1. 总览

```
icdf2c_2027/
├── DIRECTORY.md          ← 本文件
├── README.md             项目简介和快速上手
├── CLAUDE.md             项目背景与当前状态（给 Claude 会话读的项目说明）
├── run_all.sh            一键复现全部实验
│
├── docs/                 文档：框架说明（framework）、主结果（results_full）、可行性验证（results_analysis）
├── qi_gen/               核心算法：风险图 + 迭代泛化（只用 CPU）
├── pipeline/             完整流水线：SAGE 复现、QI 抽取、评估、画图（会调用 LLM）
├── prompts/              所有 LLM prompt 的原文（论文附录用）
├── run_experiments.py    TAB 原标注上的 pilot 实验（主实验也复用其中的指标函数）
├── run_extra.py          pilot 的补充实验（介数中心性、order 3 等）
│
├── data/                 所有中间数据和生成的数据集
└── results/              所有实验结果（表格、日志、图）
```

两轮实验共用同一套代码，用环境变量 `QI_RUN` 区分：

| 实验轮次 | 文档 | 设置方式 | 结果目录 | 数据文件后缀 |
|---|---|---|---|---|
| 可行性验证 | 254 篇（TAB dev + test） | 不设置 | `results/sage/` | 无后缀，如 `sage_echr.jsonl` |
| **全数据集（主结果）** | 1,266 篇（TAB 全部 1,268 篇，去掉 2 篇 few-shot 示例） | `QI_RUN=full` | `results/full/` | `_full`，如 `sage_echr_full.jsonl` |

---

## 2. 代码

### `qi_gen/`：核心算法（论文方法本身，只用 CPU）

| 文件 | 内容 |
|---|---|
| `core.py` | QI 值规范化、泛化层级（规则部分）、风险图、贪心覆盖、迭代泛化循环、各项隐私指标、泛化后文本渲染 |
| `io.py` | 读取 TAB 原始标注，以及通用 JSONL 格式（SAGE 文本 + QI 标注） |

### `pipeline/`：完整流水线

| 文件 | 对应步骤 | 是否调用 LLM |
|---|---|---|
| `sage.py` | 第 0 步：SAGE 的 ECHR 版复现（属性抽取 → 生成 → 隐私审查 ⇄ 改写） | 是 |
| `qi_extract.py` | 第 1 步：LLM few-shot 抽取 QI；在 TAB test 上与人工标注对比验证 | 是 |
| `ladders.py` | 第 4 步：为规则覆盖不了的 QI 值生成泛化层级 | 是 |
| `run_main.py` | 第 2–4 步：在 SAGE 输出上跑所有方法和 k₀，输出结果表、各版本语料、风险图 | 否 |
| `evaluate.py` | 评估：原文 QI 残留率、RAG 问答效用 | 是 |
| `robustness.py` | 分组与稳健性分析：SAGE safe 与不 safe 对比、只用 safe 文档重跑、QI 抽取在 SAGE 文本上的召回 | 否 |
| `utility_diagnostics.py` | 效用评估的诊断分析（不调用 LLM） |
| `analysis.py` | 描述性统计：QI 类型分布、各类型泛化程度、文本示例 | 否 |
| `figures.py`、`plot_doc_graph.py` | 画图：隐私-效用曲线、单篇文档泛化前后的风险图等 | 否 |
| `prompts.py`、`export_prompts.py` | 所有 prompt 的定义；导出到 `prompts/` | 否 |
| `llm.py` | DeepSeek 调用、缓存、并发 | — |
| `paths.py` | 两轮实验的路径配置（`QI_RUN`） | — |

---

## 3. 数据 `data/`

大文件在 git 中以 `.gz` 压缩保存，使用前先解压：`gunzip -k <文件>.gz`。

| 目录 | 内容 | 254 篇 | 1,266 篇 |
|---|---|---|---|
| `sage/` | **SAGE 改写后的数据集**：每篇的原文、改写结果、属性摘要、每轮审查意见、是否 safe | `sage_echr.jsonl.gz` | `sage_echr_full.jsonl.gz` |
| `qi/` | SAGE 文本上抽出的 QI（类型、DIRECT/QUASI、字符位置） | `sage_qi.jsonl.gz` | `sage_qi_full.jsonl.gz` |
| `qi/validation*/` | QI 抽取在 TAB test 原文上的验证；`validation` 用 2 个示例（最终采用），`validation_shots{0,4,8}` 是示例数量对比 | 共用 | 共用 |
| `generalisation/` | LLM 生成的泛化层级 | `llm_ladders.json` | `llm_ladders_full.json` |
| `corpora/`、`corpora_full/` | **各方法处理后的语料**，每种方法、每个 k₀ 一个文件（命名规则见第 5 节） | `corpora/` | `corpora_full/` |
| `corpora_common10/` | 敏感性分析的语料（出现在 ≥10% 文档中的值视为不可识别） | ✓ | — |
| `graphs/`、`graphs_full/` | **风险图**：`risk_graphs_k{2,5,10}` 是每篇文档泛化前的风险图（节点、不安全边、k 值）；`cooccurrence_graphs` 是 QI 共现图 | `graphs/` | `graphs_full/` |
| `utility/` | 效用评估的问答题：`qa.json`、`qa_full.json` 是我们用 DeepSeek 生成的（每篇 2 题）；`tab200/` 是学弟用 ChatGPT 生成的外部题库（200 篇 × 5 题，带原文出处）。**全数据集的正式效用评估用 `tab200/`** | `qa.json` | `tab200/` |
| `human_eval/` | **人工验证材料（待标注）**：标注表、对应的 SAGE 文本、LLM 判断（标完前不要看）、κ 计算脚本 | ✓ | — |
| `llm_cache/` | **所有 LLM 调用的完整记录**（输入、输出、模型名、token 数），两轮共用。重跑时命中缓存即不再调用 API | 共用 | 共用 |

---

## 4. 结果 `results/`

### 最终结果

| 目录 | 内容 |
|---|---|
| **`full/`** | **全数据集（1,266 篇）主结果** |
| `sage/` | 254 篇可行性验证的结果，图在 `sage/figures/` |
| `sage_common10/` | 254 篇的敏感性分析（ρ = 10%） |
| `tab_gold/` | 直接在 TAB 人工标注（原文，1,268 篇）上的 pilot 结果，最终代码版本 |

`full/` 和 `sage/` 里的文件：

| 文件 | 内容 |
|---|---|
| `results.csv` | **主表**：每种方法 × 每个 k₀ 的组合风险（两两、三元组）、信息损失、泛化步数、轮数 |
| `residual.csv`、`residual_per_doc.json(.gz)` | 原文 QI 残留率（总体 + 按类型）；逐篇逐项的 LLM 判断 |
| `utility_tab200.csv`（full）、`utility.csv`（sage） | RAG 问答效用：得分、检索命中率；`tab200` 版本还按"依赖 / 不依赖 QI"分组 |
| `utility_*_answers.json(.gz)` | 每道题、每个版本的检索结果、回答和得分 |
| `utility_bootstrap.json`（sage） | 效用差异的配对 bootstrap 置信区间 |
| `utility_tab200_diagnostics.json`（full） | 效用损失的诊断：bootstrap、检索与回答拆分、按题型、按是否依赖 QI、被泛化的相关 QI |
| `robustness.json` | SAGE safe 与不 safe 分组、只用 safe 文档重跑、QI 抽取在 SAGE 文本上的召回 |
| `analysis.json`（sage） | QI 类型分布、各类型泛化程度、泛化前后文本示例 |
| `sage_stats.json` | SAGE 改写统计：safe 数量、平均轮数、长度变化 |
| `round_logs.json` | 每种方法每一轮泛化了多少个值 |
| `figures/` | 隐私-效用曲线、001-83927 泛化前后风险图、SAGE 长度散点图等 |
| `*.log` | 运行日志 |

### 归档 `archive/`（不再更新，仅供追溯）

| 目录 | 内容 | 为什么归档 |
|---|---|---|
| `tab_gold_pilot_v1/` | 最早的 TAB pilot 结果（含 `results_pilot_old_code.csv`） | 被 `tab_gold/` 取代 |
| `tab_gold_v2/` | 第一次重跑 TAB pilot | 早于"占位符不计入组合"和"值规范化"两处修正 |
| `sage254_v1/` | 254 篇第一版结果（包括当时的风险图） | 早于方法改进（值规范化、主方法改为最稀有优先） |
| `full_utility_generated_qa_partial/` | 全数据集用我们自己生成的题做的效用评估，只完成了 4/10 个版本（API 余额不足中断） | 已改用 `tab200` 题库 |

---

## 5. 命名约定

**语料版本名**（`data/corpora*/` 的文件名、结果表的 `key` 列）：`<方法>_k<k₀>`

| 名称 | 含义 |
|---|---|
| `sage` | SAGE 原始输出，不做泛化 |
| `suppress` | 所有 QUASI 抹成占位符（隐私上界） |
| `rarity` | 只按单个值的稀有度泛化（不看组合） |
| `random_budget_o2` / `_o3` | 和 ours 同样的泛化步数，随机分配 |
| `rg_random_o2` | 风险图 + 随机选节点 |
| `rg_degree_o2` / `_o3` | 风险图 + 风险加权度（消融） |
| `rg_cost_o2` / `_o3` | 风险图 + 考虑代价的加权度（消融） |
| `rg_lookahead_o2` | 风险图 + 带前瞻的考虑代价（消融） |
| `rg_betweenness_o2` | 风险图 + 介数中心性（消融） |
| **`ours_o2` / `ours_o3`** | **本文方法：风险图 + 迭代覆盖 + 最稀有优先**；o2 = 只检查两两组合，o3 = 再检查三元组 |

**k₀**：k-匿名的门槛，组合出现的文档数少于 k₀ 即不安全，取 2、5、10。

---

## 6. 复现

```bash
gunzip -k data/llm_cache/*.gz          # 先解压缓存，重跑时不再调用 API
bash run_all.sh                        # 254 篇
QI_RUN=full bash run_all.sh            # 1,266 篇
```

需要 TAB 数据集（`git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark.git`，路径用环境变量 `TAB` 指定）。
