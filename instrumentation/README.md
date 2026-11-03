# 成本计量工具

这套工具记录 DGM 和 HGM 运行时的每一笔开销，用来测出自我改进循环中各个环节的实际成本占比。所有记录写入同一个 JSONL 账本，再由分析脚本汇总成报告。

## 组成

| 文件 | 作用 |
| --- | --- |
| `rsi_ledger.py` | 计量钩子。没有设置 `RSI_LEDGER` 环境变量时，所有钩子都不做任何事，仓库行为与上游完全一致。 |
| `llm_proxy.py` | 本地代理。把 OpenAI、OpenRouter 和本地 vLLM 的请求原样转发给对应的服务商，并记录每次调用的 token 数和美元花费。OpenRouter 的单价在启动时从其官方接口自动读取。 |
| `gpu_sampler.py` | 每隔几秒记录一次 GPU 利用率、显存和占用 GPU 的进程，用来计算本地模型的 GPU 时间，并发现与其他任务共用 GPU 的情况。 |
| `prices.json` | OpenAI 模型的单价（美元 / 百万 token），2026-10-11 已对照官方价格页核对。 |
| `vllm_metrics_sampler.py` | 每隔几秒读取 vLLM 自带的 `/metrics`：正在运行和排队的请求数、KV cache 占用、抢占次数、累计 token 数。写入单独的 `runs/vllm_metrics.jsonl`，不写账本。nvidia-smi 不可用时，它是判断 GPU 空闲还是忙碌的依据；`--summary --ledger --run` 按某个 run 的时间窗汇总。 |
| `analyze_ledger.py` | 汇总账本，输出各环节的花费、耗时和浪费指标，以及 GPU 占用小时（run 的墙钟时间，去掉 30 分钟以上没有任何记录的停机间隔）。`--max-hours` 按 GPU 占用小时截断（主要口径），`--max-evals` 按评估次数截断，都用于等预算对比。 |
| `results.py`、`compare.py` | 选出最终版本的规则，以及跨种子的自助法置信区间和方法间对比。最终评估脚本是 HGM 仓库里的 `final_eval.py`（`--max-hours` 截断，`--name` 指定检查点的输出目录）。 |
| `repeat_stats.py` | 实验 1 和 2 的统计：同一个智能体重复评估的噪声（每道题结果是否稳定、区分两个智能体需要多少道题），以及初始智能体的 pass@k、每遍的 GPU 成本，和搜索出的智能体比较时搜索成本在多少道题后摊平。重复评估脚本是 HGM 仓库里的 `repeat_eval.py`。 |
| `early_stop/` | 提前终止的离线回放，见该目录的 README。 |
| `tests/` | 单元测试，运行方式见文末。 |

两个仓库（`repos/dgm`、`repos/HGM`）的 `efficient-rsi` 分支里，`rsi_ledger.py` 是指向本目录的符号链接，并已加入 `.dockerignore`，因此智能体在容器中看不到它。

## 为什么用代理记录大模型调用

智能体会修改自己的代码，包括 `llm.py`。如果把计量代码写进智能体的文件，就等于改变了被测对象本身，智能体也可能改掉这些代码。因此，大模型调用改由代理在外部记录：编排代码给每个容器设置 `OPENAI_BASE_URL`，形如 `http://host.docker.internal:8787/t/<环节>/<版本>/<题目>/v1`。代理去掉前缀后，把请求原样转发给 OpenAI，同时记下这次调用属于哪个环节、哪个版本、哪道题。

## 账本记录的事件

| 事件 | 来源 | 含义 |
| --- | --- | --- |
| `llm_call` | 代理 | 一次大模型调用。`tag` 为 `propose`（生成修改）、`diagnose`（问题诊断）或 `eval`（评估时解题） |
| `propose` | `sample_child` / `self_improve` | 生成一个新版本的各阶段耗时：`setup`、`diagnose`、`agent_run` |
| `eval` | 各 `harness.py` | 评估一道题的各阶段耗时：`setup`（建容器、重放补丁、安装依赖）、`agent_run`、`test`（仅 Polyglot） |
| `eval_report` | SWE-bench 评测报告 | SWE-bench 在独立容器中跑测试的耗时 |
| `select_parent`、`node_created`、`propose_failed` | 外层循环 | 搜索树的结构：谁被选为父代，生成了哪些子代 |
| `eval_result`（HGM）、`node_evaluated`（DGM） | 外层循环 | 每个版本的评估成绩 |

带容器的阶段还会记录容器实际消耗的 CPU 秒数（`container_cpu_s`）。

## 运行步骤

以下命令假设在 `efficient RSI/` 目录下执行，且两个仓库都已按各自 README 完成环境准备（Docker、conda 环境、SWE-bench 或 Polyglot 数据）。

```bash
export OPENAI_API_KEY=...                       # 只给代理和编排进程使用
export RSI_LEDGER=$PWD/runs/ledger.jsonl
export RSI_PROXY_URL=http://127.0.0.1:8787
mkdir -p runs

# 1. 启动代理和 GPU 采样器（各开一个终端）
#    代理只绑定 docker0 网桥地址，校园网上的其他机器访问不到
python instrumentation/llm_proxy.py --bind 172.17.0.1 --port 8787 --vllm-upstream http://127.0.0.1:8000
python instrumentation/gpu_sampler.py --interval 5 --expect-pid <vLLM 进程号>

# 2a. HGM，修复版（默认）
cd repos/HGM && RSI_RUN=hgm-fixed-seed0 python hgm.py --polyglot --max_task_evals 800

# 2b. HGM，上游原版行为（对照组）
cd repos/HGM && HGM_FIX_DESCENDANT_EVALS=0 RSI_RUN=hgm-orig-seed0 python hgm.py --polyglot --max_task_evals 800

# 2c. DGM 对照组：在 HGM 的代码框架上运行 DGM 的搜索策略（dgm_search.py）
#     初始智能体、评测代码和模型都与 HGM 相同，区别只在搜索方式
cd repos/HGM && RSI_RUN=dgm-seed0 python hgm.py --search dgm --polyglot --max_task_evals 800

#     （补充验证用）直接运行 DGM 自己的代码时，先用实验所用的模型重新评估初始智能体
# cd repos/dgm && DGM_CODE_MODEL=... DGM_SELF_IMPROVE_MODEL=... DGM_DIAGNOSE_MODEL=... \
#   python eval_initial_agent.py --polyglot --out initial_polyglot_<model>
# DGM_INITIAL_DIR=initial_polyglot_<model> python DGM_outer.py --polyglot

# 3. 最终评估：在第 800 次评估处选出最终版本，并与初始智能体一起在 166 道留出题上评估
#    最终评估记在 RSI_RUN=<原标签>-final 下，不计入搜索成本
cd repos/HGM && python final_eval.py --output_dir output_hgm/<run> --run hgm-fixed-seed0 --polyglot --max-evals 800

# 4. 跨种子对比：留出题准确率的自助法置信区间、方法之间的差值，以及各自的搜索成本
python instrumentation/compare.py --result hgm-fixed=<run1>/final_eval/final_result.json \
    --result hgm-fixed=<run2>/final_eval/final_result.json ... --result dgm=... --reference dgm --ledger runs/ledger.jsonl

# 5. 单次运行的成本分解
python instrumentation/analyze_ledger.py runs/ledger.jsonl --run hgm-fixed-seed0 --max-evals 800 --nodes-csv runs/hgm-fixed-seed0-nodes.csv
```

**Linux 注意事项：** Linux 上的 Docker 默认不解析 `host.docker.internal`，需要改为设置 `RSI_PROXY_URL_CONTAINER=http://172.17.0.1:8787`（docker0 网桥地址），并确认防火墙允许容器访问该端口。

## 模型命名规则（HGM）

HGM 按模型名决定调用方式：名字里含 `gpt` 或以 `o` 开头的走 OpenAI；含 `vllm` 的走本地 vLLM；其余的走 OpenRouter。因此本地模型的名字必须含 `vllm`（例如 `vllm-qwen3c-local`），OpenRouter 上的模型直接用其编号（例如 `qwen/qwen3-coder`），不能写成 `openrouter/...`，否则会因为以 `o` 开头而被当成 OpenAI 模型。计量开启时，三种调用都会经过代理。

## 已知限制

- 代理覆盖 OpenAI、OpenRouter 和本地 vLLM，不覆盖 Anthropic 和 Bedrock。DGM 上游默认用 Bedrock 上的 Claude 3.5 Sonnet 解题、用 o1 诊断，因此 DGM 必须设置 `DGM_CODE_MODEL`、`DGM_SELF_IMPROVE_MODEL` 和 `DGM_DIAGNOSE_MODEL`；不设置时保持上游默认行为。
- OpenRouter 的单价是其公开的标价；实际承接请求的服务商可能按不同价格计费，长提示词的阶梯价格也没有计入。
- DGM 自带的 `initial/` 和 `initial_polyglot/` 是上游模型的成绩。换模型后必须先运行 `eval_initial_agent.py`，否则根节点的分数来自另一个模型。HGM 没有这个问题：它在初始结果不存在时会自动用 `downstream_llm` 评估初始智能体。
- 流式（streaming）调用会被转发，但不解析 token 用量。两个仓库目前都没有使用流式调用。
- HGM 每次只评估一道题，因此 `eval` 事件是逐题的；DGM 一次评估多道题，各题的 `eval` 事件会并行写入。

## 运行测试

```bash
cd instrumentation && python3 -m unittest discover -s tests -v
```

测试覆盖账本写入、代理的转发与计费、分析脚本的浪费指标和 GPU 小时截断、vLLM 指标的解析、重复评估的统计，以及 HGM `tree.py` 修复前后的行为。提前终止的测试在 `early_stop/` 目录下单独运行：`cd early_stop && python3 -m unittest test_early_stop`。
