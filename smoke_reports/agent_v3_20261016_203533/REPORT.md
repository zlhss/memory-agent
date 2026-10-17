# Agent V3 真实端到端 Smoke Test

执行时间：2026-10-16 20:35:33–20:37:57（Asia/Shanghai）。

原始严格判定为 **5/6 PASS、1/6 FAIL**。六个请求全部完成，均生成非空 `personalized_reply` 和完整 `memory_v2_details`；实际 ADD、UPDATE、DELETE 均成功且写盘与内存一致。唯一 FAIL 来自本次 Smoke 自定义的“更新后不得包含旧城市名”断言：更新结果保留迁居历史，当前城市仍是南京。因此不能将其报告为全绿，也不能将它误报为 UPDATE 未执行。未发现这六个请求中的程序异常、重复记录、跨联系人污染或计时丢失。

## 环境与隔离证据

- 解释器：`C:\Users\33167\anaconda3\envs\memory_agent\python.exe`，实际 Python **3.12.14**（与 README 中历史实验的 3.10.8 不同）。
- Ollama：本地 `http://localhost:11434/api/generate`，服务版本 **0.40.5**。
- 文本模型：**qwen2.5:7b-instruct**，7.6B / Q4_K_M；digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`。
- Embedding：真实 SentenceTransformer **paraphrase-multilingual-MiniLM-L12-v2**，CPU，384 维；sentence-transformers 6.0.1。禁用 Hub 联网，使用现有缓存，实际执行 **28 次 encode**。
- 调用入口：`main.process_user_message(sender_id, incoming, memory_version="v2")`。LLM、Embedding、Alignment、Evidence/Worthiness 和 CRUD 均使用实际实现；测量包装只转发调用及记录结果，不替换推理返回。
- 在导入 `memory_search` 前切换工作目录至 `C:\Users\33167\AppData\Local\Temp\agent-v3-real-smoke-km6mln9r`，长期记忆读写指向该目录的 `memories.json`。
- `style_memory.load_style_samples` 用真实加载函数显式绑定该目录的 `style_samples.json`；`main.record_feedback` 用真实反馈函数显式绑定该目录的 `feedback.json`。不能仅修改模块常量，因为函数默认参数已在定义时绑定。
- 首个真实请求前验证 cwd、三个绝对路径、空反馈、合成样本、内存状态与临时文件状态一致。Python audit hook 拒绝访问这些数据名的其他路径，并拒绝向报告目录之外的项目文件写入；运行中 **0 次拦截**。
- 运行前后对项目根目录原有 **53 个文件**逐一核对 SHA-256，全部不变，包括真实 `memories.json`、`feedback.json`、`style_samples.json`、冻结评测脚本和 JSON 结果。没有运行会重置真实记忆的旧 `integration_smoke_eval.py`。
- 测试临时目录已删除；报告内仅保留合成输入、运行快照和真实响应。

联系人 A：`agent_v3_smoke_alpha`，初始记忆为空。联系人 B：`agent_v3_smoke_beta`，合成初始记忆为“联系人住在厦门”。独立哨兵 `agent_v3_smoke_sentinel` 的“联系人喜欢天文观测”在所有请求中保持不变。风格样本也由本次脚本生成，各联系人只使用自己的样本。

## 每个请求的实际结果

下表 LLM 耗时由独立 HTTP 测量获得，包含客户端传输时间；请求耗时在主入口外独立测量，不包括反馈确认步骤和初始化。

| 测试 | 联系人输入 | 关键结果 | 严格状态 | 真实 LLM 次数 | LLM 合计 / s | 请求 / s |
|---|---|---|---|---:|---:|---:|
| NEW | A：我住在苏州。 | persistent → NEW；Evidence/Worthiness 通过；ADD 一条“联系人住在苏州。” | PASS | 7 | 26.229 | 26.423 |
| UPDATE | A：我现在已经搬到南京居住了。 | persistent_change → UPDATE index 0；旧记录被替换为“联系人从苏州搬到南京居住。” | FAIL¹ | 7 | 23.784 | 23.937 |
| 持久化检索 | A：我现在住在哪个城市？ | 先从磁盘重载记忆并清空 A 的 Working Memory；召回南京；query 不写入 | PASS | 5 | 15.870 | 15.963 |
| 语义去重 | A：我住在南京。 | persistent → DUPLICATE index 0；0 个 CRUD action；所有记忆保持不变 | PASS | 7 | 21.539 | 21.692 |
| 联系人隔离 | B：我现在住在哪个城市？ | 仅召回厦门；B 的 Prompt、Embedding 输入及整个返回对象没有 A ID、苏州或南京；query 不写入 | PASS | 5 | 14.926 | 15.031 |
| DELETE | A：我不再住在南京了。 | persistent_change → DELETE index 0；删除 A 的唯一目标记录；B 和哨兵不变 | PASS | 7 | 21.429 | 21.579 |

¹ 唯一失败断言 `update_replaces_old_fact_once` 包含条件 `"苏州" not in final[0]`。实际结果中的“从苏州搬到南京”保留了历史信息，触发该条件；这与“旧的当前居住记录仍存在”并不等价。原始 FAIL、断言和输入全部保留，未为了通过修改阈值、Prompt 或删除断言。

实际 `personalized_reply`：

1. NEW：您好！您现在住在苏州啊。苏州是个美丽的历史文化名城，有啥旅游、生活或工作上的需求尽管告诉我，我会尽力帮忙。
2. UPDATE：您好！听说您现在搬到南京居住了，那边的生活您适应得怎么样？苏州那边还有啥未了的心愿或回忆呢？
3. 检索：您记得吗？您是从苏州搬到南京居住的，现在您就在南京。
4. 去重：确认一下，您现在住在南京，对吧？之前从苏州搬过来后，生活适应得怎么样？
5. B 隔离：记得您住在厦门。
6. DELETE：了解，那您现在住哪里了？生活一切顺利吗？

所有请求的 `memory_v2_details` 中，`pipeline_version`、sender、Working Memory 快照、facts、fact retrieval、alignment、CRUD 结果、coverage 和 final_memories 都已检查。`final_memories` 与实时状态及重新加载的磁盘状态一致；所有已执行 action 的 success 为 true；change coverage 没有遗漏。UPDATE 的 Restricted Consistency 返回空次级动作，没有连带修改。

每轮随后调用真实 `finalize_response(..., "accepted")`，仅用于模拟反馈闭环，不代表人工质量评分。临时 feedback 精确新增一条记录，Working Memory 写入实际 incoming/reply；六轮均通过。

最终临时长期记忆为 A `[]`，B `["联系人住在厦门"]`，哨兵 `["联系人喜欢天文观测"]`。不存在重复条目。

## 计时与调用核对

- **38 次真实文本生成 POST，38/38 HTTP 200，38/38 LLM timing status=success**；没有额外 Judge、重试、隐式 V1 fallback 或用假结果替代失败。
- 新增阶段顺序：Planner → Draft → Style Rewriter → Atomic Fact Extractor → Fact Alignment → ADD Evidence Gate → Worthiness Gate。
- UPDATE / DUPLICATE / DELETE 还包含 Contextual Query Resolver 和 Relevance Gate；两个无 Working Memory 的查询跳过 Contextual Query Resolver，非持久事实不运行 Alignment。因此各轮调用次数不同。
- 六轮 `timing_profile.total_llm_calls` 与独立 HTTP 计数逐轮一致；主入口完整保留回复和 Memory 两部分的计时，未被 V2 生命周期清空。
- 主入口 `total_request_elapsed_seconds` 与外部测量的差异每轮小于 0.001 秒；LLM 汇总与独立 HTTP 合计的差异每轮小于 0.003 秒。
- 请求累计 **124.626 秒**；真实 LLM 客户端累计 **123.776 秒**；初始化 **19.639 秒**；完整 Smoke 进程内部累计 **144.368 秒**（含初始化及报告操作，不含另外执行的回归）。
- Ollama 服务返回的 `total_duration` 合计 **45.477 秒**，明显低于客户端 HTTP 合计。这是后续可单独排查的客户端/传输开销证据，不能把服务侧推理耗时替代为端到端延迟；本轮未修改连接配置。

## 问题、修复决策与回归

本轮没有确认需要修改业务代码的集成或 CRUD Bug。UPDATE 的 FAIL 来自新增 Smoke 判定对自然语言历史信息的限制，不能据此把正确执行的 UPDATE 改成另一种业务行为。保留该失败作为测试契约待明确项，没有修改业务代码、业务阈值、Prompt、冻结数据或原有测试断言。

质量观察：虽然所有回复都生成成功，A 的风格输出仍偏正式、冗长；Style Rewriter 的执行成功和 sender scope 正确不能证明风格质量达标。NEW 与 UPDATE 的 `constraints` 包含完整“我……”片段，而 canonical fact 改写为“联系人……”；字段没有逐字一致，说明 Prompt 中的约束保真要求还需独立验证，本次基础 CRUD 检查没有覆盖这一点。未将这些观察包装为已修复能力。

在另一个隔离子进程重跑既有五个测试模块，**40/40 PASS，0 failures，0 errors，0 skipped**。测试执行 0.406 秒，含真实 Embedding 导入的回归进程合计 15.425 秒。回归没有真实 LLM 调用，0 次隔离拦截，原文件哈希保持不变。详见 `regression.log` 与 `regression_results.json`。

下一步建议：先明确记忆是允许保留迁居历史，还是只保存当前状态，再设计能分别检测旧记录残留和错误当前城市的断言；为 constraints 保真与风格质量建立独立开发用例；单独测量客户端与 Ollama 服务侧耗时差。继续保留冻结评测，本次六条 Smoke 不代表泛化性能、并发安全或请求事务已得到验证。

## 可复现命令与产物

从项目目录执行（创建新的报告目录与临时数据，不复用本次状态）：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\33167\anaconda3\envs\memory_agent\python.exe' -B agent_v3_real_smoke.py
```

本次命令返回 **exit code 1**，与严格状态 5/6 一致。完整机器结果、各轮返回值、HTTP 原始响应、实际 Prompt、Embedding 输入与哈希证据保存在 `results.json`。回归命令为：

```powershell
& 'C:\Users\33167\anaconda3\envs\memory_agent\python.exe' -B smoke_reports\agent_v3_20261016_203533\run_isolated_regressions.py
```

回归返回 **exit code 0**。未更新 README 中旧的“尚未真实验证”描述，避免本次运行后改写被纳入保护哈希的原文件；本报告是此次运行的独立证据。
