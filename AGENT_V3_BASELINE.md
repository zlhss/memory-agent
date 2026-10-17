**Agent V3：Response Pipeline 兼容性基线**

验证日期：2026-10-14，Asia/Shanghai。本轮只修复测试环境和测试替身、增加回归测试；未接入 Memory V2。

1. **环境修复与复现方式**

原默认解释器为 `C:\Users\33167\anaconda3\python.exe`（Python 3.13.5、Conda base），缺少 `sentence-transformers`。两组原有测试在该环境均导入失败，未运行实际用例。

复用现有项目专用环境 `C:\Users\33167\anaconda3\envs\memory_agent\python.exe`（Python 3.12.14）。没有创建新的 `.venv`，没有向 base 或全局 Python 安装包。

| 依赖 | 本次测试环境版本 |
| --- | --- |
| requests | 2.34.2 |
| scikit-learn | 1.9.1 |
| sentence-transformers | 6.0.1 |
| psutil | 7.2.2 |
| torch | 2.14.0 |
| transformers | 5.17.0 |
| huggingface-hub | 1.31.0 |

前三个直接依赖已存在；本次仅向 `memory_agent` 环境补装 `psutil`。`requirements.txt` 保持不变，`python -m pip check` 返回 `No broken requirements found.`。所有直接依赖的导入位置均属于该专用环境。

模型 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 已缓存于 `C:\Users\33167\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`。快照 revision 为 `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`，包含约 470 MB 的 safetensors 权重、tokenizer 和 pooling 配置。设置离线模式后，真实 `memory_search` 导入成功，CPU 编码单条文本得到 `(1, 384)` 向量；未下载 Embedding 模型。

可从项目目录重复执行：

```powershell
.\run_baseline_tests.ps1
```

脚本优先使用项目 `.venv`，否则使用上述现有 `memory_agent` 环境，也接受 `-PythonPath` 显式指定解释器。它设置进程级离线和 UTF-8 环境变量，逐组执行全部测试并汇总退出码；任何一组失败都返回非零，执行后恢复环境变量。它不会安装依赖或修改全局 Python。默认 shell 的 `python` 仍可能指向 base，手动执行时应先 `conda activate memory_agent`。

2. **实际测试及修复依据**

| 最终执行命令（由脚本选定专用解释器） | 用例数 | 通过 | 错误/失败/跳过 |
| --- | ---: | ---: | --- |
| python -m unittest test_contextual_memory_flow -v | 2 | 2 | 0 / 0 / 0 |
| python -m unittest test_ollama_timing -v | 1 | 1 | 0 / 0 / 0 |
| python -m unittest test_response_pipeline_mock -v | 7 | 7 | 0 / 0 / 0 |
| 合计 | 10 | 10 | 0 / 0 / 0 |

另外，专用解释器执行 `python -m py_compile main.py test_contextual_memory_flow.py test_response_pipeline_mock.py` 通过。

在正确环境首次运行上下文测试时，结果为 1 项通过、1 项错误：测试替身 `_fake_ollama(prompt)` 不接受生产调用已有的 `stage` 关键字。使用上一轮保存的重构前 `main.py` 原始源码在内存中重放同一测试，也得到相同错误；该问题先于函数提取存在。

对 `test_contextual_memory_flow.py` 的修复仅限 `_fake_ollama`：签名匹配 `call_ollama(prompt, stage="Unspecified")`，校验各 Prompt 对应的 stage，并为已有 Action Coverage Checker 返回完整覆盖情况下的空 `missing_actions`。没有删除、放宽原有业务断言，没有跳过测试，没有修改生产调用或 Prompt。

新增的 7 项测试加载真实 `main.py`，仅在加载期间替换外部服务模块，并恢复 `sys.modules`；不读取模型、不调用 Ollama、不修改数据文件。验证内容包括：四个回复步骤的顺序与完整参数、四项回复产物、空风格样本路径、主入口所有返回字段、每轮一次 Memory 管理及写入、四个阶段的原始异常对象传播、回复失败时不进入 Memory 管理，以及生成阶段不更新 Working Memory/feedback。独立测试亦在原 base Python 中通过。

本轮文件变更：

- `test_contextual_memory_flow.py`：修复已有测试替身的接口和阶段响应。
- `test_response_pipeline_mock.py`：新增独立回归测试。
- `run_baseline_tests.ps1`：新增正确选择解释器的重复验证入口。
- `AGENT_V3_BASELINE.md`：保存本报告。

原有文件逐文件 SHA-256 对比确认：只有上述测试替身所在文件发生变化；生产代码、`requirements.txt`、原计时测试、历史评测脚本/数据/结果、`memories.json`、`feedback.json` 和 `style_samples.json` 均未修改。忽略运行测试生成的 `__pycache__`。

3. **尚未验证的内容**

没有执行真实 Ollama 请求、实际回复质量/耗时评估、完整 CLI 人工 accept/edit/reject 交互或 Memory V2 动态集成。现有计时测试只验证空记录汇总；Mock 回归测试验证入口复位和错误汇总行为，不证明真实 LLM 计时准确性。未重新运行任何冻结评测，也未改写历史结果。本基线不能替代后续 V2 集成测试。

4. **Memory V2 静态分析与接口差异**

`main._process_user_message_impl()` 中，相关性过滤后调用 `build_personalized_reply()`（约 165 行），成功返回后才执行第 5–8 阶段（约 179–619 行）：Memory Manager/Action Coverage → 写入验证 → 确定性 CRUD → Consistency。V1 捕获单次写入、Consistency 检索/检查/写入异常并记录结果。`finalize_response()` 在人工反馈后才写入 feedback 和真实 Working Memory；它不再执行 Memory 生命周期。

`memory_v2.py` 在 13 行导入 `main`，通过 `main.get_working_memory()` 获取对话（509 行），并在 atomic fact/alignment/NEW 验证中调用 `main.is_valid_contact_memory()`（186、358、568 行）。反向在 `main` 顶层导入 V2 会形成循环依赖。仅改成延迟导入仍需处理 `python main.py` 以 `__main__` 运行时又导入名为 `main` 的模块，产生两套 Working Memory 注册表的问题。

`main.process_user_message()` 在 644 行复位 Ollama 全局计时及 Prompt 记录；`memory_v2.process_user_message_v2()` 在 639 行再次复位。直接嵌套公开入口会清掉上下文解析、回复规划、草稿等已经记录的调用，同时输出两份请求汇总。当前 V2 内部 `_process_user_message_v2_impl()` 不复位计时。

| 返回内容 | main / V1 | 独立 V2 |
| --- | --- | --- |
| sender_id、incoming、processed_actions | 有 | 有 |
| contextual_query、candidate_memories、relevant_memories | 回复编排提供 | 不提供，fact-specific retrieval 不等价于回复检索 |
| style_samples、plan、draft、personalized_reply | 有 | 不生成回复 |
| action_results | action / success / error | action / success；未捕获的执行异常向上抛出 |
| processed_actions 中的 target_index | UPDATE/DELETE 保留对 relevant_memories 的 index，并补真实 target | 执行动作只保留真实 target；alignment index 属于各 fact 的候选集 |
| consistency_results | 实际执行的次级动作结果 | 受限一致性检查的说明记录，内含 actions=[]，不是执行结果 |
| consistency_errors | 独立错误列表 | 不提供 |
| V2 专属字段 | 无 | pipeline_version、working_messages、atomic_facts、fact_results、coverage、final_memories |
| timing_profile | main 外层添加 | V2 独立入口添加；内部实现不添加 |

不能用一个字典覆盖另一个字典来完成集成。必须保留回复字段，明确适配 action/error 结构；不得把 V2 说明记录当作 V1 的 Consistency 执行结果，也不得把每个 fact 的 index 当作回复检索列表的 index。V2 出错时可能已有前面的 fact 写入成功，不能声称请求级回滚或自动回退 V1。

5. **推荐的最小风险接入方案（本轮未实现）**

- `main.py`：将第 5–8 阶段提取为 V1 生命周期函数；在回复生成成功后设置唯一后端分支，只调用 V1 或 V2 其中一个。保留现有 `process_user_message(sender_id, user_text)` 默认 V1 行为供历史评测使用，V3 通过显式配置/入口启用 V2；不要在同一消息中串行执行两套，也不要在 V2 已有部分写入后回退执行 V1。
- 新增轻量共享上下文模块（例如 `agent_context.py`）：仅承接 Working Memory 注册表、get/reset 和现有 contact schema 判断；`main` 保留兼容导出，V2 直接引用共享模块。这样同时消除循环导入和 `__main__` 双注册表问题，不改业务判断。
- `memory_v2.py`：从现有内部实现暴露不复位、不汇总计时的核心接口，例如 `run_memory_lifecycle_v2(sender_id, user_text, working_messages)`。集成时显式传入与回复相同的对话快照。保留独立 `process_user_message_v2(sender_id, user_text)` 包装和其评测契约，由它自行取共享上下文并负责独立调用的计时；Agent 内部只调用核心接口。
- `main.py` 的结果组装：保留原回复字段和已执行动作字段；V2 的 facts/alignment/coverage/一致性说明放入明确的独立调试字段。对返回成功/失败动作补齐旧接口需要的 error 字段时，不虚构错误原因；对未捕获异常制定显式策略并继续向上报告，不转换为空列表掩盖失败。V1/V2 的 target_index 语义不能直接互换。
- `finalize_response()`、`demo_cli.py`：保留现有消息/回复/反馈契约和时机；可切换后端的展示或配置另行接线，不在 CLI 中实现 Memory 算法。
- 新增独立 V3 集成测试，覆盖后端互斥、生成失败零 Memory 写入、成功每条消息一次生命周期、完整计时不被清空、同一 Working Memory 快照、字段适配、部分写入失败不回退、sender 隔离和人工反馈闭环。保持冻结评测不变。

执行顺序建议：共享上下文去环 → 暴露 V2 非计时核心 → 增加独立接入测试 → 显式启用 V3/V2。回复前检索和回复成功后写 Memory 的现有顺序保持不变；需要调整该顺序时应另立行为变更测试。
