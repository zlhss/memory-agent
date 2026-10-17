# 项目事实清单

本文档只记录能够由代码或已完成实验支持的事实，用于后续准备面试和撰写简历，不使用营销性表述。

## 最终架构

- 项目是一个本地运行的多联系人对话 Agent 研究原型。
- `sender_id` 是 Working Memory、Long-Term Memory、Style Samples 和 feedback 的隔离 namespace。
- Long-Term Memory 在 `memories.json` 中使用 `{sender_id: [memory strings]}` 结构。
- Working Memory 是进程内、按 sender 分组、保留有限最近消息的映射。
- Style Memory 使用独立 JSON 样本存储，不与事实记忆混合。
- 个性化回复流程为：Context Resolution → Long-Term Retrieval → Relevance Filtering → Planner → Neutral Draft → Style Retrieval → Style Rewrite → Human Feedback。
- `main.process_user_message` 是现有交互式个性化编排入口，其中仍包含 V1 memory lifecycle。
- `memory_v2.process_user_message_v2` 是独立冻结并完成评估的记忆生命周期实现。它不生成个性化回复，也尚未接入 `main.py`。
- `demo_cli.py` 只是对 `main` 公共函数的展示层包装，不包含 Prompt 或 Memory 算法。

## 模型与运行环境

- 文本 LLM：通过本地 Ollama 运行的 `qwen2.5:7b-instruct`。
- Ollama 调用使用 JSON 格式、`temperature: 0` 和 300 秒请求 timeout。
- Embedding 模型：通过 SentenceTransformers 使用 `paraphrase-multilingual-MiniLM-L12-v2`。
- 模型迁移来自真实性能实验，而不是主观选择。
- Qwen2.5 7B control benchmark：15/15 完成，15/15 JSON 合法，平均 3.544s，中位数 3.591s，最大 5.693s，没有调用超过 10 秒。
- Qwen3-VL 8B 及其 instruct 版本出现严重随机延迟尖峰，包括 streaming 持续输出 reasoning、超过 60 秒仍无最终 response。
- 项目收尾时本地验证环境为 Python 3.10.8 与 Ollama 0.34.4。

## Memory V1

- V1 使用 sender-aware contextual retrieval，再由多动作 Memory Manager 处理当前消息。
- 支持 ADD、UPDATE、DELETE、Worthiness、Duplicate、Target Resolution/Index、Action Coverage、ADD Evidence、Schema Validation 和 Consistency。
- 为 mixed facts 不断扩展 Memory Manager Prompt 曾引入 no-op knowledge question 回归。
- 后续改为窄职责组件：Action Coverage Checker、ADD Evidence Gate 和确定性的 Contact Memory Schema Guard。
- ADD / UPDATE 后的 memory 必须是以“联系人”开头的非空字符串。

## Memory V2

- V2 是架构重构，不是继续扩展 V1 Prompt。
- 先提取 Atomic Facts，再进行 Memory 决策。
- 每个 persistent fact 独立执行 embedding retrieval。
- Fact Alignment 返回分类决策和 candidate index。
- 支持 NEW、DUPLICATE、UPDATE、DELETE 和 AMBIGUOUS。
- Target 证据不足或 destructive support 不足时 abstain。
- CRUD 保持确定执行和 sender-scoped。
- 字符串存储无法证明记忆之间的依赖，因此 Consistency 权限受限。

## 为什么引入 Atomic Fact

- Whole-message management 会压缩独立事实并丢失限制条件。
- 一条消息可能同时包含新增事实、状态变化、瞬时细节、否定和指代。
- Atomic extraction 让每条 persistent fact 拥有独立的 retrieval 与 alignment。
- 这样可以观察遗漏覆盖，并减少一个 action 影响无关事实的风险。

## 为什么歧义时必须 Abstain

- UPDATE 与 DELETE 是破坏性操作。
- 关键词或主题重合不足以证明两条 memory 描述同一当前状态。
- 多个候选都可能匹配时，强行选择会破坏无关 memory。
- 当前策略接受漏写风险，以避免没有证据的状态修改。

## 为什么限制 Consistency 权限

- 当前 memory 只是字符串，没有 provenance 或显式依赖图。
- 语义相关不等于一条 memory 依赖另一条 memory。
- 早期较宽的 Consistency 行为可能修改相关但独立的事实。
- V2 因此不根据模糊相关性生成次级操作；当前实现返回受限的 no-action consistency result。

## 冻结的 Memory Benchmark

### Memory V1 Final Unseen Evaluation

- Case 数：20。
- PASS：11/20。
- Pass rate：55%。
- Behavior failures：9。
- Cross-sender pollution：0。
- 平均延迟：21.041s。
- 平均 LLM calls：6.4。
- Benchmark SHA-256：`1e6542023ab2f93b86643e1710ea261f7f1844a6d7a9eb8cff9102a0cf3ce1be`。

### Memory V2 Final Unseen Evaluation

- Case 数：20。
- PASS：13/20。
- Pass rate：65%。
- Behavior failures：7。
- Destructive failures：0。
- Safe-abstain failures：4。
- Program errors：0。
- Environment errors：0。
- Cross-sender pollution：0。
- 平均延迟：12.531s。
- 平均 LLM calls：2.85。
- Benchmark SHA-256：`cb22b32dad5b057067967c877f23f5de419ae613c00a92b047f28e70b4071754`。

### V2 相对 V1 的实测变化

- Unseen pass rate：增加 10 个百分点。
- 平均延迟：降低 40.4%。
- 平均 LLM calls：减少 55.5%。
- V2 destructive failures：0。

## 个性化回复评估

- Development set：15 条 synthetic cases，没有将真实私人聊天复制到 benchmark。
- 三个模式：
  - Generic：只使用当前消息和理解它所必需的可见对话。
  - Memory-only：sender-scoped context/retrieval、Planner 和 Draft，不执行 Style Rewrite。
  - Full：复用同一 Memory-only Draft，再执行 sender-scoped Style Retrieval 与 Style Rewrite。
- 平均 Agent 延迟：
  - Generic：3.969s。
  - Memory-only：10.854s。
  - Full：14.692s。
- 平均 LLM calls：
  - Generic：1.00。
  - Memory-only：3.07。
  - Full：4.07。
- Cross-sender leakage：0。
- Unsupported personalization：0。
- 自动 LLM Judge 退化为重复占位输出，因此表面上的 100% factual/semantic 结果不属于有效证据。

## 单 Reviewer 盲评结果

适用范围：15 条 synthetic development cases、一个 reviewer。

- “更愿意发送”：Full 9，Memory-only 0，Tie 6。
- Style Samples match：Full 3，Memory-only 0，Tie 12。
- Meaning preservation：Full 3，Memory-only 1，Tie 11。
- Naturalness：Full 3，Memory-only 0，Tie 12。
- 以上结果不能外推为普遍人类偏好。

## 为什么最终保留 Style Rewriter

- 它将内容规划和表达方式明确分离。
- 单 reviewer development 盲评中，“更愿意发送”指标为 Full 9、Memory-only 0、Tie 6。
- Development set 上，Full 将 embedding style-similarity proxy 从 0.4090 提高到 0.4486。
- 相比 Memory-only，实测代价约为一次额外 LLM 调用和 3.8 秒。
- 它并非默认安全：一个 development case 完整复制了 12 字 Style Sample，另一个 rewrite 删除了 Draft 中的实质建议。
- Style Rewrite 可能改变内容或不适合当前情境，因此最终仍保留人工确认。

## 已知限制

- Memory V2 的 final unseen pass rate 为 65%，生命周期处理仍不完美。
- V2 final 失败包括保守弃权、错误 Duplicate、不完整状态迁移和 extraction omission。
- JSON 存储不支持事务、schema migration、认证、加密或并发写入协调。
- Working Memory 位于进程内。
- Memory 字符串没有 provenance、confidence、timestamp 或 dependency metadata。
- 交互回复 CLI 仍使用 V1 memory lifecycle；冻结后的 V2 integration 尚未进行。
- 个性化结果来自 synthetic cases 与单个 reviewer。
- 自动个性化 Judge 不可靠。
- Style embedding similarity 只是 proxy，不能替代人工评估。
- Style Rewrite 可能复制样本表达或改变语义。
- 项目不包含 Web UI、数据库、Docker、云部署或消息平台接入。

## 失败或被否决的方案

- 因为严重延迟不稳定，Qwen3-VL 8B reasoning 与 instruct 版本没有继续用于文本 pipeline。
- 继续向 Memory Manager Prompt 增加职责造成 regression，并未修复 mixed-action case。
- Free-form UPDATE/DELETE target 在合理释义下解析失败，后续改为 candidate-index selection。
- Broad Consistency 因无法从语义相关证明依赖关系而被限制。
- 自动聚合式 LLM Judge 返回模板化且无区分度的结果，因此不再作为主要证据。
- 没有使用 Style exact match；embedding 与结构指标只作为 proxy。

## 关键工程决策

- 使用独立 sender namespace，而不是在每条 memory 中重复名字。
- Style Memory 与事实型 Long-Term Memory 分离。
- 所有结构化 LLM 输出必须在状态变化前完成校验。
- ADD 先经过 Evidence 与 Contact Schema，再进入 Worthiness 和 Duplicate。
- Destructive action 只能通过 index 选择已有候选。
- 在不确定时优先 abstain。
- Judge latency 不计入实际 Agent inference latency。
- 保留 accept / edit / reject，让人工成为回复的最终决策者。
- Final unseen benchmark 在执行前冻结，测试结果不能反向用于调参。

