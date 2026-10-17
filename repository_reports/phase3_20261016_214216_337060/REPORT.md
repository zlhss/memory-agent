# Agent V3 SQLite Retriever 第三阶段实际交付报告

执行：2026-10-16 21:42:16（Asia/Shanghai）。**PASS，132/132**，其中新增 29/29，原有 103/103 全部通过；0 failures、0 errors、0 skipped，命令 exit code 0。

| 测试组 | 实际结果 | 执行器测得耗时 |
|---|---|---:|
| 原有 Repository | 26/26 PASS | 0.242s |
| 原有迁移与 sender registry | 37/37 PASS | 0.891s |
| 新增 Retriever / Snapshot / 目标适配 | 29/29 PASS | 21.025s |
| 原有 Agent 回归 | 40/40 PASS | 0.394s |
| 综合流程，含真实 Embedding 加载 | PASS | 25.256s |

unittest 自身对新增 29 项报告 20.065s，执行器时间另含测试模块加载与日志处理。环境：现有 `memory_agent` Conda 环境，Python 3.12.14，SQLite 3.53.4。0 次真实 Ollama/LLM 调用；包含一次使用现有本地缓存模型的真实 Embedding 测试。

## 文件与接口

- 新增 `memory_candidate.py`：frozen `MemoryCandidate(fact_id, sender_id, content, version, score)` 与 `CandidateSnapshot(sender_id, candidates)`。
- 新增 `sqlite_memory_retriever.py`：`SQLiteMemoryRetriever(repository, embedding_model)`；`retrieve(sender_id, query, top_k=2, high_recall=False)` 与 `retrieve_fact_candidates(sender_id, atomic_fact, top_k=5)` 均返回相同不可变快照契约。
- 新增 `memory_candidate_targets.py`：`update_from_snapshot` / `delete_from_snapshot`，只从保留快照提取稳定 ID 与 version，原样返回 Repository 的操作结果。
- 新增 `test_sqlite_memory_retriever.py`：29 项测试，使用临时数据库。
- 新增 `SQLITE_RETRIEVER.md`：完整契约、调用示意、兼容规则及正式集成方案。
- 最小扩展 `memory_repository.py` / `sqlite_memory_repository.py`：提供 `list_facts()` 别名，保留原 `list()` 及 SQL 逻辑。
- 扩展 `run_repository_tests.py --include-retriever`：自动包含之前全部测试，报告写入新建 phase3 目录。

## 检索契约与兼容证据

Retriever 只依赖 `list_facts(sender_id)`，每轮读取一次 MemoryFact 列表并固定为本轮快照。没有 SQL、内容定位 ID、缓存或模型初始化。模型由调用方注入，余弦相似度使用原有 sklearn 实现。

标准检索：稳定降序排序，保留 `score >= 0.10` 后 Top-K。V2 standard fact 检索继续在普通检索/Top-K 后保留 `score >= 0.40`。当 kind 为 persistent_change 或 operation_hint 为 update/end 时，high_recall 不做相关性过滤，排序后直接 Top-K，零分和负分也保留。同分保持原始新增顺序，重复文本保留不同 ID。

对照测试在临时 JSON 下执行未修改的 `memory_search.retrieve`，以及从现有 `memory_v2.py` AST 中提取的原 `retrieve_fact_candidates` 函数和阈值；没有导入其模型/LLM 初始化路径。多个 Top-K、0.10/0.40 边界和 standard/high_recall 返回的 score/content 与旧实现逐项一致。旧 JSON 生产模块没有修改。

## target_index 安全性证据

快照防御性复制输入为 tuple，候选和快照均不可变。`contents` 只提供与候选一一对应的文本 tuple。正式调用 LLM 时应在排序/过滤完成后固定快照，并原样传入 `list(snapshot.contents)`；需将该快照与本次 fact-specific LLM 调用绑定，不能使用顶层回复候选或后续快照解释它。

`resolve_target` 要求 sender 匹配，index 为精确 Python int，且非负并在原始范围内。bool、float、字符串、NumPy integer、越界值全部拒绝。映射和写入 helper 不读取 Retriever、不重跑 Embedding、不获取最新版本；没有接收 LLM fact_id 或 expected_version 的接口。

实际隔离用例确认：

- 后续查询将两个候选排序反转；保留原快照的 index 0 仍更新原来的 ID，另一个记录不受影响。解析期间把重读/重编码设置为报错，操作仍正常成功。
- 两条相同文本分别有不同 ID；index 1 精确更新或删除第二个 ID，第一条保持不变。
- 非法 index 和错误 sender 没有产生写入。
- 另一连接更新目标后，旧快照 UPDATE 和 DELETE 均返回 VERSION_CONFLICT，不强制覆盖、不重试。
- 原目标被删除、仍有同文另一条时，返回 NOT_FOUND，不重定位同文记录。
- Embedding 计算期间目标发生更新，候选仍保留读取时的 version；随后的写入报告版本冲突。
- 伪造 LLM fact_id/version 不能传入 helper；仅提取合法 index 时，从原快照取得真实第二条 ID/version。

这些 helper 只负责存储目标身份与版本，不执行或替代 Alignment、Evidence、Worthiness、歧义和 destructive-target-support 业务检查。正式接入时需在原有 gates 授权后调用。

## 真实 Embedding 验证

从现有缓存离线加载真实 `paraphrase-multilingual-MiniLM-L12-v2` SentenceTransformer：

1. 在临时 SQLite 为 A 新增“联系人住在南京”和“联系人喜欢摄影”，为 B 新增厦门事实。
2. 以“联系人住在南京”检索 A，Top-1 命中南京事实的稳定 ID，score > 0.99。
3. 将同一 ID 更新为“联系人住在杭州”，version=2；下一次检索命中更新后的文本及版本。
4. 原快照仍为南京/version=1，其条件写入返回 VERSION_CONFLICT。

此外，确定性向量测试确认新增/更新即刻可见、每次重新读取、空 namespace 不编码、只编码当前 sender 的内容。确定性测试仍使用真实 SQLite 和余弦计算，未调整业务阈值或删除断言。

## 隔离与未修改范围

综合执行器确认测试临时数据库全部清理、baseline 临时数据目录清理、0 次隔离拦截；运行前后所有已存在文件哈希相同。与前次保存的原始业务/数据哈希比对，`main.py`、`memory_v2.py`、`memory_search.py`、Prompt、三个真实 JSON 及冻结评测/历史结果仍未变化。

本轮没有迁移真实 JSON、没有 SQLite 生产切换、没有修改 Alignment 决策、没有双写、没有新增缓存。独立模块导入也不加载 Agent、旧 JSON 存储或 SentenceTransformer。

## 下一阶段正式集成

先在临时 SQLite 注入 Repository/Retriever；为每个 atomic fact 保留与 Prompt 完全对应的内部快照。原有 LLM 输入继续只包含编号和文本，所有 schema/evidence/worthiness、歧义及 destructive-support 阈值保持原语义。

只在 CRUD 适配点使用快照 ID/version，将冲突/不存在明确写入版本化诊断。DUPLICATE/AMBIGUOUS 继续不写，重复目标保护需使用 ID 并补集成用例；不能用新检索或最新版本绕过旧快照失效。

保留原来的回复生成顺序、V1/V2 生命周期互斥、整轮计时、反馈闭环和当前部分成功异常契约；请求事务语义变化另行决定。通过临时集成及真实 Ollama Smoke 后，才单独授权实际迁移和后端切换。正式迁移前暂停旧系统写入并等待在途请求完成，避免同一联系人混用 JSON 与 SQLite 写入。

复现命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\33167\anaconda3\envs\memory_agent\python.exe' -B run_repository_tests.py --include-retriever
```

逐项日志为 `repository.log`、`migration.log`、`retriever.log`、`baseline.log`；机器证据为本目录 `results.json`。每次综合运行创建新目录，不覆盖历史结果。
