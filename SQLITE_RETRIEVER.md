# Agent V3 SQLite Retriever 与目标快照：第三阶段

本阶段只增加独立检索和目标映射适配层。`main.py`、`memory_v2.py`、`memory_search.py`、LLM Prompt、模型和 Alignment 判断没有修改；默认 JSON 路径继续运行。不迁移真实 JSON、不启用 SQLite 生产存储、不添加缓存，也不改写冻结评测或历史结果。

## 修改文件与接口

| 文件 | 接口与职责 |
|---|---|
| `memory_candidate.py` | 不可变 `MemoryCandidate`、`CandidateSnapshot` 和安全 index 解析 |
| `sqlite_memory_retriever.py` | `SQLiteMemoryRetriever(repository, embedding_model)`，通过 Repository 读取事实并计算候选 |
| `memory_candidate_targets.py` | `update_from_snapshot` / `delete_from_snapshot`，将选中快照的 ID/version 交给 Repository |
| `memory_repository.py` / `sqlite_memory_repository.py` | 添加 `list_facts(sender_id)` 别名，保留原有 `list()` 契约和 SQL 实现 |
| `test_sqlite_memory_retriever.py` | 29 项隔离测试，包含真实 SQLite、真实余弦相似度及真实 Embedding 验证 |
| `run_repository_tests.py` | 添加 `--include-retriever`，运行第三阶段及之前全部测试，生成新的 phase3 报告 |

### 候选与快照

```python
MemoryCandidate(
    fact_id: str,
    sender_id: str,
    content: str,
    version: int,
    score: float,
)
CandidateSnapshot(sender_id: str, candidates: tuple[MemoryCandidate, ...])
```

两者均为 frozen dataclass；快照会把输入列表或生成器防御性复制成 tuple。字段要求非空字符串、正整数 version 和有限实数 score，不转换内容或 namespace。快照拒绝混合联系人、错误条目类型或重复 Fact ID；**不同 ID 的重复文本完整保留**。

`snapshot.contents` 返回与 candidates 一一对应的不可变文本 tuple，只包含 content。调用现有 Alignment 时可以用 `list(snapshot.contents)`，不增加数据库 ID/version 到 Prompt。该列表应原样传入，不能在形成快照后再修改排序或过滤；若调用前还需变换候选，必须先完成变换，再创建最终快照。

`snapshot.resolve_target(sender_id, target_index) -> MemoryCandidate` 只访问此快照，不调用 Repository 或 Retriever。只接受 `type(index) is int` 且 `0 <= index < len(candidates)`；bool、负数、浮点数、字符串、NumPy integer、越界值及空列表上的 index 均明确抛 `ValueError`。调用 sender 必须等于快照 sender。

每次 fact-specific 检索保留独立快照，并将其与对应那次 LLM 调用绑定。后续检索得到的新快照只能解释新调用的 index，不能替换旧快照。顶层回复相关记忆与各 atomic fact 的候选是不同索引空间，不能交叉解释。

### Retriever

```python
retriever.retrieve(
    sender_id, query, top_k=2, *, high_recall=False,
) -> CandidateSnapshot

retriever.retrieve_fact_candidates(
    sender_id, atomic_fact, top_k=5,
) -> CandidateSnapshot
```

Retriever 每次只调用一次 `repository.list_facts(sender_id)`，将返回的不可变 MemoryFact 固定为本轮数据快照。先检查所有事实的类型和 sender，再编码文本与查询，使用原有 `sklearn.metrics.pairwise.cosine_similarity` 和稳定降序排序。没有 SQL、按内容回查 ID、缓存或模型构造。支持只有 `list_facts` 的只读接口；真实 SQLite Repository 提供的该方法委托原有 `list()`。

注入模型应是现有 `paraphrase-multilingual-MiniLM-L12-v2` 的 SentenceTransformer 实例；未来组装主流程时复用已有模型对象。本模块导入不读取 JSON、不加载 SentenceTransformer 或模型权重，也不导入 Agent/LLM 模块。

| 检索入口/模式 | 与旧实现相同的排序及阈值语义 |
|---|---|
| `retrieve` standard | 对全部 sender facts 编码，按 cosine score 稳定降序，保留 `score >= 0.10`，再取 Top-K |
| `retrieve` high_recall | 相同计算与排序，不做相关性过滤，直接取 Top-K，保留 0 分与负分 |
| `retrieve_fact_candidates` standard | 先做普通 standard 检索和 Top-K，然后保留 `score >= 0.40` |
| `retrieve_fact_candidates` high_recall | 当 kind 为 `persistent_change` 或 operation_hint 为 `update/end`，使用上述 high_recall |

同分保持 Repository 的原始新增顺序，重复内容不去重。两种模式都返回相同 CandidateSnapshot/MemoryCandidate 契约，不通过不同数据形状隐藏 ID/version。代码中的 0.10/0.40 镜像当前旧实现；对照测试直接执行旧检索代码，避免静默漂移。本轮不修改旧常量或引入新的业务阈值。

新增接口要求非空 sender/query、非负且非 bool 的整数 top_k，以及真正 bool 的 high_recall。top_k=0、未知 sender 或空 sender namespace 返回空快照，不调用 Embedding。模型返回的矩阵尺寸必须与固定事实数量、单条 query 对应；异常明确传播，不变成成功的空列表。旧接口对非法参数的历史行为没有修改。

### 目标写入适配

```python
update_from_snapshot(
    repository, snapshot, *, sender_id, target_index, content,
) -> MutationResult

delete_from_snapshot(
    repository, snapshot, *, sender_id, target_index,
) -> MutationResult
```

两个适配函数先验证 sender/index，再直接使用原快照的 fact_id 和 version 调用 Repository 条件写入。接口不接受 LLM 生成的 fact_id 或 expected_version，不重新检索、不从 content 定位事实、不提取最新版本来绕过冲突，也不自动重试或回退 JSON。

Repository 的 SUCCESS / VERSION_CONFLICT / NOT_FOUND 结果原样返回，调用方必须检查 status。目标被更新返回 VERSION_CONFLICT；目标被删除返回 NOT_FOUND，即使数据库仍有内容相同的另一条记录也不能改写它。真实存储错误原样抛出。

这些适配函数**只负责目标身份和版本**。正式接入时仍需先通过现有 Atomic Fact/schema/evidence/worthiness、Alignment、歧义与 destructive-support gates；它们不会替代这些业务检查或根据 LLM 原始 decision 自行执行操作。UPDATE 的 content 应来自已验证的 canonical atomic fact。

## 独立使用示意

```python
# repository 和 embedding_model 由调用方提供，不在这里隐式创建。
retriever = SQLiteMemoryRetriever(repository, embedding_model)
snapshot = retriever.retrieve_fact_candidates(sender_id, validated_atomic_fact)

# 下一阶段才在主流程接线：现有函数/Prompt 不变。
alignment = existing_alignment_function(
    validated_atomic_fact, list(snapshot.contents),
)

# 先运行原有业务 gates，再根据已授权的 UPDATE 分支调用：
result = update_from_snapshot(
    repository, snapshot,
    sender_id=sender_id,
    target_index=alignment["target_index"],
    content=validated_atomic_fact["fact"],
)
# 此处必须处理 result.status，不能把版本冲突当作写入成功。
```

这是下一阶段的接口示意，本轮没有增加真实 LLM 调用或主流程接线。

## 测试与实际执行

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\33167\anaconda3\envs\memory_agent\python.exe' -B run_repository_tests.py --include-retriever
```

该开关自动包含第二阶段迁移测试，输出新的 `repository_reports/phase3_*/`。原有 `--include-migration` 和默认入口继续可用。新增测试覆盖：排序、同分顺序、Top-K、0.10/0.40 边界、标准/high_recall 与旧函数逐项对照、重复文本不同 ID、空联系人、只读 Repository 契约、无缓存、新增/更新可见、冻结后重排、非法 index、跨 sender、过期版本、删除目标、Embedding 期间数据变化、伪造 LLM ID/version、错误传播和无导入副作用。

确定性测试只在 Embedding seam 提供向量，余弦计算和 SQLite 均使用真实实现。真实模型测试从现有本地缓存加载相同 SentenceTransformer，验证新增南京事实、更新为杭州后版本=2 的检索，以及旧快照写入冲突。没有用 LLM Judge 或改变阈值使测试通过。

## 下一阶段正式集成方案

1. 显式注入 Repository 和 Retriever，存储后端选择与 V1/V2 算法选择分开管理；JSON 保持默认，先使用临时 SQLite 完成集成回归。
2. 每个 atomic fact 的检索、阈值过滤、Prompt 和 Alignment 共用同一快照。将其放入本轮内部上下文；诊断可序列化 ID/version/score，但不能从 LLM 响应重建可信快照。
3. 保留当前 atomic extraction、Alignment Prompt、相关性/歧义/target-support gates 和 NEW 的 Evidence/Worthiness。只在最终 CRUD 执行点把原候选字符串目标替换成快照中的 ID/version。DUPLICATE/AMBIGUOUS 继续不写入。
4. 在 `memory_v2_details` 中明确报告冲突/不存在及目标快照；不强制覆盖、不重解释旧 index、不在 SQLite 部分成功后回退 JSON。为同一消息重复触及目标与内容相同但 ID 不同的记录建立集成用例，明确 used_targets 的 ID 语义。
5. 保持回复先生成、每轮只执行一个生命周期、外层计时、人工反馈和现有部分写入异常契约。整轮事务是否改变现有部分成功语义须另行决定，不因本轮 helper 存在而悄悄修改，也不在 LLM 推理期间持有数据库写锁。
6. 通过临时数据库集成测试后再运行真实 Ollama Smoke。生产迁移/切换仍需单独授权：暂停旧系统 Memory 写入、等待在途写入完成、执行迁移校验、保留 JSON 只读备份，并避免同一联系人混用两个写入后端。
