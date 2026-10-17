# Agent V3 SQLite Repository：第一阶段

本阶段新增独立存储接口与 SQLite 实现。现有 `main.py`、`memory_v2.py`、`memory_search.py`、LLM Prompt 和算法均未接入或修改；没有迁移真实 JSON 数据，也没有修改冻结评测和历史结果。仅使用 Python 标准库，无需新增依赖。

## 文件与接口

- `memory_repository.py`：不可变 `MemoryFact`、`MutationStatus`、`MutationResult` 和 `MemoryRepository` Protocol。
- `sqlite_memory_repository.py`：显式接收数据库路径的 `SQLiteMemoryRepository`，实现 CRUD、乐观版本检查和事务。
- `test_sqlite_memory_repository.py`：临时 SQLite 文件上的存储契约测试，不导入 Agent、Embedding 或 Ollama。
- `run_repository_tests.py`：隔离运行新增测试及原有 40 项回归，输出独立报告，并校验原有文件哈希。

`MemoryFact(id: str, sender_id: str, content: str, version: int)` 是不可变快照。`id` 为 Repository 生成的 UUID4，更新保持 ID 不变；删除后重新新增使用新 ID。初始 version=1，每次成功 UPDATE 加 1，即使内容相同。保留原始字符串，不做 trim、语义转换或联系人前缀修正；仅拒绝空白字符串和错误参数类型。Repository 允许相同内容拥有不同 ID，避免把记忆算法的判重规则引入持久层。

| 方法 | 返回与语义 |
|---|---|
| `list(sender_id) -> list[MemoryFact]` | 仅当前 sender，按新增顺序排列；未知 sender 返回空列表 |
| `add(sender_id, content) -> MemoryFact` | 创建新 ID，version=1 |
| `update(sender_id, fact_id, content, *, expected_version) -> MutationResult` | sender、ID、版本全部匹配后更新；返回更新后快照 |
| `delete(sender_id, fact_id, *, expected_version) -> MutationResult` | sender、ID、版本全部匹配后删除；返回删除前快照 |
| `transaction()` | 异常回滚，正常退出提交；支持嵌套保存点 |
| `close()` | 关闭连接，可重复调用；事务内关闭明确报错 |

`MutationResult.status` 区分 `SUCCESS`、`VERSION_CONFLICT` 和 `NOT_FOUND`。失败结果的 `fact=None`。不存在的 ID、已删除 ID，以及属于其他 sender 的 ID 都返回 `NOT_FOUND`，不泄露其他联系人的版本。只有当前 sender 拥有记录且版本不匹配时返回 `VERSION_CONFLICT`。非法参数抛 `ValueError`；遗漏必填参数抛 `TypeError`；SQLite 锁超时、约束失败等真实存储错误原样向上抛出，不伪装成“不存在”。

示例仅演示接口，请使用临时目录中的显式路径：

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from memory_repository import MutationStatus
from sqlite_memory_repository import SQLiteMemoryRepository

with TemporaryDirectory() as directory:
    with SQLiteMemoryRepository(Path(directory) / "facts.sqlite3") as repository:
        fact = repository.add("demo_contact", "联系人住在南京")
        result = repository.update(
            "demo_contact", fact.id, "联系人住在厦门",
            expected_version=fact.version,
        )
        assert result.status == MutationStatus.SUCCESS
        assert result.fact.id == fact.id
        assert result.fact.version == 2
```

## SQLite 表结构与索引

```sql
CREATE TABLE memory_facts (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE CHECK (length(trim(id)) > 0),
    sender_id TEXT NOT NULL CHECK (length(trim(sender_id)) > 0),
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    version INTEGER NOT NULL DEFAULT 1
        CHECK (typeof(version) = 'integer' AND version >= 1)
);
CREATE INDEX idx_memory_facts_sender_sequence
    ON memory_facts (sender_id, sequence);
```

`id` 的 UNIQUE 约束产生唯一索引。内部 `sequence` 只用于稳定新增顺序，不暴露为 Fact ID；这让未来 JSON 导入保留每个 sender 原始列表顺序，也避免改变相似度相同候选的先后次序。联系人列表使用 `(sender_id, sequence)` 索引；UPDATE/DELETE 可通过唯一 ID 快速定位，再校验 namespace 与版本。所有数据参数均通过 `?` 绑定；保存点名是固定内部标识，SQL 不拼接外部输入。

每次独立写入都在 `BEGIN IMMEDIATE` 事务中进行。UPDATE 的 SQL 同时包含 `sender_id = ? AND id = ? AND version = ?`；未命中后在同一个写事务中检查该 sender 的记录是否存在，以区分版本冲突和不存在，避免检查/写入之间的竞争。DELETE 在同一个写事务内取得版本匹配的快照并条件删除。多个 Repository 实例共享数据库时，SQLite 串行化写入；相同旧版本只能有一个成功写入者。

显式事务中的操作通过保存点参与外层事务，不会逐条独立提交。内层异常可以被外层捕获，仅回滚内层；外层失败仍回滚此前成功释放的内层保存点。SQLite 自行回滚整个事务时保留其原始异常；即使调用方捕获该异常，失效事务内后续操作和外层正常提交都会明确报错，退出外层作用域后可重新使用连接。

**冲突/不存在是返回值，不自动触发整组回滚。** 调用方若要求批次全成功，应在遇到失败 status 时抛异常。每个实例只在创建它的线程使用；多线程应各自创建实例。默认锁等待 5 秒，可以通过 `timeout` 配置。这里没有 Agent 请求级事务、反馈事务、自动重试或数据库模式迁移框架。

## target_index → 稳定 Fact ID 的兼容方案

现状：`memory_search.retrieve` 返回 `(score, memory_string)`；V2 为每个 atomic fact 单独召回并排序，Alignment 的 `target_index` 指向该 fact 的候选列表。`_apply_alignment` 目前取候选字符串，再由旧 CRUD 在 sender 的列表中定位字符串。顶层回复相关记忆列表与 V2 fact-specific 候选并不是同一索引空间。

未来接入时，每次检索保留不可变、有序的内部候选快照：

```python
# 示意结构；本阶段不实现检索适配器。
candidate_snapshot = [
    {"index": index, "fact": fact, "score": score}
    for index, (score, fact) in enumerate(ranked_facts)
]
prompt_candidates = [item["fact"].content for item in candidate_snapshot]
# Alignment 继续使用现有 Prompt，只看到编号和 content。
selected = candidate_snapshot[alignment["target_index"]]["fact"]
result = repository.update(
    sender_id, selected.id, alignment["memory"],
    expected_version=selected.version,
)
```

排序、过滤和 top-k 必须一起携带 Fact 快照，保持原有阈值、算法及 Prompt 文本。不能在 LLM 返回后重新检索、排序、按 content 查找 ID 或取“最新版本”替换 expected_version；这些做法会破坏候选与目标的对应关系或绕过并发保护。ID/version 只留在内部元数据与诊断，不要求 LLM 生成它们。

UPDATE/DELETE 使用选中快照的 ID 和 version；DUPLICATE 不写入。版本冲突或记录消失时报告本轮动作失败，不能盲目重试旧 Alignment，也不能回退按字符串写入。若同一条消息多次修改同一 ID，需保留现有目标冲突保护，并为从字符串集合转成 ID 集合建立专门回归。相同内容不同 ID 的场景必须按候选 index 精确映射，不能任取第一条。

## 隔离测试与实际结果

测试覆盖：CRUD 与重开持久化、不可变 UUID 快照、版本递增、稳定列表顺序、相同内容不同 ID、联系人读取与写入隔离、旧版本/未来版本冲突、已删除/未知目标、两个连接的过期快照、并发写入仅一个成功、SQL 注入字符串、参数校验、批次提交与未提交数据不可见、跨 sender 的完整回滚、真实 SQLite 错误、嵌套回滚、SQLite 自动整组回滚，以及资源关闭。

执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\33167\anaconda3\envs\memory_agent\python.exe' -B run_repository_tests.py
```

独立 Repository 测试也可执行 `python -B -m unittest test_sqlite_memory_repository -v`。它仅创建 `TemporaryDirectory` 中的数据库，不访问 JSON。

综合执行器先在临时数据库上运行 Repository 测试，再切换临时工作目录并显式绑定临时 style/feedback 路径，运行原有五个模块的 40 项测试。审计钩子拒绝打开临时目录以外的 SQLite 数据库，拒绝访问三个真实 JSON 文件，也拒绝写入源码或旧报告；输出每项测试日志、计数、耗时、临时数据库清理结果及原有文件 SHA-256 对照。实际执行结果保存在新建的 `repository_reports/phase1_*/results.json`，不覆盖历史运行。

开发过程中修复了测试辅助连接在 Windows 上未关闭的问题，并通过新增失败用例发现、修复了 SQLite 自动回滚后保存点清理掩盖原始异常的问题。修复范围仅限本次新建模块和测试。

## 下一阶段：JSON 迁移与 V2 接入

1. **独立迁移工具与 dry-run。** 显式输入、输出路径；只读验证 JSON 的 sender→字符串列表结构、每条内容和每 sender 的顺序与计数。为原文件生成备份/校验清单，不隐式纠正或删除异常记录。style/feedback 继续独立存储，不混入 MemoryFact 表。
2. **导入新的数据库。** 全部 facts 按原列表顺序导入，初始 version=1。导入与迁移台账在一个事务提交，台账记录源文件校验和、计数和导入标识；成功重跑识别已导入内容并跳过，失败重跑不留下部分数据。UUID 在首次成功导入后稳定，不按 content 生成 ID；即使源数据含重复字符串，也要先明确处理策略，不能静默合并。该台账需下一阶段增加，本轮没有实现迁移。
3. **核对后再切换。** 比较 sender 集合、逐 sender 内容/顺序/数量、UUID 唯一性及重开后的数据；保留原 JSON 备份。先用合成数据验证迁移幂等、失败回滚、非法数据和重复项，再在获得该阶段授权后迁移真实数据。
4. **检索适配与写入接线。** 增加 Repository 注入和携带 Fact ID/version 的内部候选结构，保留现有 Prompt、Embedding 模型、过滤阈值与行为；保持每个 fact 的 target_index 映射快照不变。增加重排、相同内容不同 ID、跨 sender、并发版本冲突、重复目标与错误传播回归。
5. **后端切换与事务范围另行明确。** Memory V1/V2 算法选择与 JSON/SQLite 存储选择分开处理。避免同一联系人在两个存储后端间交替写入；V2 SQLite 启用后，未适配 SQLite 的 V1 路径需要明确禁用或一起适配。现有 V2 允许异常前部分动作已写入，若改成整轮 all-or-nothing，属于需单独确认和测试的行为变化；不应仅因 Repository 提供事务而悄悄改变它。也不要在 LLM 推理期间长期持有 SQLite 写锁。
