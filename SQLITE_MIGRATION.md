# Agent V3 JSON → SQLite 安全迁移工具

本阶段实现独立迁移工具与隔离测试。没有迁移真实 JSON，没有改变默认存储，没有接入 `main.py` / `memory_v2.py`，没有双写或灰度切换，也没有修改冻结评测和历史结果。

## 真实结构检查

只用 `Path.read_bytes()`、`json.loads()` 和 SHA-256 检查原始 `memories.json`，不导入 `memory_search.py`。当前真实文件是空对象 `{}`，联系人 0、事实 0、空联系人 0、重复项 0；检查时 SHA-256 为：

```text
44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a
```

旧存储格式为 `{sender_id: [memory_string, ...]}`。`memory_search.py` 从当前工作目录加载/保存整个 JSON 对象；`ensure_sender` 会创建空列表，删除最后一条事实不会删除联系人 key。原始列表可以包含重复字符串，因此迁移必须逐项保存，不能以内容作为唯一键或进行去重。旧模块导入时初始化 SentenceTransformer，迁移工具没有导入它。

输入必须是严格 UTF-8 JSON object，每个 sender 为非空白字符串、每个值为列表、每个条目为非空白字符串。重复 JSON 对象 key、NaN/Infinity、错误类型、空白条目或已有 SQLite CHECK 无法表达的前导 NUL 字符串均明确失败；不静默跳过或 trim 内容。普通中文、emoji、引号、反斜杠、换行、制表符以及文本内部的 NUL 可原样保存。

## Repository 事务与空联系人兼容

当前写入方法名是 `add()`，没有 `add_fact()`。它通过 `transaction()` 写入：无外层事务时独立提交，有外层事务时仅释放内层保存点。迁移在 **一个外层 `repository.transaction()`** 内登记全部 sender 并通过 `repository.add()` 导入全部事实；不另写事实 INSERT/UPDATE/DELETE 实现。原有事务接口已满足整批原子性，没有增加绕过事务的写入通道。

原 facts 表不能表达空联系人，本阶段最小扩展为：

```sql
CREATE TABLE memory_senders (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id TEXT NOT NULL UNIQUE CHECK (length(trim(sender_id)) > 0)
);
```

新增 `ensure_sender(sender_id)` 与 `list_senders()` 接口。空联系人有正式登记行和零条事实，后续删除最后一条事实也保留 namespace。联系人登记和 facts 在同一事务提交/回滚，按首次登记顺序返回；`add()` 自动登记 sender。facts 表结构、ID/version 和原有 CRUD 契约保持不变。

打开旧 facts-only SQLite 数据库时，在一个受控事务中按每 sender 首条事实的序号回填登记，不改变已有事实或 ID。旧 facts-only 数据库已经丢失的空联系人无法凭事实推断恢复；本工具从完整 JSON 迁移时明确登记每一个 key，因此可保留源 JSON 中的空联系人。未修改已运行过的真实数据库。

每条源记录都调用 Repository 生成新的 UUID4，version=1。重复内容也产生独立 ID。正式发布后 ID 持久稳定；两次独立迁移到不同新路径无需生成相同 ID。重跑相同输出路径明确拒绝，不做隐式覆盖或追加。

## 执行与安全发布顺序

1. 只读读取源字节，同时计算该字节快照的 SHA-256，严格解析完整格式并统计数量。
2. 确认输出父目录已存在，拒绝已有目标、悬空目标符号链接和已有 `-journal/-wal/-shm` 文件。dry-run 在此基础上再次核对源哈希，不创建数据库；dry-run 不证明文件系统支持硬链接或 SQLite 实际导入已成功。
3. 在目标父目录内用 `mkstemp` 独占创建 `.目标名.migration-随机值.sqlite3`，保证与目标位于同一文件系统。临时名不会被当作用户指定的正式数据库。
4. 在 Repository 的单个外层事务中导入全部联系人与事实，逐联系人核对数量、文本、顺序、重复项、初始版本、ID 唯一性，再核对源哈希。异常时整批回滚，同时核实联系人表和 facts 均回到空状态。
5. 整批提交并关闭 Repository。重新打开一个 Repository，完整查询并再次比较内容、顺序和生成的 ID，随后关闭。
6. 如存在 WAL 模式，执行 `wal_checkpoint(TRUNCATE)` 并拒绝 busy 状态；切换为 DELETE journal mode，执行 `PRAGMA integrity_check`，仅接受单条 `ok`。关闭检查连接，拒绝残余 staging 附属文件，对数据库文件 fsync。
7. 再次核对源 JSON SHA-256，在所有提交、校验和连接关闭之后，以 **`os.link(staging, target)` 原子创建正式目标目录项**。该操作本身拒绝已有目标，包含检查之后并发出现的目标。不会使用 `os.replace`、可能覆盖目标的 rename，或先创建一个看似正式的空数据库。硬链接不受支持、权限不足或目标发生竞争均明确失败，没有降级发布路径。
8. 删除 staging 别名和已知附属文件。发布前任何失败都不创建正式目标，并清理私有临时文件。目标竞争时仅清理本工具的 staging，不删除竞争方文件。

成功硬链接创建是**发布提交点**。如果在此之后仅 staging 别名删除失败，数据库已完整、安全发布：报告仍为 SUCCESS，但明确给出 `staging_cleanup_passed=false` 和 `cleanup_error`。不将这种情况误报成“迁移失败且没有目标”，也不冒险撤销一个可能已被其他进程使用的正式路径。清理失败或进程崩溃可能遗留带 `.migration-` 前缀的临时名，不能将其视为正式成功目标；正常失败路径和正常成功路径均由测试验证清理完成。

安全发布要求目标父目录可信、相关目标写入已协调；本工具保证发布动作不覆盖既有文件，不承诺对不受控进程发布之后替换目标文件提供保护，也不把文件 fsync 等同于所有文件系统上的断电恢复保证。

## 源数据一致性与正式迁移前置条件

源哈希在读取时、导入事务提交前，以及最终发布前核对。源内容变化、源文件消失或无法重新读取都拒绝发布；提交后才发现变化时，整个已提交数据仅在 staging 中，正式目标仍不发布。

**正式迁移前必须暂停旧系统的所有 Memory 写入，并等待在途写入完成。** 保持停写直到数据核对和后端切换完成。哈希检查只能发现观测到的变化，不能消除最后一次读取与发布之间的竞争窗口，也无法识别先改写后恢复原字节的变化；它不能替代写入协调。用户本轮未授权生产迁移，当前仅执行真实文件只读分析/预检和合成数据迁移。

## 使用方法与报告

输入、输出均为必填参数，没有默认真实数据路径，也没有 overwrite 开关。输出父目录必须已存在。

```powershell
$env:PYTHONIOENCODING='utf-8'
$python = 'C:\Users\33167\anaconda3\envs\memory_agent\python.exe'

# 先预检自己准备的合成输入；不会创建 facts.sqlite3。
& $python -B migrate_json_to_sqlite.py --input synthetic_memories.json --output facts.sqlite3 --dry-run

# 导入至尚不存在的新路径；本轮未对真实 JSON 执行此操作。
& $python -B migrate_json_to_sqlite.py --input synthetic_memories.json --output facts.sqlite3

# 重跑原有 66 项及本阶段新增隔离测试。
& $python -B run_repository_tests.py --include-migration
```

Python 接口为 `migrate_json_to_sqlite(source_path, target_path, dry_run=False) -> dict`。CLI 向 stdout 输出 UTF-8 JSON 报告，不依赖终端默认字符编码；SUCCESS / DRY_RUN_PASSED 返回 exit code 0，FAILED 返回 exit code 1，并附异常类型和原因。

报告包含源/目标路径、初始及结束源 SHA-256、联系人数量、事实条目总数、各联系人数量、空联系人数量、重复出现次数、事务提交/回滚核实、事务内及重开后的逐联系人比较结果、完整性检查、发布状态和清理状态。这里事实数量是旧列表条目数量，没有调用 LLM 重新提取 Atomic Fact。dry-run 的完整性检查为未运行（`passed=null`），不会伪报数据库校验成功。

测试与本次实际执行报告写入新建的 `repository_reports/phase2_*/`，不覆盖第一阶段或历史评测报告。所有迁移测试使用临时 JSON / SQLite，真实文件预检仅输出报告，不创建目标数据库。

## 下一阶段 V2 接入方案

先保留迁移工具的独立使用方式，实际生产迁移需要单独授权并执行上述停写协调。接入时分离 Memory V1/V2 算法选择与存储后端选择，避免同一联系人交替写 JSON 与 SQLite；确认 SQLite 数据后再显式切换，JSON 保留只读备份，不实施隐式双写。

Embedding 排序、筛选与 top-k 全程携带同一份 `MemoryFact(id, sender_id, content, version)` 快照。LLM 的候选文本、Prompt 和 `target_index` 保持现有语义；仅在内部从该 fact-specific 候选列表映射出 ID/version 执行 UPDATE/DELETE。版本冲突或目标消失必须报告，不重新按字符串定位、盲目重试或回退 JSON。加入重排、重复文本不同 ID、空 namespace、sender 隔离、过期版本和错误传播回归后，再进行真实端到端 Smoke。

迁移整批事务不意味着 V2 整轮请求事务。当前 V2 存在部分动作先成功的语义；是否改变它需要独立行为决策，不在本轮顺带修改，也不在 LLM 推理期间长期持有 SQLite 写锁。
