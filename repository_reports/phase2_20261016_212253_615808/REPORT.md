# Agent V3 JSON → SQLite 第二阶段实际交付报告

执行时间：2026-10-16 21:22:53（Asia/Shanghai）。结果 **PASS：103/103**，原有 66 项全部保持通过。综合命令 exit code 0，0 failures、0 errors、0 skipped。

| 测试组 | 结果 | 执行器测量耗时 |
|---|---|---:|
| 原有 Repository | 26/26 PASS | 0.205s |
| 新增迁移与联系人登记 | 37/37 PASS | 0.879s |
| 原有 Agent 五模块回归 | 40/40 PASS | 0.330s |
| 完整执行（含旧回归的 Embedding 初始化） | PASS | 11.053s |

环境为已有 `memory_agent` Conda 环境，Python 3.12.14，SQLite 3.53.4。0 次真实 LLM 调用。测试全部使用临时数据，执行器记录的 SQLite 文件及 baseline 临时目录均已清理，原有文件运行前后哈希保持一致。

## 修改文件

- 新增 `migrate_json_to_sqlite.py`：显式路径、dry-run、源哈希、受控批次事务、逐联系人校验、重开校验、integrity_check、WAL 处理、安全发布和结构化报告。
- 新增 `test_json_to_sqlite_migration.py`：32 项迁移测试及 5 项 sender registry 测试。
- 新增 `SQLITE_MIGRATION.md`：完整接口、事务边界、正式迁移前置条件、安全发布及下一阶段接入说明。
- 最小扩展 `memory_repository.py` / `sqlite_memory_repository.py`：新增 `ensure_sender`、`list_senders` 和 `memory_senders` 登记表，保证空联系人可表达；不改变 facts ID、version 或原有 CRUD 的返回契约。
- 扩展 `run_repository_tests.py --include-migration`：保留原有运行入口，并为第二阶段创建独立报告目录。

未修改 `main.py`、`memory_v2.py`、`memory_search.py`、LLM Prompt、原有 66 项测试断言、冻结评测集或历史结果。没有生产迁移、默认后端切换或双写。

## 真实 JSON 只读检查

真实 `memories.json` 当前为 `{}`，联系人 0、事实 0。只用标准库读取，没有导入带模型副作用的模块。初始和结束 SHA-256 均为：

```text
44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a
```

真实文件仅执行 dry-run，报告 `DRY_RUN_PASSED`、`published=false`。没有创建 `real_dry_run_NOT_CREATED.sqlite3`。dry-run 的 integrity_check 为未运行（null），未伪报 SQLite 检查完成。详见 `real_json_dry_run.json`。

## 合成迁移实际报告

合成源 SHA-256：

```text
c88cc867692fed9a7a53f8eae687f299ea1cd39fc266eb477f09da1fe5cbd7e9
```

| 联系人 | 源条目 | 迁移条目 | 内容、顺序、重复项 |
|---|---:|---:|---|
| zhangsan | 0 | 0 | PASS，空联系人保留 |
| lisi | 3 | 3 | PASS，重复“喜欢摄影”两条保留 |
| 王老师 | 2 | 2 | PASS |
| 合计 | 5 | 5 | 3 个联系人、1 个空联系人、1 次重复出现 |

状态为 **SUCCESS**。事务内及关闭后重开的完整内容校验全部通过；每条新记录 version=1，ID 唯一，重开后 ID 不变；SQLite `PRAGMA integrity_check` 返回单条 `ok`。源 SHA-256 前后相同，提交与连接关闭均完成，最终没有 staging 或附属文件。正式目标仅存在于本例临时目录，取证结束后已清理，未留下示例数据库作为真实存储。

源快照保存为 `synthetic_source.json`，完整机器报告为 `synthetic_success.json`，清理证据为 `example_cleanup.json`。

## 整批原子性的实际证明

现有方法名是 `add()`，而非 `add_fact()`。它在外层事务中仅释放保存点；本工具将全部 sender 登记与全部 Repository.add 放入一个外层事务。

独立失败取证在第 **3** 条插入前注入异常，前两条执行真实 Repository 和真实 SQL：

| 观测点 | 事实条数 |
|---|---:|
| 异常前，同一事务内部查询 | 2 |
| 异常前，另一数据库连接查询 | 0 |
| 异常后，重新连接查询 facts | 0 |
| 异常后，重新连接查询 sender 登记 | 0 |

结果为 **FAILED**，`transaction.committed=false`、`rollback_verified=true`、`published=false`。源文件哈希未变，正式目标未创建，staging 已清理。该证据说明前 N−1 条没有单独提交；回滚包括联系人登记和全部先前事实。完整报告为 `synthetic_failure_atomicity.json`。

## 安全边界及故障覆盖

新增测试覆盖空对象、多联系人、空联系人、重复内容、中文/emoji/特殊字符、原始顺序、非法 JSON/类型/重复 key、缺失路径、目标已存在、同路径输入输出、已有附属文件、重复执行、关闭后重开、版本和 ID 保持、校验失败、源导入期间变化、提交后源变化/消失、WAL 检查与清理、完整性检查失败及资源关闭。

发布以同目录 staging 的 **原子硬链接创建** 为提交点；系统调用本身不覆盖目标，测试覆盖“预检查后目标在 link 时出现”及两个迁移同时竞争同一路径，只允许一个成功。竞争方文件字节保持不变。硬链接不可用时明确失败，没有 replace/rename 回退。

发布前连接关闭、事务提交、重开内容校验、integrity_check 和附属文件处理全部完成。发布前故障不留下本工具创建的正式目标。成功发布后若仅 staging 别名清理失败，报告 SUCCESS 加明确 cleanup_error，而不误报为没有目标的失败，也不删除可能已被其他进程使用的正式目标；此边界也有独立测试。

开发补测发现过 CLI 在 ASCII 默认 stdio 下无法输出中文/emoji 报告，可能在已发布后以编码异常退出；已固定 CLI 为 UTF-8，并通过真实子进程回归。未改变业务 Prompt 或记忆算法。

## 使用与下一阶段

```powershell
$env:PYTHONIOENCODING='utf-8'
$python = 'C:\Users\33167\anaconda3\envs\memory_agent\python.exe'
& $python -B migrate_json_to_sqlite.py --input synthetic_memories.json --output facts.sqlite3 --dry-run
& $python -B migrate_json_to_sqlite.py --input synthetic_memories.json --output facts.sqlite3
& $python -B run_repository_tests.py --include-migration
```

输入/输出都必填，输出父目录应已存在；已有目标拒绝覆盖；报告为 stdout UTF-8 JSON。SUCCESS/DRY_RUN_PASSED 退出 0，FAILED 退出 1。

下一阶段再做 Memory V2 的 Repository 注入和候选元数据适配：Embedding 排序/过滤全程携带不可变 Fact 快照，Prompt 内容不变，`target_index` 从同一份 fact-specific 候选列表映射出稳定 ID 与 expected_version。版本冲突显式返回，不按字符串回查或盲目重试。存储后端切换与 V1/V2 算法选择分开管理，避免 JSON/SQLite 交替写入。

**正式迁移前必须暂停旧系统 Memory 写入，并等待在途写入完成。** 哈希校验不能替代写入协调，不能消除最后一次检查与发布之间的窗口，也不能检测改写后又恢复原字节的过程。本轮没有实际切换或生产迁移。迁移整批原子性也不代表 Agent 回复、反馈和 Working Memory 已成为一个请求级事务。
