# SQLite Repository 第一阶段实际验证报告

2026-10-16 20:58:36（Asia/Shanghai）运行。结果：**PASS，新增 26/26 + 原有回归 40/40，共 66/66**；0 failures、0 errors、0 skipped。综合命令 exit code 0。

| 验证 | 实际结果 | 执行器测得耗时 |
|---|---|---:|
| SQLite Repository 隔离测试 | 26/26 PASS | 0.197s |
| 原有五个模块回归 | 40/40 PASS | 0.413s |
| 综合流程（含旧回归所需 Embedding 导入） | PASS | 17.169s |

unittest 自身报告的测试运行时间分别为 0.176s 和 0.407s；与上表差别来自测试加载和日志处理。环境：现有 `memory_agent` Conda 解释器，Python 3.12.14，内置 SQLite 3.53.4。

## 实现与接口

- `memory_repository.py` 定义不可变 `MemoryFact(id, sender_id, content, version)`、三态 `MutationStatus`、携带成功快照的 `MutationResult` 和 `MemoryRepository` Protocol。
- `sqlite_memory_repository.py` 实现 `list`、`add`、`update`、`delete`、`transaction`、`close`。数据库路径必须显式传入，模块导入不创建数据库。
- ID 为 UUID4；版本初始为 1，UPDATE 保持 ID 并递增版本。UPDATE/DELETE 必须传入 `expected_version`，数据库内同时校验 sender、ID 与版本。
- 返回成功、版本冲突或不存在；跨 sender 的 ID 返回不存在且不附带 Fact。所有数据 SQL 使用参数绑定，存储错误不会被转换成“不存在”。
- 表 `memory_facts`：内部 `sequence INTEGER PRIMARY KEY AUTOINCREMENT`、`id TEXT UNIQUE NOT NULL`、`sender_id TEXT NOT NULL`、`content TEXT NOT NULL`、`version INTEGER NOT NULL DEFAULT 1`，包含非空与正整数 CHECK 约束。
- 索引：ID 唯一索引，以及 `idx_memory_facts_sender_sequence(sender_id, sequence)`。按新增顺序 list；Repository 不负责内容去重。
- 每次独立写入使用 `BEGIN IMMEDIATE`；显式批次使用同一事务，嵌套使用保存点。异常触发回滚；普通失败返回值需调用方抛异常才能回滚批次。

## 测试覆盖与开发中修复

26 项测试覆盖 CRUD、重开持久化、稳定 ID/顺序、不可变快照、内容重复但 ID 独立、版本递增、联系人读写隔离、未知和已删除记录、旧/未来版本冲突、两个连接与真正并发写入、参数化 SQL、非法参数、提交与未提交数据隔离、跨 sender 完整回滚、嵌套回滚、SQLite 错误传播及连接关闭。

开发时首次 24 项运行出现一处测试 fixture 的 Windows 文件清理错误，原因是原生 SQLite 辅助连接仅退出事务上下文而没有关闭连接；改用 `contextlib.closing` 后通过。随后新增的两项测试复现 SQLite 自动整组回滚会销毁保存点、导致清理异常掩盖原始错误的问题；实现现在检查实际事务状态，保留原始 SQLite 异常，并拒绝在已失效的外层作用域内继续写入或假装提交。新增测试和完整 26 项均已重新通过。

## 数据与兼容性证据

- 综合执行器记录 **26 个唯一临时数据库路径**，数据库文件全部清理。
- 旧回归在另一临时工作目录加载空 `memories.json`，style/feedback 加载和写入显式绑定临时文件；该目录也已清理。
- 审计拒绝临时目录以外的 SQLite 打开操作、真实三个 JSON 的任何访问，以及源码/既有结果写入；运行中 0 次拦截。
- 执行前后记录全部既有文件的 SHA-256，`original_files_unchanged=true`。另外与前次真实 Smoke 保存的原始源码哈希比较，53 个旧根目录文件仍全部一致。
- 不修改 `main.py`、`memory_v2.py`、`memory_search.py`、Prompt、冻结数据或历史报告；没有执行真实 JSON 迁移。Repository 可独立导入，不加载 Agent、SentenceTransformers 或 Ollama。
- **0 次真实 LLM 调用**。原有回归保留其原本的推理 mock；SQLite 测试使用真实数据库和真实 SQL。

## 下一阶段方案

迁移先做只读 JSON dry-run，再导入全新数据库；保留每个 sender 的原始顺序、内容和数量，初始 version=1，迁移数据和校验和台账在单个事务提交，使重跑幂等。异常数据和重复内容需明确策略，不能静默修正；导入后核对 sender 集合、顺序、计数与重开结果，原 JSON 保留备份。

V2 接入时，Embedding 的排序、筛选、top-k 全程携带同一份 `MemoryFact` 快照。Prompt 继续接收现有编号与 content，LLM 返回的 fact-specific `target_index` 映射回该列表中的 ID/version，再条件写入。不能重新按字符串查 ID，也不能用最新版本替代快照版本绕过冲突。冲突需显式报告，不盲目重试或回退 JSON。

V1/V2 算法切换和存储后端切换需分开管理，避免同一联系人在 JSON 与 SQLite 间交替写入。是否将现有 V2 的部分成功语义改成整轮事务回滚，需要独立行为决策；本阶段没有改变它。

完整接口、SQL 和接入说明见项目新增的 `SQLITE_REPOSITORY.md`；逐项测试日志为 `repository.log` 和 `baseline.log`；机器证据为同目录 `results.json`。

复现：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\33167\anaconda3\envs\memory_agent\python.exe' -B run_repository_tests.py
```

该命令每次生成新的 `repository_reports/phase1_*` 目录。
