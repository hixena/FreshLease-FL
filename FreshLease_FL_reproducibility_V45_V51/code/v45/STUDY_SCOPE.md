# V45：晋升后权限新鲜度租约

## 为什么做 V45

V44 的 50 轮正式结果表明，`access_full` 在晋升前显著降低目标新节点的聚合份额并延迟
ASR 上升，但约第 5 轮晋升后，晚期目标份额与直接准入趋同。因此 V45 只回答一个窄问题：
**新节点获得满权后，定期回到受限观察能否持续限制长期暴露，同时保持良性新节点效用？**

这不是攻击识别器。V44 的现有风险遥测不能可靠区分攻击与正常节点，V45 不据此调阈值，
也不读取攻击 `profile` 作权限决策。

## 冻结机制

`access_freshness_lease` 只作用于预声明的 newcomer：

1. 首先积累 5 次结构合格的 `LIMITED` 观察更新；
2. 获得连续 5 次满权更新租约；
3. 租约到期后自动回到 `LIMITED`，再积累 5 次结构合格更新；
4. 续租后重复上述周期。

`LIMITED` 权重固定为 0.50。incumbent 不进入租约周期，始终保持 `ADMITTED`。硬失败、
隔离规则和 V44 的其余冻结参数不变。租约长度和复核窗口不是根据 V45 结果选择的；本版本
只验证这个预注册工作点。

## 实验矩阵

- 数据集：Fashion-MNIST；
- 20 个客户端，其中 18 个 incumbent、2 个 target newcomer；
- Non-IID α=0.5，攻击从第 2 轮开始；
- 50 轮、5 个独立种子；
- 场景：`diverse_then_repeat_backdoor` 与 `benign_concept_drift`；
- 方法：`access_freshness_lease`、V44 `access_full`、直接准入 `access_no_limited`；
- 正式矩阵：3×2×5=30 个配置。

后门场景用于安全效果，良性概念漂移用于测量相同 newcomer 权限周期的效用代价。两者必须
分别报告，不得用良性场景的低 ASR 作为安全收益。

## 运行

在项目根目录首次运行冒烟实验（6 个配置、每个 16 轮，以覆盖到期和续租）：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\flower_prototype\run_freshness_lease_v45.ps1 -Stage smoke
```

冒烟通过后运行正式矩阵：

```powershell
.\flower_prototype\run_freshness_lease_v45.ps1 `
  -Stage formal `
  -Repeats 5 `
  -Rounds 50 `
  -SkipBuild
```

中断后从同一个根目录恢复：

```powershell
.\flower_prototype\run_freshness_lease_v45.ps1 `
  -Stage formal `
  -Repeats 5 `
  -Rounds 50 `
  -ResultDirectory "D:\你的完整结果目录" `
  -Resume `
  -SkipBuild
```

`-Resume`复用完整 batch；不完整 batch 会写入新的 retry 目录，不会把半成品追加到旧 CSV。

## 冒烟验收

只有同时满足以下条件才进入正式矩阵：

- 6/6 配置完成，并生成 5 个 V45 汇总文件；
- 租约方法在逐轮表中出现 `LEASE_EXPIRED`，之后出现 `RENEWED`；
- `normal_lease_expirations=0`（incumbent 不应被周期降权）；
- 三个变体相同场景/种子均有相同轮数，无缺失配置；
- 日志没有 HTTP 400、容器提前退出或分析器异常。

## 主要输出与判定

- `freshness_lease_v45_runs.csv`：每个独立运行；
- `freshness_lease_v45_summary.csv`：均值和 Student-t 95% CI；
- `freshness_lease_v45_paired_effects.csv`：同场景同种子配对差值；
- `freshness_lease_v45_round_runs.csv`：逐轮独立运行；
- `freshness_lease_v45_round_summary.csv`：逐轮均值和区间。

主安全指标是第 10 轮至末轮的目标有效聚合份额、ASR-AUC 和最终 ASR；主效用指标是良性
概念漂移场景的最终准确率及轨迹。V45 只有在同种子相对 `access_full` 持续降低后期暴露和
ASR，同时良性准确率代价可接受时，才能支持“晋升后持续重验证优于一次晋升”的结论。
若只降低聚合份额却不降低 ASR，应报告为机制暴露控制，不宣称攻击防御成功。
