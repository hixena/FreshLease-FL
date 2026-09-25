# V49：先晋升、后启动后门的独立补充实验

这组实验专门回答旧稿未覆盖的问题：目标 newcomer 在第 5 轮左右实际获得
`ADMITTED` 与完整聚合权重以后，才从第 **6** 轮开始注入后门。
V47/V48 的第 2 轮启动矩阵保留原样；两组结果不得混合成同一个 10 种子统计量。

## 冻结设置

| 项目 | 设置 |
| --- | --- |
| 主数据与模型 | CIFAR-10；原 V48 多项逻辑回归及相同训练配置 |
| 客户端与目标 newcomer | 20 个客户端，2 个目标节点；18 个预声明成熟正常节点 |
| 非 IID | Dirichlet α=0.5；mixed_maturity |
| 对照 | `access_freshness_lease` L=5、`access_full`、独立复现的 `rffl_reputation` |
| 权限机制 | LIMITED λ=0.50，观察 O=5，完整权限租约 L=5 |
| 场景 | `diverse_then_repeat_backdoor` 和 `benign_concept_drift` |
| 攻击起始 | 第 6 轮；第 1—5 轮不得施加后门或概念漂移 |
| 正式样本 | 50 轮 × 种子 0—9 × 3 方法 × 2 场景 = 60 个独立配置 |
| 计算控制 | 每客户端每轮最多 512 样本；每类验证 100、独立测试 200 样本 |

第 6 轮是**预先固定**的启动轮次，不随单次运行的最终效果调整。
后门比较采用相同种子、客户端分区、训练长度和评估数据；RFFL-style 从初始准入
即授予完整权限，对其不要求“先从 LIMITED 晋升”，但采用完全相同的攻击起始轮次。

## Windows 运行

在解压后**项目根目录**打开 PowerShell，确保 Docker Desktop 已启动：

```powershell
Set-ExecutionPolicy -Scope Process Bypass

# 若已从 V48 项目复制同一份 cifar10.npz，可跳过数据准备。
python -m pip install numpy
python .\flower_prototype\prepare_cifar10.py

# 6 个配置 × 16 轮；优先验证第 5 轮晋升及第 6 轮攻击时序。
.\flower_prototype\run_post_promotion_v49.ps1 -Stage smoke

# 60 个配置 × 50 轮；只在上述冒烟完整通过且镜像未改动时使用 SkipBuild。
.\flower_prototype\run_post_promotion_v49.ps1 -Stage formal -SkipBuild
```

脚本保留逐轮进度条。两个阶段会生成彼此独立的时间戳结果目录。
若中断，请在**原阶段、原轮数、同一 ResultDirectory** 下加 `-Resume`：

```powershell
$resultRoot = "D:\实验目录\flower_prototype\post_promotion_v49_results-formal-实际时间戳"
.\flower_prototype\run_post_promotion_v49.ps1 -Stage formal `
  -ResultDirectory $resultRoot -Resume -SkipBuild
```

`-Resume` 只重用已完整的批次；一个未完成批次会进入新的 retry 子目录。
冒烟目录不能当正式目录复用；分析程序逐配置核对 1—50 轮、攻击启动轮次与晋升顺序。
每批结束即输出 `post_promotion_v49_promotion_guard.csv`；若目标节点未在攻击
之前晋升并完成一次干净的完整权重聚合，分析立即失败，不得选择性删除该种子。

## 输出与预设分析

正式目录包括四个原始 `real_fl_*.csv`、`post_promotion_v49_runs.csv`、
`post_promotion_v49_summary.csv`、`post_promotion_v49_paired_effects.csv`、
`post_promotion_v49_paired_summary.csv`、逐轮结果及晋升校验表。

主对照是租约与无租约渐进权限。主指标分别是第 10—50 轮聚合有效权重中的
目标份额、以及第 6—50 轮逐轮后门 ASR 的梯形积分除以 44 轮区间长度；
最终 ASR、正常任务精度、运行时间和 RFFL-style 比较作为补充。
第 6—9 轮为早期攻击窗口，第 10—50 轮为后期窗口；不会把无攻击的第 2—4 轮
误记为早期攻击窗口。报告均值、配对差、95% CI、逐种子方向和双侧配对检验；
无显著性时仅报告区间，不以“无差异”或“等效”替代。

## 运行环境限制

源码包不包含 CIFAR-10 数据，需在本地准备 `flower_prototype\data\cifar10.npz`。
本次交付只完成代码、分析保护条件及本地单元测试；正式 Flower/Docker 结果须由
上述命令在装有 Docker Desktop 的机器上产生。论文数值只能在正式 CSV 完整且
晋升校验通过后更新。
