# 工程化基线记录

前半部分为第 1–3 步完成时的历史记录，目录及旧配置格式以当时版本为准。第 4–5 步的当前验证见文末，运行方式以 README 为准。

日期：2026-09-12。重构前 Git HEAD：`b36bb250af1e29baefd02995003763f55dfad6f4`。开始时工作区无未提交修改。

使用既有 conda 环境 `/home/ruhe/miniconda3/envs/dptabgen`，Python 3.12.8、PyTorch 2.7.0+cu128；没有安装、升级或替换依赖。本次实验明确使用 CPU，线程环境变量 `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。原版 DataLoader 的本地进程通信受到沙箱限制，因此训练及分进程测试经授权在沙箱外运行。

## 保存的证据

本机目录 `exp/engineering_baseline_20260912/` 保存了：

- `source-before.tar.gz`：修改前全部 Git 跟踪文件的快照。
- `git-head.txt`、`git-status-before.txt`：版本及工作区状态。
- `environment.yml`、`requirements-freeze.txt`：conda 导出与 pip 完整依赖记录。
- `adult-before.sha256.json`：原有 exp/adult 全部文件的校验值。
- `input-data.sha256.json`：Adult 输入数据的校验值。
- `original.toml`、`original.log`、`original/`：原版配置、日志与完整实验产物。
- `refactored.toml`、`refactored-*.log`、`refactored/`：改造后配置、各阶段日志与产物。
- `commands.txt`、`verify_baseline.py`、`verification.json`：命令、对照程序与结果。

`exp/` 沿用现有 Git 忽略规则，以上较大的快照和实验产物只保存在本机；迁移机器时需要单独备份该目录。本文件与源码、回归测试、smoke 配置可纳入版本管理。

## 实验设置与复现

使用原有 Adult 数据划分：训练 29305、验证 9768、测试 9769 行。编码 minmax + alb，MLP 隐层 [32, 32]，训练 1 轮，batch_size 256，20 个扩散步，非 DP，训练 seed 0、采样 seed 4，生成 512 条样本。CatBoost 使用原有 configs/adult/catboost.json。仅用于工程流程对照，不用于报告生成模型的最终性能。

修改前实际执行：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/ruhe/miniconda3/envs/dptabgen/bin/python -B scripts/pipeline.py \
  --config exp/engineering_baseline_20260912/original.toml --train --sample --eval
```

再次复现原版需使用 source-before.tar.gz 中的源码，并在原版配置中改用新的 exp_path。新版已要求显式 --encode，不能直接用新版代码运行上述命令来复现原版入口行为。

新版日常运行示例见 README：configs/adult/smoke.toml 具有相同计算设置，但输出到独立的 exp/adult_smoke。请从项目根目录运行；新实验使用新的 exp_path。

## 对照结果

| 指标 | 原版 | 四进程新版 |
|---|---:|---:|
| 验证集 ROC-AUC | 0.566644969405307 | 0.566644969405307 |
| 测试集 ROC-AUC | 0.567129855865868 | 0.567129855865868 |
| 测试集 F1（weighted） | 0.6572732564090966 | 0.6572732564090966 |
| 测试集 Accuracy | 0.7606715119254785 | 0.7606715119254785 |

额外检查全部通过：

- 普通模型和 EMA 模型的所有权重逐张量相等。
- reverse.csv 逐字节相同，完整 CatBoost 指标 JSON 数值相同。
- 独立采样、评估前后的六个编码产物、三份模型文件、训练配置校验值不变。
- 旧 exp/adult 所有文件校验值不变。
- `python -m pytest -q tests/test_stage_contracts.py`：2 passed。集成用例在临时目录创建小型分类数据，分别启动四个子进程，额外验证 simple 结果保存、采样不依赖训练配置，以及缺文件/重复编码/编码不匹配时的失败行为。

编码阶段原有 DP 直方图未统一设置随机种子，其元数据噪声不保证逐字节复现；本次非 DP 训练和采样不使用该分布。GPU、DP 训练、回归任务和全量调参尚未作为本轮验收范围运行。


## 第 4–5 步：统一配置与 src 包迁移

迁移前源码及配置快照位于 `exp/engineering_structure_20260912/source-before.tar.gz`，包含上一轮尚未提交的工作。现有 requirements.txt 保持不变；仅执行了本地包可编辑安装，未解析或升级依赖。

使用相同 Adult 短训练设置，从 `/tmp` 分别启动 `python -m tabulargen` 的四个阶段。新实验在 `exp/engineering_structure_20260912/run_001`。各阶段日志、实际 config.toml、对照脚本 verify.py、verification.json 位于该目录的上一级。

检查结果：

- 普通与 EMA 模型的每个权重张量均与原版基线相等。
- 原始空间合成 CSV 逐字节相同，CatBoost 的 train/val/test 指标全部相同。
- 测试集 ROC-AUC 仍为 0.567129855865868。
- 原有 exp/adult 校验值保持不变。
- 用上一轮 v1 检查点和旧模块名 pickle 独立采样，输出也与原版基线逐字节一致；保存在 legacy_checkpoint_check/。
- 全部包模块可无副作用导入；仓库所有 TOML 配置都能从配置位置解析到实际数据和参数文件。
- tests/test_stage_contracts.py 验证独立进程运行、编码种子复现、原始/编码维度区分、多采样种子与评估种子共存、统一结果结构、来源校验、真实 CatBoost 无需生成产物，以及两个调参入口对新目录和结果格式的读取。

本次没有验证 GPU、DP 训练、回归任务或长时间性能实验。编码直方图现在受 encoding.seed 控制，解决了上一轮元数据噪声不可复现的问题；非 DP 基线的训练与采样数值保持不变。simple 的平均结果现在保留完整精度并保存各个模型的指标，因此文件结构和原先提前舍入的数字表示有所改变。


## 兼容层清理

上述 v1/旧 pickle 验证属于迁移时的历史记录。当前按要求删除了旧入口和兼容分支，只支持当前包路径、v2 检查点和 encoded_dim 元数据。历史实验及源码快照仍保留在 exp/，没有为适配当前代码而改写。
