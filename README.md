# tabulargen

表格条件高斯扩散模型。实现位于 `src/tabulargen/`，编码、训练、采样、下游评估可在独立进程运行。

## 在已有 conda 环境运行

```bash
conda activate dptabgen
python -m pip install -e . --no-deps --no-build-isolation

python -m tabulargen --config configs/adult/smoke.toml --encode
python -m tabulargen --config configs/adult/smoke.toml --train
python -m tabulargen --config configs/adult/smoke.toml --sample
python -m tabulargen --config configs/adult/smoke.toml --eval
```

安装命令仅注册本地可编辑包，不安装或升级其他依赖。当前工作机器已完成该操作。也可使用 `tabulargen --config ...`。仅保留包入口，原根目录 scripts/、models/、utils/ 已删除。

`smoke.toml` 使用 Adult 全量数据、CPU、小网络、1 轮训练、20 个扩散步、512 条合成样本，只用于流程检查。默认输出到新的 `exp/adult_smoke/run_001`。文件中的相对路径以 TOML 所在目录为基准；从其他工作目录调用时给 `--config` 绝对路径即可。

## 联合评估与多次采样

默认 `[evaluation] model = "all"`，一次 `--eval` 会运行 CatBoost、决策树（tree）、随机森林（rf）、逻辑回归（lr）和 MLP。`results.json` 的 `metrics` 是五个模型在 train/val/test 上各项指标的等权平均，`per_model` 保留五个模型各自的结果。终端也会打印平均指标。平均时直接对五个模型求和除以 5，不把 CatBoost 和 simple 两组平均后再除以 2。

已有样本可直接执行，无需重新训练扩散模型或采样：

```bash
python -m tabulargen --config configs/adult/smoke.toml --eval
```

Python 调用同样使用 `run(config, ["eval"])`，确保 `config["evaluation"]["model"] = "all"`。

```bash
python -m tabulargen --config configs/adult/smoke.toml --sample --sample-seed 5
python -m tabulargen --config configs/adult/smoke.toml --eval --sample-seed 5
python -m tabulargen --config configs/adult/smoke.toml --eval --sample-seed 5 --eval-model simple
python -m tabulargen --config configs/adult/smoke.toml --eval --sample-seed 5 --eval-seed 1
```

可用 `--eval-model catboost` 或 `--eval-model simple` 单独评估。配置默认采样 seed 为 4。不同采样种子、评估器和评估种子各自保存结果；同一目录的重复产出会报错，避免覆盖。新训练请更改 `[experiment].path`，例如 `../../exp/adult_smoke/run_002`，再执行编码与训练。

```text
exp/adult_smoke/run_001/
  config.json
  encoded/                       # 编码数据、编码器、元数据及编码配置
  checkpoints/                   # checkpoint.pt、model.pt、model_ema.pt
  logs/loss.csv
  samples/seed_4/                 # reverse.csv、unreverse.csv、生成配置和来源校验
  evaluation/seed_4/all/seed_0/
    config.json
    results.json
```

- [配置、种子与旧配置迁移](docs/configuration.md)
- [模块边界、检查点与结果格式](docs/stage-contracts.md)
- [基线与迁移验证](docs/baseline.md)

```bash
python -m pytest -q
```

GPU、DP 训练与回归任务没有纳入本轮验证；当前编码入口明确限定分类任务。
