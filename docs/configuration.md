# 配置与实验运行

所有仓库 TOML 已迁移为 `schema_version = 2`。解析器只读配置，补齐默认值后返回新的字典；训练根目录的 config.json 保存实际解析后的配置。原始 TOML 不会被训练或调参改写。

## 配置项

| 配置 | 内容 |
|---|---|
| seed | 各阶段未单独设置时的默认种子，默认 0 |
| device | 扩散训练/采样设备，如 cpu、cuda:0 |
| data.path | 已划分的原始 CSV 与 info.json 所在目录 |
| experiment.path | 一次训练运行的根目录，例如 ../../exp/adult/run_001 |
| encoding | num_encoder、cat_encoder、seed、histogram_epsilon、histogram_delta |
| model | 去噪网络参数；d_in 由编码维度推导，无需手填 |
| diffusion | num_timesteps、gaussian_loss_type、scheduler |
| train | epochs、lr、weight_decay、batch_size、num_workers、seed |
| sample | num_samples、batch_size、seed、class_distribution（empirical 或 uniform） |
| evaluation | model（all/catboost/simple，默认 all）、mode（synthetic/real）、seed、catboost_params_path、可选 params |
| privacy | is_dp、epsilon、delta、max_grad_norm；作用于扩散训练 |

`encoding.histogram_*` 管理原有标签直方图预算，独立于训练预算。旧配置中的 sigma 和 noise_multiplicity_K 原本不参与当前训练，迁移后不再保留，以免误以为它们生效。simple 模型依照 scikit-learn 在 CPU 上运行；CatBoost 的 task_type/thread_count 等设备参数仍由它自己的参数配置控制。

所有相对数据、实验、CatBoost 参数路径以 **TOML 文件所在目录** 为基准，绝对路径原样使用。例：configs/adult/config.toml 中的 data.path 为 `../../data/adult`，catboost_params_path 为 `catboost.json`。主流程不会依赖当前工作目录搜索数据或参数。

每个阶段的 seed 优先于顶层 seed；CLI 的 --sample-seed 和 --eval-seed 再覆盖对应阶段配置。CLI 修改只作用于本次运行。编码包括原有直方图随机噪声在内均设置种子；相同输入和环境下编码产物可复现。GPU 跨设备、跨依赖版本的位级复现不作保证。

联合评估 `model="all"` 时，evaluation.params 按五个模型分组。CatBoost 参数文件继续由 catboost_params_path 指定，params.catboost 覆盖其中同名参数，其他四个模型使用 params.tree/rf/lr/mlp。例如：

```toml
[evaluation]
model = "all"
mode = "synthetic"
seed = 0
catboost_params_path = "catboost.json"

[evaluation.params.catboost]
iterations = 1000
```

随机森林参数：

```toml
[evaluation.params.rf]
n_jobs = 2
max_depth = 12
```

单独 `model="catboost"` 时 params 直接存放 CatBoost 参数；`model="simple"` 时使用 tree/rf/lr/mlp 分组。联合评估的平均值对五个模型等权计算。

评估种子由 evaluation.seed 统一传给分类器；不要在 CatBoost 参数中重复设置 random_seed。只运行采样、评估时，无需提供 model、diffusion、train、privacy 的训练参数，完整模型参数来自检查点。

## 从上一版配置迁移

仓库配置已经改好，私有 TOML 请参照 configs/adult/smoke.toml：

| 上一版 | 当前版 |
|---|---|
| data_path | data.path |
| exp_path | experiment.path（请使用新的 run 目录） |
| model_params | model |
| diffusion_params | diffusion |
| train.main | train |
| eval.type.eval_model | evaluation.model |
| eval.type.eval_type | evaluation.mode |
| dp | privacy |

新增 schema_version=2；将路径改为相对于 TOML 文件；加入 encoding 和 evaluation.catboost_params_path，移除手填 model.d_in。旧 TOML 会明确报版本错误，不会猜测路径或自动重写旧实验。

## 保存与覆盖规则

一个 experiment.path 对应一次固定编码和训练。编码与训练拒绝覆盖已有产物；新训练使用新 run 路径。采样按 seed 独立保存，评估按样本 seed/模型/评估 seed 独立保存；完全相同的输出位置也拒绝覆盖。该实现尚未提供 --overwrite 或断点续训。

真实数据评估保存在 `evaluation/real/<model>/seed_<evaluation_seed>/`，与合成数据结果分开。CatBoost 真实数据评估无需编码或训练；simple 和 all 的真实数据评估沿用生成模型编码，需要先 --encode。

调参入口也采用相同接口，保留每个 trial 的配置、检查点和各 seed 的结果：

```bash
python -m tabulargen.tools.tune_ddpm --config configs/adult/smoke.toml --output exp/adult/tuning_001 --num-trials 2 --sample-seeds 0 1
python -m tabulargen.tools.tune_catboost --config configs/adult/smoke.toml --output exp/adult/catboost_tuning_001 --num-trials 2
```

--output 必须是新目录，路径相对于命令的工作目录。不再自动删除 trial，也不覆盖仓库里的 CatBoost 参数。数据拆分、CSV 转换和 SDMetrics 辅助工具的原有业务参数尚未统一进主流程配置。
