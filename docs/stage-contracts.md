# 模块与阶段约定

## 包结构

```text
src/tabulargen/
  cli.py                  # 薄阶段编排、CLI 与产物记录
  config.py               # 配置读取、默认值、路径解析与基础校验
  layout.py               # 单次训练运行的目录约定
  artifacts.py            # 输入产物完整性校验
  checkpoints.py          # 当前检查点加载与编码校验
  io.py                   # JSON/TOML 读写
  data/                   # 编码、逆变换、预处理、Dataset
  models/                 # MLP、高斯扩散过程、数学函数
  training/               # 训练循环、隐私机制、梯度诊断
  sampling/               # 检查点加载与采样
  evaluation/             # 联合评估入口、CatBoost/simple 实现、指标函数
  tools/                  # 调参、拆分、转换、SDMetrics 辅助工具
```

根目录 scripts/、models/、utils/ 已删除，仅保留 src/tabulargen/ 中的实现。主流程使用 `python -m tabulargen`；辅助工具使用 `python -m tabulargen.tools.<工具名>`。所有导入使用 tabulargen 包路径，不提供旧导入名兼容；导入辅助工具不会启动实验或修改数据。

## 四个阶段

| 阶段 | 输入 | 输出 |
|---|---|---|
| encode | data.path 下训练/验证/测试 CSV、info.json；encoding 配置 | encoded/ 下 CSV、编码器、info.json、config.json |
| train | encoded/ 产物；模型、扩散、训练、隐私配置 | checkpoints/ 下检查点与权重；logs/loss.csv；根 config.json |
| sample | 完整检查点及绑定编码产物；采样设置 | samples/seed_N 下原始空间/编码空间 CSV 和 config.json |
| eval | 真实数据划分、合成表格或真实训练集、评估配置 | evaluation/来源/模型/seed_N 下 results.json 和 config.json |

编码器只在训练集拟合。raw_feature_count 是标签以外的原始列数，encoded_dim 是数值归一化与类别展开后的输入维度，不再提供 n_features 别名。训练读取编码维度，在局部复制参数后补齐 d_in；模型构造不再改变传入字典。训练会核对编码配置与实际编码产物一致。

后续阶段不得重建上游产物。采样和评估校验来源；更换编码文件、检查点或样本文件会在写出下游结果前报错。simple 仍使用生成模型的编码器对数据变换；本轮没有改变这项评估协议。

## 检查点

新 checkpoint.pt 使用 format_version=2，包含普通和 EMA 权重、完整 model_params、diffusion_params、实际 train_params，以及 encoding.path / encoding.sha256。

encoding.path 相对于检查点目录，通常为 ../encoded。采样需要保留六个被校验的编码产物（train/val/test.csv、info.json、两个 pickle），实验目录可整体移动。样本来源记录同时保存检查点和样本的 SHA-256，评估时检查两者匹配；移动整个运行目录后，优先使用当前运行的检查点进行验证。

采样默认使用 EMA。单独 --sample 不依赖训练配置内的网络/扩散参数。--checkpoint 只接受 format_version=2 的完整检查点；编码器直接使用当前包类名进行 pickle 加载，不映射历史模块名。

检查点加载统一在 checkpoints.py 中实现，不接受 v1 或仅 state_dict 的早期权重文件。普通 model.pt / model_ema.pt 仍输出，供研究代码使用。尚未实现优化器、随机状态和隐私会计状态恢复的断点续训。

## 统一评估结果

统一入口 evaluation.runner.evaluate_models 默认运行五个模型，返回 metrics 和 per_model，由编排层统一写文件：

```json
{
  "schema_version": 1,
  "model": "all",
  "mode": "synthetic",
  "sample_seed": 4,
  "evaluation_seed": 0,
  "inputs": {},
  "metrics": {
    "train": {"f1": 0.0, "accuracy": 0.0, "roc_auc": 0.0},
    "val": {"f1": 0.0, "accuracy": 0.0, "roc_auc": 0.0},
    "test": {"f1": 0.0, "accuracy": 0.0, "roc_auc": 0.0}
  },
  "per_model": {}
}
```

示例数值只是结构占位。inputs 记录真实数据位置、样本路径及相关校验值。默认 all 的 per_model 包含 catboost/tree/rf/lr/mlp，metrics 为五个模型的算术平均（每个模型占 1/5），逐个 split 和 metric 计算。单独 CatBoost 的 per_model 只有 catboost；simple 只包含其四个模型。所有模式共用 average_metrics，平均时不提前舍入。显示时仍可舍入。真实数据模式的 sample_seed 为 null。

两种评估器仍保留既有统计定义：二分类 weighted F1，多分类 macro F1。SDMetrics 的分布/隐私距离指标属于辅助工具，未混入下游分类结果结构。
