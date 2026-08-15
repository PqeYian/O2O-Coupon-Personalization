# v2 · 无泄漏修复版

对 v1 的**数据泄漏问题**做了系统性修复，提供可信的模型评估与测试集预测。

## 相对 v1 的关键修复

| # | 修复点 | 说明 |
|---|---|---|
| 1 | **时间切分** | 1-5 月训练 / 6 月验证 / 7 月测试，取代 v1 的随机切分 |
| 2 | **无泄漏聚合特征** | 用户/商户/券聚合特征**只在训练段计算**，验证/测试特征值完全来自训练历史，不含自身行、不含未来信息 |
| 3 | **修复 `coupon_used_rate`** | v1 分子分母统计同一批记录致比率恒等于 1，v2 分母改为"总核销次数" |
| 4 | **恢复 `discount_rate`** | v1 错误地将该真实券面特征排除在模型外 |
| 5 | **新增时间特征** | 领券日星期 / 是否周末 / 日序号 |

## 目录约定

```
code_v2/
├── data/           # 原始数据放入此目录 (train.csv / test.csv)
├── preprocess.py   # 清洗 + 时间切分 + 无泄漏特征工程
├── data_out/       # 预处理输出 (train_feats.csv / val_feats.csv / test_feats.csv)
├── model.py        # 训练 + 诚实评估 + 测试集预测
└── models/         # 训练好的模型 (不入库)
```

> 原始数据也可放在其他目录，通过环境变量 `O2O_DATA_DIR=<目录>` 指定。

## 运行

```bash
pip install -r ../requirements.txt
# 将原始 train.csv / test.csv 放入 ./data/
python preprocess.py
python model.py
```

输出：`submission.csv`（7 月测试集核销概率，用于投放排序）、`model_comparison_v2.png`（ROC / 特征重要性 / 混淆矩阵）。

## v1 → v2 结果对比（验证段 = 6 月）

| 指标 | v1（泄漏，随机切分） | v2（无泄漏，时间切分） |
|---|---|---|
| 决策树 AUC | 0.9839 | 0.5864 |
| XGBoost AUC | 0.9870 | **0.6271** |

详细泄漏诊断见根目录 [`README.md`](../README.md) 第 10 节。
