# legacy —— 旧版代码归档

本目录保存重构前的代码与产物，**仅供追溯历史，不是项目主线**。

> 主线请见仓库根目录的 [`README.md`](../README.md) 与 [`notebooks/`](../notebooks/) ——
> 十个步骤的完整流程已重构为可复现的 notebook 序列，本目录的内容已被其完全覆盖。

## 内容

| 目录 | 内容 | 说明 |
|---|---|---|
| `code_v1/` | 初版三个 notebook + `feature_name1.py` | **含数据泄漏**的版本，AUC 虚高。⚠️ notebook 内使用硬编码绝对路径（`F:\my item\...`），**换机器无法直接运行** |
| `code_v2/` | `preprocess.py` + `model.py` | 第一次无泄漏修复：时间切分 + 仅训练段聚合 |
| `docs/` | v1 的 HTML 导出 + 图表 | `docs/images/` 中的部分图表曾被根 README 引用 |

## 为什么保留

- **`code_v1` 的泄漏代码有教学价值**：它是一份「看起来完全正常、实则埋雷」的真实样本，
  被用于对照说明数据泄漏有多隐蔽。
- git 历史里也能找回，但保留在工作区更方便直接对照。

## 新旧对应

| legacy | 重构后 |
|---|---|
| `code_v1/探索性分析.ipynb` | `notebooks/02_数据探索EDA.ipynb` |
| `code_v1/数据预处理.ipynb` | `notebooks/01_数据预处理.ipynb` |
| `code_v1/feature_name1.py` | `notebooks/03_特征工程_v1.ipynb` |
| `code_v1/模型构建+评价.ipynb` | `notebooks/04_模型结果_v1.ipynb` + `05_特征排名_v1.ipynb` |
| `code_v2/preprocess.py` | `notebooks/07_解决数据泄漏.ipynb` + `08_最终特征工程.ipynb` |
| `code_v2/model.py` | `notebooks/09_最终模型结果.ipynb` + `10_最终特征排名.ipynb` |
