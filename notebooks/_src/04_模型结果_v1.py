# %% [markdown]
# # 04 · 模型结果（v1）
#
# > O2O 优惠券个性化投放项目 —— 第 4 步
#
# ---
#
# ## 本节目标
#
# 用第 03 节的特征训练两个模型（决策树 + XGBoost），产出**指标结果**。
#
# ## 建模设置
#
# | 模型 | 参数策略 |
# |---|---|
# | **决策树** | `GridSearchCV`：`max_depth∈{5,8,10,15}`，`min_samples_split∈{50,100,200}`，`min_samples_leaf∈{20,50,100}`，3 折 CV 以 AUC 选优，`class_weight='balanced'` |
# | **XGBoost** | `GridSearchCV`：`max_depth∈{4,6,8}`，`learning_rate∈{0.05,0.1}`，`n_estimators∈{100,200}`，`scale_pos_weight = 负/正 ≈ 15.6` |
#
# ## 类别不平衡的处理
#
# `class_weight='balanced'` / `scale_pos_weight` 本质是**提高少数类误分的惩罚权重**，等效于过采样。
#
# ## 为什么用 AUC
#
# 不平衡场景下 Accuracy 无意义（全猜负也有 93.98%）。AUC = 随机正样本排在随机负样本之前的概率，
# 只看**排序**，与「投放 = 给用户排序取头部」的业务天然对齐。
#
# ---
#
# > ### ⚠️ 本节还有第二颗雷：**随机切分**
# >
# > `train_test_split` 让同一个 user / merchant / coupon 同时出现在训练集和验证集里，
# > 验证集「见过」训练集里这些主体的画像。叠加第 03 节的聚合泄漏，指标会被抬到**不可信的高度**。

# %%
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import xgboost as xgb
import joblib

pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 180)
pd.set_option('display.unicode.east_asian_width', True)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 110
matplotlib.rcParams['savefig.bbox'] = 'tight'

ROOT = Path.cwd()
for cand in [ROOT, *ROOT.parents]:
    if (cand / '.git').exists():
        ROOT = cand
        break

OUT_DIR = ROOT / 'data_out'
FIG_DIR = ROOT / 'results' / 'figures'
TAB_DIR = ROOT / 'results' / 'tables'
MODEL_DIR = OUT_DIR / 'models_v1'
for d in (FIG_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

C_MAIN, C_ALT, C_WARN, C_GREY = '#2F6F9F', '#E08A3C', '#C0504D', '#8C8C8C'

# %% [markdown]
# ## 1. 读取特征表

# %%
data = pd.read_csv(OUT_DIR / 'feats_v1.csv')
FEATURES = [c for c in data.columns if c not in ['user_id', 'merchant_id', 'coupon_id', 'date_received', 'class']]
X = data[FEATURES]
y = data['class']

print(f'样本: {data.shape[0]:,} 行 x {len(FEATURES)} 特征')
print(f'正样本率: {y.mean() * 100:.2f}%')
print(f'特征: {FEATURES}')

# %% [markdown]
# ## 2. 随机切分（⚠️ 泄漏来源之二）

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f'训练集: {X_train.shape[0]:,}   验证集: {X_val.shape[0]:,}')
print(f'训练集正样本率: {y_train.mean() * 100:.2f}%   验证集正样本率: {y_val.mean() * 100:.2f}%')
print()

# 泄漏的直接证据：同一个主体跨越了切分边界
tr_idx, va_idx = X_train.index, X_val.index
same_user = len(set(data.loc[tr_idx, 'user_id']) & set(data.loc[va_idx, 'user_id']))
same_merchant = len(set(data.loc[tr_idx, 'merchant_id']) & set(data.loc[va_idx, 'merchant_id']))
same_coupon = len(set(data.loc[tr_idx, 'coupon_id'].dropna()) & set(data.loc[va_idx, 'coupon_id'].dropna()))
print('⚠️  随机切分的主体重叠：')
print(f'   同时出现在训练集与验证集的用户数  : {same_user:,} / {data["user_id"].nunique():,}')
print(f'   同时出现在训练集与验证集的商户数  : {same_merchant:,} / {data["merchant_id"].nunique():,}')
print(f'   同时出现在训练集与验证集的券种类  : {same_coupon:,} / {data["coupon_id"].nunique():,}')

# %% [markdown]
# ## 3. 决策树

# %%
dt_param_grid = {
    'max_depth': [5, 8, 10, 15],
    'min_samples_split': [50, 100, 200],
    'min_samples_leaf': [20, 50, 100],
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    dt_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
t0 = time.time()
dt_grid.fit(X_train, y_train)
dt_time = time.time() - t0

dt_best = dt_grid.best_estimator_
print(f'训练耗时: {dt_time:.0f}s')
print(f'最优参数: {dt_grid.best_params_}')
print(f'最优 CV-AUC: {dt_grid.best_score_:.4f}')

# %%
pred_dt = dt_best.predict(X_val)
prob_dt = dt_best.predict_proba(X_val)[:, 1]

m_dt = {
    'AUC': roc_auc_score(y_val, prob_dt),
    'Accuracy': accuracy_score(y_val, pred_dt),
    'Precision': precision_score(y_val, pred_dt),
    'Recall': recall_score(y_val, pred_dt),
    'F1': f1_score(y_val, pred_dt),
}
print('决策树（随机切分验证集）指标：')
for k, v in m_dt.items():
    print(f'  {k:<10} {v:.4f}')

# %% [markdown]
# ## 4. XGBoost

# %%
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f'scale_pos_weight = {scale_pos_weight:.2f}')

xgb_param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
}

xgb_grid = GridSearchCV(
    xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.8, random_state=42,
                      n_jobs=1, scale_pos_weight=scale_pos_weight, eval_metric='auc'),
    xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
t0 = time.time()
xgb_grid.fit(X_train, y_train)
xgb_time = time.time() - t0

xgb_best = xgb_grid.best_estimator_
print(f'训练耗时: {xgb_time:.0f}s')
print(f'最优参数: {xgb_grid.best_params_}')
print(f'最优 CV-AUC: {xgb_grid.best_score_:.4f}')

# %%
pred_xgb = xgb_best.predict(X_val)
prob_xgb = xgb_best.predict_proba(X_val)[:, 1]

m_xgb = {
    'AUC': roc_auc_score(y_val, prob_xgb),
    'Accuracy': accuracy_score(y_val, pred_xgb),
    'Precision': precision_score(y_val, pred_xgb),
    'Recall': recall_score(y_val, pred_xgb),
    'F1': f1_score(y_val, pred_xgb),
}
print('XGBoost（随机切分验证集）指标：')
for k, v in m_xgb.items():
    print(f'  {k:<10} {v:.4f}')

# %% [markdown]
# ## 5. 结果汇总
#
# > 看到下面的 AUC 时，先别急着高兴 —— **第 06 节会证明它是假的**。

# %%
metrics_df = pd.DataFrame({
    '指标': ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1'],
    '决策树': [f'{m_dt[k]:.4f}' for k in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']],
    'XGBoost': [f'{m_xgb[k]:.4f}' for k in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']],
})
metrics_df.to_csv(TAB_DIR / '04_model_v1_metrics.csv', index=False, encoding='utf-8-sig')
print(metrics_df.to_string(index=False))
print()
print(f'推荐模型: {"XGBoost" if m_xgb["AUC"] > m_dt["AUC"] else "决策树"}')
print(f'最高 AUC: {max(m_dt["AUC"], m_xgb["AUC"]):.4f}')

# %%
# 参数搜索的完整结果，供复查
pd.DataFrame(dt_grid.cv_results_).to_csv(TAB_DIR / '04_gridsearch_dt_v1.csv', index=False)
pd.DataFrame(xgb_grid.cv_results_).to_csv(TAB_DIR / '04_gridsearch_xgb_v1.csv', index=False)
print('已保存网格搜索结果')

# %% [markdown]
# ## 6. 可视化

# %%
fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

ax = axes[0]
cm = confusion_matrix(y_val, pred_xgb)
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['预测未核销', '预测已核销'])
ax.set_yticklabels(['实际未核销', '实际已核销'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=12,
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_title('XGBoost 混淆矩阵（随机切分）', fontweight='bold')

ax = axes[1]
fpr_dt, tpr_dt, _ = roc_curve(y_val, prob_dt)
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, prob_xgb)
ax.plot(fpr_dt, tpr_dt, label=f'决策树 (AUC={m_dt["AUC"]:.4f})', color=C_MAIN, lw=2)
ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={m_xgb["AUC"]:.4f})', color=C_ALT, lw=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC 曲线（随机切分）', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

ax = axes[2]
names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
xpos = np.arange(len(names))
w = 0.38
ax.bar(xpos - w / 2, [m_dt[k] for k in names], w, label='决策树', color=C_MAIN, alpha=0.9)
ax.bar(xpos + w / 2, [m_xgb[k] for k in names], w, label='XGBoost', color=C_ALT, alpha=0.9)
for i, k in enumerate(names):
    ax.annotate(f'{m_dt[k]:.3f}', xy=(i - w / 2, m_dt[k]), xytext=(0, 3),
                textcoords='offset points', ha='center', fontsize=8)
    ax.annotate(f'{m_xgb[k]:.3f}', xy=(i + w / 2, m_xgb[k]), xytext=(0, 3),
                textcoords='offset points', ha='center', fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(names)
ax.set_ylabel('得分'); ax.set_ylim(0, 1.08)
ax.set_title('指标对比（v1）', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / '04_model_result_v1.png', dpi=140)
plt.show()

# %% [markdown]
# ## 7. 保存模型

# %%
import json

joblib.dump(dt_best, MODEL_DIR / 'dt_v1.pkl')
joblib.dump(xgb_best, MODEL_DIR / 'xgb_v1.pkl')
joblib.dump(FEATURES, MODEL_DIR / 'feature_list_v1.pkl')

# 落盘最优超参：第 06 节的对照实验要用固定超参跑，避免重复网格搜索
best_params = {'dt': dt_grid.best_params_, 'xgb': xgb_grid.best_params_}
with open(MODEL_DIR / 'best_params_v1.json', 'w', encoding='utf-8') as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)

print(f'模型已保存至: {MODEL_DIR}')
print(f'最优超参: {best_params}')

# %% [markdown]
# ## 本节小结
#
# | 项 | 值 |
# |---|---|
# | 最高 AUC | 见上表（远超该任务合理水平） |
# | 切分方式 | 随机切分 —— **同一主体跨越切分边界** |
# | 特征来源 | 全量聚合 —— **含自身行与未来信息** |
#
# 一个正样本率仅 6% 的任务，AUC 不可能到 0.98。**这个数字反常得刺眼。**
#
# 下一步 → **05 特征排名（v1）**：看看模型把票投给了谁。
# 如果排名第一的是那个「历史核销间隔均值」，那基本就实锤了。
