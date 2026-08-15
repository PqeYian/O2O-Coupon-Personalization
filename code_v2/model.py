# -*- coding: utf-8 -*-
"""
O2O优惠券个性化投放 - v2 模型构建与诚实评估
=============================================
基于 v2 预处理输出的时间切分特征表:
  训练段(1-5月) -> 训练模型
  验证段(6月)   -> 诚实评估 (特征全部来自训练段历史, 无泄漏)
  测试段(7月)   -> 生成提交预测

用法:
  python model.py   (需先运行 preprocess.py)
输出:
  models/xgb_model.pkl  models/dt_model.pkl
  model_comparison_v2.png  submission_v2.csv
"""
import pandas as pd
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
import xgboost as xgb
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data_out')   # preprocess.py 的输出目录
MODEL = os.path.join(BASE, 'models')
os.makedirs(MODEL, exist_ok=True)

# ============================================
# 1. 读取特征表
# ============================================
print("=" * 70)
print("【1. 读取时间切分特征表】")
print("=" * 70)

train = pd.read_csv(os.path.join(DATA, 'train_feats.csv'))
val = pd.read_csv(os.path.join(DATA, 'val_feats.csv'))
test = pd.read_csv(os.path.join(DATA, 'test_feats.csv'))

FEATURES = [c for c in train.columns if c != 'class']
print(f"训练: {train.shape}, 验证: {val.shape}, 测试: {test.shape}, 特征: {len(FEATURES)}")

Xtr, ytr = train[FEATURES], train['class']
Xva, yva = val[FEATURES], val['class']
Xte = test[FEATURES]

print(f"验证段正样本率: {yva.mean():.4f}")

# ============================================
# 2. 决策树
# ============================================
print("\n" + "=" * 70)
print("【2. 决策树 (时间切分诚实评估)】")
print("=" * 70)

dt = DecisionTreeClassifier(max_depth=15, min_samples_leaf=100,
                            min_samples_split=50, random_state=42,
                            class_weight='balanced')
t0 = time.time()
dt.fit(Xtr, ytr)
pred_dt = dt.predict(Xva)
prob_dt = dt.predict_proba(Xva)[:, 1]

print(f"训练耗时 {time.time()-t0:.0f}s")
print(f"AUC:       {roc_auc_score(yva, prob_dt):.4f}")
print(f"Accuracy:  {accuracy_score(yva, pred_dt):.4f}")
print(f"Precision: {precision_score(yva, pred_dt):.4f}")
print(f"Recall:    {recall_score(yva, pred_dt):.4f}")
print(f"F1:        {f1_score(yva, pred_dt):.4f}")

# ============================================
# 3. XGBoost
# ============================================
print("\n" + "=" * 70)
print("【3. XGBoost (时间切分诚实评估)】")
print("=" * 70)

scale_pos_weight = len(ytr[ytr == 0]) / max(len(ytr[ytr == 1]), 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
)
t0 = time.time()
xgb_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
pred_xgb = xgb_model.predict(Xva)
prob_xgb = xgb_model.predict_proba(Xva)[:, 1]

print(f"训练耗时 {time.time()-t0:.0f}s")
print(f"AUC:       {roc_auc_score(yva, prob_xgb):.4f}")
print(f"Accuracy:  {accuracy_score(yva, pred_xgb):.4f}")
print(f"Precision: {precision_score(yva, pred_xgb):.4f}")
print(f"Recall:    {recall_score(yva, pred_xgb):.4f}")
print(f"F1:        {f1_score(yva, pred_xgb):.4f}")

# ============================================
# 4. 特征重要性
# ============================================
print("\n【4. XGBoost 特征重要性】")
importance = pd.DataFrame({'feature': FEATURES,
                           'importance': xgb_model.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
print(importance.head(15).to_string(index=False))

# ============================================
# 5. 可视化对比
# ============================================
print("\n【5. 生成可视化】")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# ROC
ax = axes[0]
fpr_dt, tpr_dt, _ = roc_curve(yva, prob_dt)
fpr_xgb, tpr_xgb, _ = roc_curve(yva, prob_xgb)
ax.plot(fpr_dt, tpr_dt, label=f'决策树 (AUC={roc_auc_score(yva, prob_dt):.3f})', color='#FF6B6B')
ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(yva, prob_xgb):.3f})', color='#4ECDC4')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC曲线 (验证段=6月, 无泄漏)'); ax.legend(); ax.grid(alpha=0.3)

# 特征重要性
ax = axes[1]
top = importance.head(10).iloc[::-1]
ax.barh(top['feature'], top['importance'], color='#4ECDC4')
ax.set_title('XGBoost 特征重要性 Top10'); ax.grid(axis='x', alpha=0.3)

# 混淆矩阵
ax = axes[2]
cm = confusion_matrix(yva, pred_xgb)
ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['未消费', '已消费']); ax.set_yticklabels(['未消费', '已消费'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
ax.set_title('XGBoost 混淆矩阵 (验证段)')

plt.tight_layout()
fig.savefig(os.path.join(BASE, 'model_comparison_v2.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("可视化已保存: model_comparison_v2.png")

# ============================================
# 6. 保存模型 + 测试集预测
# ============================================
print("\n【6. 保存模型并预测测试集】")

# 以验证集 AUC 选择最优模型
best_model = xgb_model if roc_auc_score(yva, prob_xgb) > roc_auc_score(yva, prob_dt) else dt
joblib.dump(xgb_model, os.path.join(MODEL, 'xgb_model.pkl'))
joblib.dump(dt, os.path.join(MODEL, 'dt_model.pkl'))
joblib.dump(FEATURES, os.path.join(MODEL, 'feature_list.pkl'))
print(f"模型已保存至: {MODEL}")

test_prob = best_model.predict_proba(Xte)[:, 1]
sub = pd.DataFrame({'user_id': test['user_id'], 'coupon_id': test['coupon_id'],
                    'date_received': test['date_received'],
                    'probability': test_prob})
sub.to_csv(os.path.join(BASE, 'submission_v2.csv'), index=False)
print(f"测试集预测已保存: submission_v2.csv  ({len(sub)} 行)")

# ============================================
# 7. 总结
# ============================================
print("\n" + "=" * 70)
print("【7. 结果总结】")
print("=" * 70)
comparison = pd.DataFrame({
    '指标': ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1'],
    '决策树': [f"{roc_auc_score(yva, prob_dt):.4f}", f"{accuracy_score(yva, pred_dt):.4f}",
             f"{precision_score(yva, pred_dt):.4f}", f"{recall_score(yva, pred_dt):.4f}",
             f"{f1_score(yva, pred_dt):.4f}"],
    'XGBoost': [f"{roc_auc_score(yva, prob_xgb):.4f}", f"{accuracy_score(yva, pred_xgb):.4f}",
               f"{precision_score(yva, pred_xgb):.4f}", f"{recall_score(yva, pred_xgb):.4f}",
               f"{f1_score(yva, pred_xgb):.4f}"],
})
print(comparison.to_string(index=False))
print(f"\n推荐模型: {'XGBoost' if roc_auc_score(yva, prob_xgb) > roc_auc_score(yva, prob_dt) else '决策树'}")
print(f"验证段(6月)诚实 AUC: {max(roc_auc_score(yva, prob_dt), roc_auc_score(yva, prob_xgb)):.4f}")
print("说明: 该 AUC 为无泄漏时间切分评估, 与 v1 报告的 0.98 有本质区别。")
