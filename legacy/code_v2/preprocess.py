# -*- coding: utf-8 -*-
"""
O2O优惠券个性化投放 - v2 无泄漏数据预处理与特征工程
=====================================================
相比 v1 的关键修复:
  1. 时间切分: 1-5月训练 / 6月验证 / 7月测试(真实预测场景), 不再随机切分
  2. 无泄漏特征工程: 用户/商户/优惠券聚合特征只在【训练段】计算,
     验证段与测试段的特征值完全来自训练段历史, 不包含自身行, 不含未来信息
  3. 修复 coupon_used_rate bug: 分母改为该券"总核销次数",
     分子为该券"15天内核销次数", 比率不再恒等于 1
  4. 恢复 discount_rate 特征 (v1 错误地将其排除在模型外)
  5. 新增时间类特征 (领券日星期/是否周末)

用法:
  python preprocess.py
输出(存到 ./data/):
  train_feats.csv  val_feats.csv  test_feats.csv
"""
import pandas as pd
import numpy as np
import os

# 数据来源解析顺序:
#   1) 环境变量 O2O_DATA_DIR 指向的目录
#   2) 本文件所在目录下的 data/ 子目录
#   请先将原始 train.csv / test.csv 放到上述任一目录
_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_SRC = os.environ.get("O2O_DATA_DIR", _DEFAULT)
os.makedirs(DATA_SRC, exist_ok=True)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# 1. 读取原始数据并合并清洗
# ============================================
print("=" * 70)
print("【1. 读取原始数据并清洗】")
print("=" * 70)

_train_p, _test_p = os.path.join(DATA_SRC, "train.csv"), os.path.join(DATA_SRC, "test.csv")
if not (os.path.exists(_train_p) and os.path.exists(_test_p)):
    raise FileNotFoundError(
        f"未找到原始数据。请将 train.csv / test.csv 放入目录: {DATA_SRC}\n"
        "或设置环境变量 O2O_DATA_DIR 指向数据所在目录。")
train_raw = pd.read_csv(_train_p)
test_raw = pd.read_csv(_test_p)
print(f"数据来源: {DATA_SRC}")
print(f"原始 train: {train_raw.shape}, test: {test_raw.shape}")

# 训练样本和测试样本合并，方便统一清洗
data = pd.concat([train_raw, test_raw], axis=0, join='outer', ignore_index=True)

# 前5列是数值型(user/merchant/coupon/discount/distance), "null"字符串 -> NaN
data.iloc[:, :5] = data.iloc[:, :5].map(lambda x: np.nan if x == 'null' else x)
# 后2列是日期型(date_received/date)
data.iloc[:, 5:] = data.iloc[:, 5:].map(lambda x: None if x == 'null' else x)

# 日期转为 datetime (原始为浮点 20160528.0)
for col in ['date_received', 'date']:
    data[col] = data[col].astype('str').str.split('.').str[0]
    data[col] = pd.to_datetime(data[col], errors='coerce')

# 满减优惠统一改写为折扣率形式, 如 '150:20' -> 0.87
data['discount_rate'] = data['discount_rate'].fillna('null')


def discount(x):
    if ':' in str(x):
        a, b = str(x).split(':')
        return round((int(a) - int(b)) / int(a), 2)
    elif x == 'null' or pd.isna(x):
        return np.nan
    else:
        return float(x)


data['discount_rate'] = data['discount_rate'].map(discount)
data['distance'] = pd.to_numeric(data['distance'], errors='coerce')

# ============================================
# 2. 时间切分: 训练/验证/测试
#    - 训练段: 领券月在 1-5 月 (含领券月缺失的未领券记录)
#    - 验证段: 领券月 = 6 月
#    - 测试段: 领券月 = 7 月 (真实待预测样本)
# ============================================
print("\n【2. 时间切分】")
month = data['date_received'].dt.month
train_seg = data[(month.isna()) | (month <= 5)].copy()
val_seg = data[month == 6].copy()
test_seg = data[month == 7].copy()
print(f"训练段: {train_seg.shape}, 验证段: {val_seg.shape}, 测试段: {test_seg.shape}")

# ============================================
# 3. 无泄漏聚合特征工程 (只在训练段计算)
# ============================================
print("\n【3. 无泄漏聚合特征工程 (仅训练段)】")
train_seg['gap'] = (train_seg['date'] - train_seg['date_received']).dt.days
both = train_seg[['date_received', 'date']].count(axis=1) == 2       # 领券且核销
has_date = train_seg['date'].notnull()                               # 有核销
has_coupon = train_seg['coupon_id'].notnull()                        # 有领券


def grp_sum(s, key):
    """按 key 分组求和 (自动对齐索引)"""
    s = s.dropna()
    key = key.loc[s.index]
    return s.groupby(key).sum()


# ---- 用户级聚合 ----
uid = train_seg.dropna(subset=['user_id'])['user_id']
du = pd.DataFrame({
    'user_use_coupon_times':   grp_sum(both.astype(int), uid),
    'user_consume_times':      grp_sum(has_date.astype(int), uid),
    'user_receive_unused':     grp_sum(((has_coupon) & (~has_date)).astype(int), uid),
    'user_mean_interval':      grp_sum(train_seg['gap'], uid),
})
du = du.reset_index().rename(columns={'index': 'user_id'})
du['user_use_coupon_rate'] = (du['user_use_coupon_times'] / du['user_consume_times']).fillna(0)
du['user_mean_interval'] = du['user_mean_interval'].fillna(du['user_mean_interval'].max() + 1)
print(f"用户数: {len(du)}")

# ---- 商户级聚合 ----
mid = train_seg.dropna(subset=['merchant_id'])['merchant_id']
dm = pd.DataFrame({
    'merchant_launch_coupon_used_count': grp_sum(both.astype(int), mid),
    'merchant_consume_times':            grp_sum(has_date.astype(int), mid),
    'merchant_launch_coupon_count':      grp_sum(has_coupon.astype(int), mid),
    'merchant_receive_unused':           grp_sum(((has_coupon) & (~has_date)).astype(int), mid),
    'merchant_mean_interval':            grp_sum(train_seg['gap'], mid),
})
dm = dm.reset_index().rename(columns={'index': 'merchant_id'})
dm['merchant_launch_coupon_used_rate'] = (
    dm['merchant_launch_coupon_used_count'] / dm['merchant_consume_times']).fillna(0)
dm['merchant_mean_interval'] = dm['merchant_mean_interval'].fillna(dm['merchant_mean_interval'].max() + 1)
print(f"商户数: {len(dm)}")

# ---- 优惠券级聚合 (修复 coupon_used_rate) ----
cid = train_seg.dropna(subset=['coupon_id'])['coupon_id']
c = train_seg.dropna(subset=['coupon_id'])
day15 = (c['date'] - c['date_received']).dt.days <= 15            # 15天内核销(正样本)
dc = pd.DataFrame({
    'coupon_receive_times':    grp_sum(has_coupon.loc[c.index].astype(int), cid),
    'coupon_consume_times':    grp_sum(c['date'].notnull().astype(int), cid),
    'coupon_fifteen_used':     grp_sum(day15.astype(int), cid),
})
dc = dc.reset_index().rename(columns={'index': 'coupon_id'})
# 修复: 分母=该券总核销次数(不限15天), 分子=15天内核销次数
dc['coupon_used_rate'] = (dc['coupon_fifteen_used'] / dc['coupon_consume_times']).fillna(0)
print(f"优惠券数: {len(dc)}")


def merge_features(seg):
    """把训练段算好的聚合特征拼到某个时段上"""
    m = seg.merge(du, on='user_id', how='left')
    m = m.merge(dm, on='merchant_id', how='left')
    m = m.merge(dc, on='coupon_id', how='left')
    return m


train_seg = merge_features(train_seg)
val_seg = merge_features(val_seg)
test_seg = merge_features(test_seg)

# 未出现在训练段历史中的实体 -> 聚合特征置 0
agg_cols = [c for c in du.columns if c != 'user_id'] + \
           [c for c in dm.columns if c != 'merchant_id'] + \
           [c for c in dc.columns if c != 'coupon_id']
for d in (train_seg, val_seg, test_seg):
    d[agg_cols] = d[agg_cols].fillna(0)

# ============================================
# 4. 时间类特征
# ============================================
for d in (train_seg, val_seg, test_seg):
    d['received_weekday'] = d['date_received'].dt.weekday
    d['received_is_weekend'] = (d['date_received'].dt.weekday >= 5).astype(int)
    d['received_day'] = d['date_received'].dt.day
    d['distance'] = d['distance'].fillna(d['distance'].median())

# ============================================
# 5. 构建标签并过滤未领券样本
# ============================================
print("\n【4. 构建标签并过滤】")
for d in (train_seg, val_seg, test_seg):
    d['class'] = 0
    d.loc[(d['date'] - d['date_received']).dt.days <= 15, 'class'] = 1
    d.dropna(subset=['coupon_id'], inplace=True)

FEATURES = ['discount_rate', 'distance',
            'received_weekday', 'received_is_weekend', 'received_day',
            'user_use_coupon_times', 'user_consume_times', 'user_use_coupon_rate',
            'user_receive_unused', 'user_mean_interval',
            'merchant_launch_coupon_used_count', 'merchant_launch_coupon_used_rate',
            'merchant_launch_coupon_count', 'merchant_receive_unused', 'merchant_mean_interval',
            'coupon_receive_times', 'coupon_consume_times', 'coupon_fifteen_used',
            'coupon_used_rate']
print(f"特征数量: {len(FEATURES)}")
print("特征:", FEATURES)

# 验证集/测试集中未在训练段出现的折扣率组合置为中位数(稳健处理)
med_dr = train_seg['discount_rate'].median()
for d in (train_seg, val_seg, test_seg):
    d['discount_rate'] = d['discount_rate'].fillna(med_dr)

# ============================================
# 6. 导出
# ============================================
train_seg[FEATURES + ['class']].to_csv(os.path.join(OUT_DIR, 'train_feats.csv'), index=False)
val_seg[FEATURES + ['class']].to_csv(os.path.join(OUT_DIR, 'val_feats.csv'), index=False)
# 测试表额外保留标识列, 供提交文件使用
test_seg[['user_id', 'merchant_id', 'coupon_id', 'date_received'] + FEATURES].to_csv(
    os.path.join(OUT_DIR, 'test_feats.csv'), index=False)

print("\n输出已保存至:", OUT_DIR)
print(f"训练: {len(train_seg)}  (正样本率 {train_seg['class'].mean():.4f})")
print(f"验证: {len(val_seg)}  (正样本率 {val_seg['class'].mean():.4f})")
print(f"测试: {len(test_seg)}")
