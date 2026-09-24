# %% [markdown]
# # 08 · 最终特征工程
#
# > O2O 优惠券个性化投放项目 —— 第 8 步
#
# ---
#
# ## 本节目标
#
# 第 07 节解决了**泄漏**（数据范围 + 切分方式）。本节在这个**无泄漏框架**下，
# 把特征定义本身遗留的三个问题也一并修掉，得到**最终特征集**。
#
# ## 相对 v1 的三项修复
#
# | # | 问题 | v1 的做法 | 最终版的做法 |
# |---|---|---|---|
# | 1 | **退化特征** `coupon_used_rate` | 分子分母取自同一 `gap<=15` 子集，退化成 0/1 指示器 | 分母改为该券**总核销次数**（不限 15 天），恢复成真正的比率 |
# | 2 | **误删** `discount_rate` | 被写进了排除列表，白白丢掉一个有区分度的字段 | **恢复**（第 02 节已证明折扣率与核销率相关） |
# | 3 | **缺失时间特征** | 完全没用领券时间的周期信息 | **新增** 领券星期 / 是否周末 / 领券日 |
#
# ## 无泄漏框架（继承自第 07 节）
#
# ```
# 训练段(1-5月) -> 算画像 + 训练
# 验证段(6月)   -> 评估（特征全部来自训练段历史）
# 测试段(7月)   -> 最终预测
# 所有 groupby 只在训练段上执行
# ```

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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

C_MAIN, C_ALT, C_WARN, C_GREY = '#2F6F9F', '#E08A3C', '#C0504D', '#8C8C8C'

# %% [markdown]
# ## 1. 时间切分与数据准备

# %%
df_train = pd.read_csv(OUT_DIR / 'clean_train.csv', parse_dates=['date_received', 'date'])
df_test = pd.read_csv(OUT_DIR / 'clean_test.csv', parse_dates=['date_received', 'date'])

train_seg = df_train[df_train['receive_month'] <= 5].copy()
val_seg = df_train[df_train['receive_month'] == 6].copy()
test_seg = df_test.copy()

print(f'训练段: {len(train_seg):,}   验证段: {len(val_seg):,}   测试段: {len(test_seg):,}')

# %% [markdown]
# ## 2. 无泄漏画像聚合（含修复 #1）
#
# > **修复 #1 的细节**：`coupon_used_rate` 的分母从「同一 `gap<=15` 子集内的记录数」
# > 改为「**该券在训练段内的总核销次数**（不限 15 天）」。这样比率才有真实分布。

# %%
s = train_seg.copy()
s['_use_coupon'] = (s[['date_received', 'date']].count(axis=1) == 2).astype(int)
s['_consumed'] = s['date'].notna().astype(int)
s['_received_unused'] = (s['coupon_id'].notna() & s['date'].isna()).astype(int)

# ---- 用户级 ----
g = s.groupby('user_id')
du = pd.DataFrame({
    'user_use_coupon_times': g['_use_coupon'].sum(),
    'user_consume_times': g['_consumed'].sum(),
    'user_receive_unused': g['_received_unused'].sum(),
    'user_mean_interval': g['gap'].mean(),
}).reset_index()
du['user_use_coupon_rate'] = (du['user_use_coupon_times'] / du['user_consume_times']).fillna(0)
du['user_mean_interval'] = du['user_mean_interval'].fillna(du['user_mean_interval'].max() + 1)

# ---- 商户级 ----
gm = s.groupby('merchant_id')
dm = pd.DataFrame({
    'merchant_launch_coupon_used_count': gm['_use_coupon'].sum(),
    'merchant_consume_times': gm['_consumed'].sum(),
    'merchant_launch_coupon_count': gm['coupon_id'].count(),
    'merchant_receive_unused': gm['_received_unused'].sum(),
    'merchant_mean_interval': gm['gap'].mean(),
}).reset_index()
dm['merchant_launch_coupon_used_rate'] = (
    dm['merchant_launch_coupon_used_count'] / dm['merchant_consume_times']).fillna(0)
dm['merchant_mean_interval'] = dm['merchant_mean_interval'].fillna(dm['merchant_mean_interval'].max() + 1)

# ---- 券级（★ 修复 #1：分母改为总核销次数）----
c15 = s[s['gap'] <= 15]
gc, g15 = s.groupby('coupon_id'), c15.groupby('coupon_id')
dc = pd.DataFrame({
    'coupon_receive_times': gc['coupon_id'].count(),
    'coupon_consume_times': gc['_consumed'].sum(),
    'coupon_fifteen_used': g15.size(),
}).reset_index()
dc['coupon_fifteen_used'] = dc['coupon_fifteen_used'].fillna(0)
# 修复：分母 = 该券【总核销次数】(不限 15 天)，不再是 15 天子集内的记录数
dc['coupon_used_rate'] = (dc['coupon_fifteen_used'] / dc['coupon_consume_times']).fillna(0)

print(f'用户画像 {len(du):,} / 商户画像 {len(dm):,} / 券画像 {len(dc):,}')
print()
print('★ 修复 #1 效果对比：coupon_used_rate 的取值分布')
print('  v1（退化）  : 只有 2 个取值（0 或 1），等价于指示器 (coupon_fifteen_used > 0)')
vc_new = dc['coupon_used_rate'].round(2).value_counts().sort_index()
print(f'  最终版      : {vc_new.shape[0]} 个不同取值，范围 [{dc["coupon_used_rate"].min():.3f}, {dc["coupon_used_rate"].max():.3f}]，'
      f'均值 {dc["coupon_used_rate"].mean():.3f}')

# %%
print('修复后的 coupon_used_rate 分布（前 12 个取值）：')
print(vc_new.head(12).to_string())
print()
print(f'与 coupon_fifteen_used 的相关系数: '
      f'{dc["coupon_used_rate"].corr(dc["coupon_fifteen_used"]):.3f}  (不再是完全冗余的单调关系)')

# %% [markdown]
# ## 3. 拼接 + 修复 #2 #3

# %%
def attach(seg):
    m = seg.merge(du, on='user_id', how='left')
    m = m.merge(dm, on='merchant_id', how='left')
    m = m.merge(dc, on='coupon_id', how='left')
    return m


train_seg = attach(train_seg)
val_seg = attach(val_seg)
test_seg = attach(test_seg)

AGG_COLS = ([c for c in du.columns if c != 'user_id'] +
            [c for c in dm.columns if c != 'merchant_id'] +
            [c for c in dc.columns if c != 'coupon_id'])

for seg in (train_seg, val_seg, test_seg):
    # ★ 修复 #3：新增时间特征
    seg['received_weekday'] = seg['date_received'].dt.weekday
    seg['received_is_weekend'] = (seg['date_received'].dt.weekday >= 5).astype(int)
    seg['received_day'] = seg['date_received'].dt.day
    seg[AGG_COLS] = seg[AGG_COLS].fillna(0)

# ★ 修复 #2：恢复 discount_rate（用训练段中位数填充）
med_dr = train_seg['discount_rate'].median()
med_dist = train_seg['distance'].median()
for seg in (train_seg, val_seg, test_seg):
    seg['discount_rate'] = seg['discount_rate'].fillna(med_dr)
    seg['distance'] = seg['distance'].fillna(med_dist)

print(f'折扣率中位数（训练段）: {med_dr}')
print(f'缺失值检查: {int(train_seg[AGG_COLS].isna().sum().sum())}')

# %% [markdown]
# ## 4. 最终特征清单

# %%
FEATURES_FINAL = [
    # 事件属性
    'discount_rate', 'distance',
    # 时间特征（新增）
    'received_weekday', 'received_is_weekend', 'received_day',
    # 用户画像
    'user_use_coupon_times', 'user_consume_times', 'user_use_coupon_rate',
    'user_receive_unused', 'user_mean_interval',
    # 商户画像
    'merchant_launch_coupon_used_count', 'merchant_launch_coupon_used_rate',
    'merchant_launch_coupon_count', 'merchant_receive_unused', 'merchant_mean_interval',
    # 券画像
    'coupon_receive_times', 'coupon_consume_times', 'coupon_fifteen_used', 'coupon_used_rate',
]

final_desc = pd.DataFrame([
    {'粒度': '事件', '特征': 'discount_rate', '业务含义': '折扣率（v1 误删，已恢复）', '相对 v1': '★ 恢复'},
    {'粒度': '事件', '特征': 'distance', '业务含义': '用户到商户距离', '相对 v1': '保持'},
    {'粒度': '时间', '特征': 'received_weekday', '业务含义': '领券是星期几', '相对 v1': '+ 新增'},
    {'粒度': '时间', '特征': 'received_is_weekend', '业务含义': '是否周末领券', '相对 v1': '+ 新增'},
    {'粒度': '时间', '特征': 'received_day', '业务含义': '领券是该月第几天', '相对 v1': '+ 新增'},
    {'粒度': '用户', '特征': 'user_use_coupon_times', '业务含义': '用券消费次数', '相对 v1': '保持'},
    {'粒度': '用户', '特征': 'user_consume_times', '业务含义': '总消费次数', '相对 v1': '保持'},
    {'粒度': '用户', '特征': 'user_use_coupon_rate', '业务含义': '用券/总消费 比率', '相对 v1': '保持'},
    {'粒度': '用户', '特征': 'user_receive_unused', '业务含义': '领了没用的次数', '相对 v1': '保持'},
    {'粒度': '用户', '特征': 'user_mean_interval', '业务含义': '历史核销间隔均值（范围已限制在训练段）', '相对 v1': '范围修复'},
    {'粒度': '商户', '特征': 'merchant_launch_coupon_used_count', '业务含义': '商户券被核销数', '相对 v1': '保持'},
    {'粒度': '商户', '特征': 'merchant_launch_coupon_used_rate', '业务含义': '商户券核销比率', '相对 v1': '保持'},
    {'粒度': '商户', '特征': 'merchant_launch_coupon_count', '业务含义': '商户发券数', '相对 v1': '保持'},
    {'粒度': '商户', '特征': 'merchant_receive_unused', '业务含义': '商户券领而未用数', '相对 v1': '保持'},
    {'粒度': '商户', '特征': 'merchant_mean_interval', '业务含义': '商户券核销间隔均值（范围已限制）', '相对 v1': '范围修复'},
    {'粒度': '券', '特征': 'coupon_receive_times', '业务含义': '券被领取次数', '相对 v1': '保持'},
    {'粒度': '券', '特征': 'coupon_consume_times', '业务含义': '券被核销次数', '相对 v1': '保持'},
    {'粒度': '券', '特征': 'coupon_fifteen_used', '业务含义': '券 15 天内核销数', '相对 v1': '保持'},
    {'粒度': '券', '特征': 'coupon_used_rate', '业务含义': '券核销比率（分母已修复）', '相对 v1': '★ 修复'},
])
final_desc.to_csv(TAB_DIR / '08_features_final.csv', index=False, encoding='utf-8-sig')

print(f'最终特征数: {len(FEATURES_FINAL)}  (v1 为 15 个)')
print()
print('特征总览：')
print(final_desc.to_string(index=False))

# %%
fix_summary = pd.DataFrame([
    {'修复项': '1. coupon_used_rate 退化', 'v1 的做法': '分子分母取自同一 gap<=15 子集 -> 退化成 0/1 指示器',
     '最终版': '分母改为该券总核销次数(不限 15 天)'},
    {'修复项': '2. discount_rate 被误删', 'v1 的做法': '写进了排除列表，字段完全没用上',
     '最终版': '恢复为特征（第 02 节已证明有区分度）'},
    {'修复项': '3. 缺时间周期特征', 'v1 的做法': '未使用领券时间的周期信息',
     '最终版': '新增 星期 / 是否周末 / 当月第几天'},
])
fix_summary.to_csv(TAB_DIR / '08_fix_summary.csv', index=False, encoding='utf-8-sig')
print(fix_summary.to_string(index=False))

# %% [markdown]
# ## 5. 导出最终特征表

# %%
train_seg[FEATURES_FINAL + ['class']].to_csv(OUT_DIR / 'final_train.csv', index=False)
val_seg[FEATURES_FINAL + ['class']].to_csv(OUT_DIR / 'final_val.csv', index=False)
test_seg[['user_id', 'merchant_id', 'coupon_id', 'date_received'] + FEATURES_FINAL].to_csv(
    OUT_DIR / 'final_test.csv', index=False)

print('已导出最终特征表：')
for f in ['final_train.csv', 'final_val.csv', 'final_test.csv']:
    p = OUT_DIR / f
    print(f'  {f:<18} {p.stat().st_size / 1024 / 1024:>6.1f} MB')

# %% [markdown]
# ## 6. 与 v1 特征集的对照

# %%
compare_features = pd.DataFrame([
    {'维度': '泄漏控制', 'v1': '全量 groupby（含自身行 + 未来），随机切分',
     '最终版': '聚合只在训练段，按时间切分'},
    {'维度': '券核销比率', 'v1': '退化成 0/1 指示器', '最终版': '真实比率，有连续分布'},
    {'维度': '折扣率', 'v1': '被误删', '最终版': '恢复'},
    {'维度': '时间特征', 'v1': '无', '最终版': '星期 / 周末 / 当月第几天'},
    {'维度': '特征数', 'v1': '15', '最终版': str(len(FEATURES_FINAL))},
])
compare_features.to_csv(TAB_DIR / '08_v1_vs_final_features.csv', index=False, encoding='utf-8-sig')
print(compare_features.to_string(index=False))

# %% [markdown]
# ## 本节小结
#
# | 项 | 结果 |
# |---|---|
# | **继承** | 第 07 节的无泄漏框架（时间切分 + 仅训练段聚合） |
# | **修复 #1** | `coupon_used_rate` 分母改为总核销次数，恢复真实分布 |
# | **修复 #2** | 恢复被误删的 `discount_rate` |
# | **修复 #3** | 新增 3 个时间周期特征 |
# | **最终特征数** | 19 个 |
#
# 下一步 → **09 最终模型结果**：用最终特征集重新建模评估。
