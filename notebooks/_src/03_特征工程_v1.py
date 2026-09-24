# %% [markdown]
# # 03 · 特征工程（v1）
#
# > O2O 优惠券个性化投放项目 —— 第 3 步
#
# ---
#
# ## 本节目标
#
# 把**事件粒度**的原始表展开成**主体画像**，让模型能泛化到没见过的用户 / 商户 / 券。
#
# ## 7.1 节思想：多粒度聚合
#
# 原始数据一行 = 一次领券事件。单看这一行，能用的信息只有距离和折扣率——太薄了。
# 而「这张券会不会被核销」**主要由「主体是谁」决定**，所以要把事件行展开成三份画像：
#
# ```
# 事件行 = 用户画像 + 商户画像 + 券画像 + 事件属性(距离/折扣/时间)
#             │           │          │
#         按user_id   按merchant_id  按coupon_id
#         分组聚合      分组聚合       分组聚合
# ```
#
# | 粒度 | 特征 | 业务假设 |
# |---|---|---|
# | 用户 | `user_use_coupon_rate` ★ | 用户越爱用券 → 越可能核销（归一化剔活跃度，纯券敏感度） |
# | 用户 | `user_use_coupon_times` / `user_consume_times` | 用券 / 消费活跃度 |
# | 用户 | `user_receive_unused` | 领了不用 → 大概率再浪费 |
# | 用户 | `user_mean_interval` | 历史核销越快 → 冲动 / 刚需 |
# | 商户 | `merchant_launch_coupon_used_rate` | 商户券转化率高 → 更可信 |
# | 商户 | `merchant_launch_coupon_count` / `merchant_mean_interval` | 发券活跃度 / 券核销速度 |
# | 券 | `coupon_used_rate` | 券本身受欢迎度 |
# | 券 | `coupon_fifteen_used` | 券历史正样本数 |
# | 事件 | `distance` | 便利度（`discount_rate` 在 v1 中被错误排除） |
#
# ## 特征构建三原则
#
# 1. **多粒度聚合**：把事件行展开成用户 + 商户 + 券画像
# 2. **归一化优先**：次数 → 比率，剔除活跃度混淆，得到纯净的「券敏感度」
# 3. **ID 抽象成画像**：才能泛化到没见过的用户 / 券（直接 one-hot ID 做不到）
#
# ---
#
# > ### ⚠️ 本节是本次项目最重要的一处「雷区」
# >
# > 下面这段代码**看起来完全正常**，但它埋了三颗雷：
# >
# > 1. **聚合在「全量数据」上做 groupby** —— 当前行自己的核销结果被算进了自己的特征
# > 2. **`user_mean_interval` 的分子 `date - date_received` 与标签 `gap` 是同一个量** —— 目标泄漏
# > 3. **`coupon_used_rate` 的分子分母取自同一子集** —— 比率恒等于 1，退化成常数
# >
# > 本节先按 v1 原样实现（保持诚实，不做事后美化），**第 06 节会把这些雷逐一点出来**。

# %%
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 180)
pd.set_option('display.unicode.east_asian_width', True)


def find_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / '.git').exists():
            return cand
    return p


ROOT = find_root()
OUT_DIR = ROOT / 'data_out'
TAB_DIR = ROOT / 'results' / 'tables'
TAB_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT_DIR / 'clean_train.csv', parse_dates=['date_received', 'date'])
print(f'读入建模样本: {len(df):,} 行')
print(f'正样本率: {(df["class"] == 1).mean() * 100:.2f}%')

# %% [markdown]
# ## 1. 聚合的公共准备
#
# 三个粒度都用 `groupby` 把事件行汇总成主体画像。
#
# > 注意这里的聚合范围是**全量数据**——不区分这一行属于谁、发生在什么时候。
# > 这正是泄漏的来源：它把「含自身、含未来」的信息汇总到了每一行上。

# %%
agg_src = df.copy()
agg_src['_use_coupon'] = (df[['date_received', 'date']].count(axis=1) == 2).astype(int)   # 领券且核销
agg_src['_consumed'] = df['date'].notna().astype(int)                                     # 有核销
agg_src['_received_unused'] = (df['coupon_id'].notna() & df['date'].isna()).astype(int)   # 领了没用

print(f'聚合源表: {agg_src.shape}')

# %% [markdown]
# ## 2. 用户级聚合

# %%
g = agg_src.groupby('user_id')

du = pd.DataFrame({
    'user_use_coupon_times': g['_use_coupon'].sum(),
    'user_consume_times': g['_consumed'].sum(),
    'user_receive_unused': g['_received_unused'].sum(),
    'user_mean_interval': g['gap'].mean(),        # ⚠️ 泄漏元凶：gap 的用户级均值
}).reset_index()
du['user_use_coupon_rate'] = (du['user_use_coupon_times'] / du['user_consume_times']).fillna(0)
du['user_mean_interval'] = du['user_mean_interval'].fillna(du['user_mean_interval'].max() + 1)

print(f'用户画像: {du.shape[0]:,} 个用户 x {du.shape[1] - 1} 个特征')
print(du.head().to_string(index=False))

# %% [markdown]
# ## 3. 商户级聚合

# %%
gm = agg_src.groupby('merchant_id')

dm = pd.DataFrame({
    'merchant_launch_coupon_used_count': gm['_use_coupon'].sum(),
    'merchant_consume_times': gm['_consumed'].sum(),
    'merchant_launch_coupon_count': gm['coupon_id'].count(),
    'merchant_receive_unused': gm['_received_unused'].sum(),
    'merchant_mean_interval': gm['gap'].mean(),    # ⚠️ 同上
}).reset_index()
dm['merchant_launch_coupon_used_rate'] = (
    dm['merchant_launch_coupon_used_count'] / dm['merchant_consume_times']).fillna(0)
dm['merchant_mean_interval'] = dm['merchant_mean_interval'].fillna(dm['merchant_mean_interval'].max() + 1)

print(f'商户画像: {dm.shape[0]:,} 个商户 x {dm.shape[1] - 1} 个特征')
print(dm.head().to_string(index=False))

# %% [markdown]
# ## 4. 券级聚合
#
# > **`coupon_used_rate` 在这里就写坏了**：分子与分母取自**同一个 `gap<=15` 子集**，
# > 所以对「有过 15 天内核销」的券比值恒为 1；对没有的券则因对齐产生 NaN 被填成 0。
# > 结果这个比率**退化成 `coupon_fifteen_used > 0` 的二值指示器**，与 `coupon_fifteen_used`
# > 完全冗余、零增量信息。下面的诊断会当场验证（但 v1 当时没有察觉）。

# %%
c15 = agg_src[agg_src['gap'] <= 15]              # gap<=15 子集（未核销为 NaN，比较为 False 被排除）
gc = agg_src.groupby('coupon_id')
g15 = c15.groupby('coupon_id')

dc = pd.DataFrame({
    'coupon_receive_times': gc['coupon_id'].count(),
    'coupon_consume_times': gc['_consumed'].sum(),
    'coupon_fifteen_used': g15.size(),
}).reset_index()
dc['coupon_fifteen_used'] = dc['coupon_fifteen_used'].fillna(0)

# ⚠️ v1 的 bug：分子与分母取自同一个 gap<=15 子集
#    分子 = 该券 15 天内核销数
#    分母 = 该券在 gap<=15 子集内的记录数（恒等于分子）
#    正确做法见第 08 节：分母应改为该券【总核销次数】
dc['coupon_used_rate'] = (dc['coupon_id'].map(g15.size()) /
                          dc['coupon_id'].map(g15.size())).fillna(0)

print(f'券画像: {dc.shape[0]:,} 种券 x {dc.shape[1] - 1} 个特征')
print(dc.head().to_string(index=False))

# %%
print('⚠️  退化特征诊断：coupon_used_rate 的取值分布')
vc = dc['coupon_used_rate'].value_counts()
print(vc.to_string())
print()
n_one, n_zero = int((dc['coupon_used_rate'] == 1).sum()), int((dc['coupon_used_rate'] == 0).sum())
print(f'恒为 1 的券: {n_one:,} / {len(dc):,} = {n_one / len(dc) * 100:.1f}%   '
      f'（= 有过 15 天内核销的券）')
print(f'恒为 0 的券: {n_zero:,} / {len(dc):,} = {n_zero / len(dc) * 100:.1f}%   '
      f'（= 没有 15 天内核销的券）')
print()
print('=> 该比率只有两个取值，等价于指示器 (coupon_fifteen_used > 0)，')
print('   与 coupon_fifteen_used 完全冗余 —— 零增量信息的退化特征（v1 未察觉）')

# %% [markdown]
# ## 5. 拼接画像到事件行

# %%
def merge_features(seg):
    m = seg.merge(du, on='user_id', how='left')
    m = m.merge(dm, on='merchant_id', how='left')
    m = m.merge(dc, on='coupon_id', how='left')
    return m


feats = merge_features(df)
print(f'拼接后: {feats.shape}')

# %%
# 补全缺失：未出现的实体置 0；比率类特征本就是 NaN -> 0
agg_cols = ([c for c in du.columns if c != 'user_id'] +
            [c for c in dm.columns if c != 'merchant_id'] +
            [c for c in dc.columns if c != 'coupon_id'])
feats[agg_cols] = feats[agg_cols].fillna(0)
feats['distance'] = feats['distance'].fillna(feats['distance'].median())

print('缺失值检查:', int(feats[agg_cols + ["distance"]].isna().sum().sum()))

# %% [markdown]
# ## 6. 特征清单
#
# 注意：**`discount_rate` 被 v1 错误排除**了（当时的排除列表里写了它），这是个建模失误——
# 第 08 节的最终版会把它恢复。

# %%
FEATURES_V1 = [
    'distance',
    'user_use_coupon_times', 'user_consume_times', 'user_use_coupon_rate',
    'user_receive_unused', 'user_mean_interval',
    'merchant_launch_coupon_used_count', 'merchant_launch_coupon_used_rate',
    'merchant_launch_coupon_count', 'merchant_receive_unused', 'merchant_mean_interval',
    'coupon_receive_times', 'coupon_consume_times', 'coupon_fifteen_used', 'coupon_used_rate',
]
print(f'v1 特征数: {len(FEATURES_V1)}')
for i, f in enumerate(FEATURES_V1, 1):
    print(f'  {i:>2}. {f}')

# %%
feat_desc = pd.DataFrame([
    {'粒度': '事件', '特征': 'distance', '业务含义': '用户到商户距离（便利度）'},
    {'粒度': '用户', '特征': 'user_use_coupon_times', '业务含义': '用券消费次数（券活跃度）'},
    {'粒度': '用户', '特征': 'user_consume_times', '业务含义': '总消费次数（整体活跃度）'},
    {'粒度': '用户', '特征': 'user_use_coupon_rate', '业务含义': '用券/总消费 比率（券敏感度）'},
    {'粒度': '用户', '特征': 'user_receive_unused', '业务含义': '领了没用的次数（浪费倾向）'},
    {'粒度': '用户', '特征': 'user_mean_interval', '业务含义': '历史核销间隔均值 ⚠️ 目标泄漏'},
    {'粒度': '商户', '特征': 'merchant_launch_coupon_used_count', '业务含义': '商户券被核销数'},
    {'粒度': '商户', '特征': 'merchant_launch_coupon_used_rate', '业务含义': '商户券核销比率'},
    {'粒度': '商户', '特征': 'merchant_launch_coupon_count', '业务含义': '商户发券数（发券活跃度）'},
    {'粒度': '商户', '特征': 'merchant_receive_unused', '业务含义': '商户券领而未用数'},
    {'粒度': '商户', '特征': 'merchant_mean_interval', '业务含义': '商户券核销间隔均值 ⚠️ 目标泄漏'},
    {'粒度': '券', '特征': 'coupon_receive_times', '业务含义': '券被领取次数'},
    {'粒度': '券', '特征': 'coupon_consume_times', '业务含义': '券被核销次数'},
    {'粒度': '券', '特征': 'coupon_fifteen_used', '业务含义': '券 15 天内核销数 ⚠️ 标签聚合'},
    {'粒度': '券', '特征': 'coupon_used_rate', '业务含义': '券核销比率 ⚠️ 退化：只有 0/1 两值，等价于指示器'},
])
feat_desc.to_csv(TAB_DIR / '03_features_v1.csv', index=False, encoding='utf-8-sig')
print(feat_desc.to_string(index=False))

# %% [markdown]
# ## 7. 导出

# %%
keep = ['user_id', 'merchant_id', 'coupon_id', 'date_received'] + FEATURES_V1 + ['class']
feats[keep].to_csv(OUT_DIR / 'feats_v1.csv', index=False)
print(f'已导出: feats_v1.csv  {feats[keep].shape}')

# %%
print('=' * 70)
print('v1 特征工程完成')
print('=' * 70)
print(f'特征数    : {len(FEATURES_V1)}')
print(f'样本数    : {len(feats):,}')
print(f'正样本率  : {feats["class"].mean() * 100:.2f}%')
print()
print('⚠️  遗留隐患（第 06 节揭穿）:')
print('   1. 聚合在全量数据上做 -> 含自身行 + 含未来信息')
print('   2. user_mean_interval / merchant_mean_interval 与标签同源 -> 目标泄漏')
print('   3. coupon_used_rate 退化成二值指示器 -> 零增量信息')
print('   4. coupon_fifteen_used 本质是"标签在券粒度的聚合"')

# %% [markdown]
# ## 本节小结
#
# 从**业务假设**出发的多粒度聚合方向是对的——把事件行展开成用户/商户/券画像，确实能捕捉
# 「主体是谁」这一核销的决定性因素。三条构建原则（多粒度、归一化、ID 抽象成画像）也站得住。
#
# **问题出在「聚合的数据范围」上**：这些画像是在全量数据上算的，于是每一行都偷看了自己、
# 也偷看了未来。
#
# 下一步 → **04 模型结果（v1）**：用这套特征建个模，看看会发生什么。
