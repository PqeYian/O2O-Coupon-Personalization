# %% [markdown]
# # 02 · 数据探索 EDA
#
# > O2O 优惠券个性化投放项目 —— 第 2 步
#
# ---
#
# ## 本节目标
#
# 用实证数据回答四个问题：
#
# | # | 问题 | 为什么重要 |
# |---|---|---|
# | 1 | `gap` 分布长什么样？**「15 天」在数据里到底是什么位置** | 直接决定标签口径能否自圆其说 |
# | 2 | 类别有多不平衡？ | 决定用什么评估指标 |
# | 3 | 距离 / 折扣率与核销的关系 | 决定这些字段值不值得进模型 |
# | 4 | 时间维度上有什么结构？ | 决定能不能按时间切分 |
#
# > **本节最有价值的产出是第 1 问**：它会推翻一个流传很广的错误论证——
# > 「15 天附近分布平滑，所以 15 天是业务边界」。实测数据表明**方向恰好相反**。

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 160)
pd.set_option('display.unicode.east_asian_width', True)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 110
matplotlib.rcParams['savefig.bbox'] = 'tight'


def find_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / '.git').exists():
            return cand
    return p


ROOT = find_root()
OUT_DIR = ROOT / 'data_out'
FIG_DIR = ROOT / 'results' / 'figures'
TAB_DIR = ROOT / 'results' / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

C_MAIN, C_ALT, C_WARN, C_GREY = '#2F6F9F', '#E08A3C', '#C0504D', '#8C8C8C'

df = pd.read_csv(OUT_DIR / 'clean_train.csv', parse_dates=['date_received', 'date'])
print(f'读入 {len(df):,} 行 x {df.shape[1]} 列')
print(f'领券日期范围: {df["date_received"].min().date()} ~ {df["date_received"].max().date()}')
print(f'核销日期范围: {df["date"].min().date()} ~ {df["date"].max().date()}')

# %% [markdown]
# ## 1. 描述性统计与数据概览

# %%
print('=' * 70)
print('数值型变量描述统计')
print('=' * 70)
print(df[['discount_rate', 'distance', 'gap']].describe().round(2).to_string())

# %%
has_coupon = df['coupon_id'].notna()
has_date = df['date'].notna()

overview = pd.DataFrame([
    {'指标': '总行数（领券记录，本节分析对象）', '值': f'{len(df):,}'},
    {'指标': '已核销（有 date）', '值': f'{has_date.sum():,}'},
    {'指标': '未核销（无 date）', '值': f'{(~has_date).sum():,}'},
    {'指标': '用户数', '值': f'{df["user_id"].nunique():,}'},
    {'指标': '商户数', '值': f'{df["merchant_id"].nunique():,}'},
    {'指标': '优惠券种类数', '值': f'{df["coupon_id"].nunique():,}'},
])
print('注：本节数据已在 01 步过滤为「领券记录」建模样本，')
print('    train.csv 全量口径下的 768,767 条有效消费记录已不含在内。')
print(overview.to_string(index=False))

# %% [markdown]
# ## 2. 【核心】gap 分布：15 天在数据里到底是什么位置
#
# `gap = date - date_received`，即**核销间隔天数**。只在「领券且核销」的 67,165 条记录上有值。
#
# ### 2.1 逐日核销笔数

# %%
gap_dist = df.loc[df['gap'].notna(), 'gap'].astype(int).value_counts().sort_index()
gap_dist = gap_dist.reindex(range(0, int(gap_dist.index.max()) + 1), fill_value=0)

day15 = int(gap_dist.loc[15])
day16 = int(gap_dist.loc[16])
within15 = int(gap_dist.loc[:15].sum())
total_used = int(gap_dist.sum())

print(f'gap 有效记录数: {total_used:,}')
print()
print('逐日核销笔数（day 1-30）：')
for start in range(1, 31, 10):
    seg = gap_dist.loc[start:min(start + 9, 30)]
    print(f'  day{start:>2}-{min(start + 9, 30):>2}: ' +
          '  '.join(f'{int(v):>5}' for v in seg.values))

print()
print(f'day15 = {day15:,}   day16 = {day16:,}')
print(f'gap<=15 核销 = {within15:,} / {total_used:,} = {within15 / total_used * 100:.2f}%')
print(f'gap>=16 核销 = {total_used - within15:,} ({(total_used - within15) / total_used * 100:.2f}%)')
print(f'gap 最大值 = {int(gap_dist.index.max())} 天')

# %% [markdown]
# ### 2.2 关键发现：15→16 是**平滑**的，最陡的断层根本不在 15 天
#
# 先看流传最广的那条论证——**「day15 = 896 → day16 = 824，平滑衰减，所以 15 天是业务边界」**。
#
# 数字本身属实，但**推理方向是反的**：
#
# - **如果是真边界**（比如券在第 15 天到期），第 16 天应该出现**断崖**（几乎没有核销），而不是平滑衰减。
# - **平滑恰恰说明 15 天是人为约定**——没有任何机制在第 15/16 天之间「切断」用户行为。
# - 更致命的是：**长尾里几乎处处都这么平滑**。day11→12、day16→17、day17→18 的环比变化都在同一个量级。
#   拿一个「哪都成立」的性质去论证「15 天特殊」，逻辑上不成立。

# %%
pct_change = gap_dist.pct_change() * 100
print('环比变化率（%）—— 找真正的陡降点：')
for d in range(2, 21):
    if d in pct_change.index and not np.isnan(pct_change.loc[d]):
        mark = '  <== 陡降' if pct_change.loc[d] < -25 else ''
        print(f'  day{d:>2}->{d:<2}: {pct_change.loc[d]:>7.1f}%   '
              f'(day{d}={int(gap_dist.loc[d]):>5}, day{d + 1}={int(gap_dist.loc[d + 1]):>5}){mark}')

print()
print('对比：15 天前后 vs 真正最陡的位置')
print(f'  day14->15 : {pct_change.loc[15]:.1f}%')
print(f'  day15->16 : {pct_change.loc[16]:.1f}%   <== 平滑，不是断层')
print(f'  day09->10 : {pct_change.loc[10]:.1f}%   <== 全区间最陡')

# %% [markdown]
# ### 2.3 分布真正的结构是 **7 天周期**，与 15 天无关

# %%
print('局部峰值检验（核销笔数）：')
for d in [7, 14, 15, 16, 21, 28]:
    prev_v = int(gap_dist.loc[d - 1]) if d - 1 >= 0 else 0
    next_v = int(gap_dist.loc[d + 1]) if d + 1 <= gap_dist.index.max() else 0
    cur = int(gap_dist.loc[d])
    is_peak = cur > prev_v and cur > next_v
    print(f'  day{d:>2} = {cur:>5}  (前一日 {prev_v:>5}, 后一日 {next_v:>5})'
          f'{"   <== 局部峰" if is_peak else ""}')

print()
print('结论: day7 / day14 / day21 / day28 全是局部峰值 —— 这是典型的"周末效应"，')
print('      周期长度 7 天。15 天既不是峰值也不是谷值，在分布中没有任何特殊性。')

# %% [markdown]
# ### 2.4 那「15 天」凭什么成立：三层论证
#
# 数据探索的结论**不是**「15 天有边界」，而是「15 天是个合理的业务切点」。主次必须分清：

# %%
cover = within15 / total_used * 100
arg = pd.DataFrame([
    {'层次': '① 主位（业务，规范性）', '内容': '两周是公认的转化黄金窗口；与天池赛题官方口径一致',
     '证据类型': '业务约定', '强度': '—'},
    {'层次': '② 确认（数据，弱）', '内容': f'gap 分布在 15 天附近平滑连续、无反常结构（day15={day15} -> day16={day16}）',
     '证据类型': '未被数据违背', '强度': '弱'},
    {'层次': '③ 确认（数据，强）', '内容': f'gap<=15 覆盖 {within15:,}/{total_used:,} = {cover:.2f}% 的核销，不丢主要转化',
     '证据类型': '覆盖率', '强度': '强'},
])
arg.to_csv(TAB_DIR / '02_gap_layering_argument.csv', index=False, encoding='utf-8-sig')
print(arg.to_string(index=False))

print()
print('⚠️  注意: 第②条的证据方向。它只能说明"取 15 不是被数据打脸的硬切点"，')
print('    绝不能用作"发现边界"的依据 —— 平滑是长尾的普遍性质，不是 15 天的特征。')

# %% [markdown]
# ### 2.5 可视化：gap 分布与真正的断崖

# %%
fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# 左：逐日分布
ax = axes[0]
ax.bar(gap_dist.index, gap_dist.values, color=C_MAIN, alpha=0.85, width=0.8)
ax.axvline(15.5, color=C_WARN, ls='--', lw=1.6)
ax.text(16.2, gap_dist.max() * 0.88, '15 天切点', color=C_WARN, fontsize=11, fontweight='bold')
ax.set_xlabel('核销间隔 gap（天）')
ax.set_ylabel('核销笔数')
ax.set_title(f'gap 逐日分布（n={total_used:,}）', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 中：环比变化率（暴露真正的断层）
ax = axes[1]
d_range = range(2, 31)
vals = [pct_change.loc[d] if d in pct_change.index else np.nan for d in d_range]
colors = [C_WARN if (not np.isnan(v) and v < -25) else C_MAIN for v in vals]
ax.bar(list(d_range), vals, color=colors, alpha=0.85, width=0.8)
ax.axhline(0, color='k', lw=0.8)
ax.axvline(15.5, color=C_GREY, ls='--', lw=1.2)
ax.set_xlabel('day d -> d+1')
ax.set_ylabel('环比变化率 (%)')
ax.set_title('环比变化率：红柱 = 陡降 (>25%)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for d in [10, 15]:
    if d in pct_change.index:
        ax.annotate(f'day{d - 1}->{d}\n{pct_change.loc[d]:.0f}%',
                    xy=(d, pct_change.loc[d]), xytext=(d + 1.5, pct_change.loc[d] - 6),
                    fontsize=9, color=C_WARN,
                    arrowprops=dict(arrowstyle='->', color=C_WARN, lw=1))

# 右：累积覆盖
ax = axes[2]
cum = gap_dist.cumsum() / total_used * 100
ax.plot(cum.index, cum.values, color=C_MAIN, lw=2.2)
ax.axvline(15, color=C_WARN, ls='--', lw=1.6)
ax.scatter([15], [cover], color=C_WARN, zorder=5, s=60)
ax.annotate(f'gap<=15 覆盖 {cover:.2f}%', xy=(15, cover), xytext=(24, cover - 22),
            fontsize=11, color=C_WARN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_WARN, lw=1.2))
ax.set_xlabel('gap 阈值（天）')
ax.set_ylabel('累计核销覆盖率 (%)')
ax.set_title('累积覆盖曲线', fontweight='bold')
ax.grid(alpha=0.3)
ax.set_ylim(0, 102)

plt.tight_layout()
fig.savefig(FIG_DIR / '02_gap_distribution.png', dpi=140)
plt.show()

# %%
rows = [{'gap': int(d), '核销笔数': int(v),
         '环比变化(%)': round(float(pct_change.loc[d]), 2) if (d in pct_change.index and not np.isnan(pct_change.loc[d])) else None,
         '累积覆盖率(%)': round(float(cum.loc[d]), 2)}
        for d, v in gap_dist.items() if d >= 1]
pd.DataFrame(rows).to_csv(TAB_DIR / '02_gap_distribution.csv', index=False, encoding='utf-8-sig')
print(f'已保存: 02_gap_distribution.png / 02_gap_distribution.csv')

# %% [markdown]
# ## 3. 类别不平衡：为什么 Accuracy 是个陷阱

# %%
model_df = df[df['coupon_id'].notna()]
pos = int((model_df['class'] == 1).sum())
neg = int((model_df['class'] == 0).sum())
pos_rate = pos / len(model_df) * 100

print(f'建模样本 : {len(model_df):,}')
print(f'  正样本 : {pos:,}  ({pos_rate:.2f}%)')
print(f'  负样本 : {neg:,}  ({100 - pos_rate:.2f}%)')
print(f'  不平衡比: 1 : {neg / pos:.1f}')
print()
print('陷阱演示：一个把所有样本都预测为"未核销"的模型：')
print(f'  Accuracy = {(neg / len(model_df)) * 100:.2f}%   <- 看起来很高！')
print(f'  Recall   = 0.00%       <- 一个正样本都抓不到')
print('  => 不平衡场景下 Accuracy 无意义，必须用 AUC 等排序指标。')

fig, ax = plt.subplots(figsize=(7, 4.6))
bars = ax.bar(['负样本 (class=0)', '正样本 (class=1)'], [neg, pos],
              color=[C_GREY, C_WARN], alpha=0.9, width=0.55)
for b, v in zip(bars, [neg, pos]):
    ax.annotate(f'{v:,}\n({v / len(model_df) * 100:.2f}%)',
                xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('样本数')
ax.set_title(f'类别不平衡：正样本仅 {pos_rate:.2f}%', fontweight='bold')
ax.set_ylim(0, neg * 1.18)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / '02_class_imbalance.png', dpi=140)
plt.show()

# %% [markdown]
# ## 4. 距离与折扣率：这两个字段值不值得进模型？

# %%
print('distance 取值分布（Top 10）：')
dist_vc = df['distance'].value_counts().sort_index()
print(dist_vc.head(10).to_string())
print(f'... 最大距离 = {df["distance"].max()}')

# %%
print('折扣率取值分布（Top 10）：')
dr_vc = df['discount_rate'].value_counts().sort_index()
print(dr_vc.to_string())

# %%
# 各折扣率的核销率对比（只看领券记录）
coupon_df = df[df['coupon_id'].notna()]
by_dr = coupon_df.groupby('discount_rate').agg(
    样本数=('class', 'size'), 核销率=('class', 'mean')).reset_index()
by_dr['核销率(%)'] = (by_dr['核销率'] * 100).round(2)
by_dr = by_dr[['discount_rate', '样本数', '核销率(%)']].sort_values('样本数', ascending=False)
print('各折扣率的核销率：')
print(by_dr.to_string(index=False))

# %%
by_dist = coupon_df.groupby('distance').agg(
    样本数=('class', 'size'), 核销率=('class', 'mean')).reset_index()
by_dist['核销率(%)'] = (by_dist['核销率'] * 100).round(2)
by_dist = by_dist[['distance', '样本数', '核销率(%)']].sort_values('distance')
print('各距离的核销率：')
print(by_dist.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
bd = by_dist[by_dist['样本数'] > 100]
ax.plot(bd['distance'], bd['核销率(%)'], 'o-', color=C_MAIN, lw=2, ms=6)
ax.set_xlabel('用户到商户距离')
ax.set_ylabel('核销率 (%)')
ax.set_title('距离 vs 核销率（距离越近核销率越高）', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes[1]
bd2 = by_dr.sort_values('discount_rate')
ax.bar(bd2['discount_rate'].astype(str), bd2['核销率(%)'], color=C_ALT, alpha=0.9, width=0.6)
for i, (v, n) in enumerate(zip(bd2['核销率(%)'], bd2['样本数'])):
    ax.annotate(f'{v:.1f}%\nn={n:,}', xy=(i, v), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=8)
ax.set_xlabel('折扣率')
ax.set_ylabel('核销率 (%)')
ax.set_title('折扣率 vs 核销率', fontweight='bold')
ax.set_ylim(0, bd2['核销率(%)'].max() * 1.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / '02_distance_discount.png', dpi=140)
plt.show()

# %% [markdown]
# ## 5. 时间维度：能不能按时间切分？

# %%
monthly = df.groupby('receive_month').agg(
    领券数=('coupon_id', lambda s: s.notna().sum()),
    核销数=('class', 'sum')).reset_index()
monthly['核销率(%)'] = (monthly['核销数'] / monthly['领券数'] * 100).round(2)

consume_monthly = df[df['date'].notna()].groupby('consume_month').size().rename('消费数').reset_index()
consume_monthly.columns = ['月份', '消费数']

print('各月领券/核销情况：')
print(monthly.to_string(index=False))
print()
print('各月消费情况：')
print(consume_monthly.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
x = np.arange(len(monthly))
w = 0.38
ax.bar(x - w / 2, monthly['领券数'], w, label='领券数', color=C_MAIN, alpha=0.9)
ax.bar(x + w / 2, monthly['核销数'], w, label='核销数', color=C_ALT, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f'{int(m)} 月' for m in monthly['receive_month']])
ax.set_ylabel('次数')
ax.set_title('各月领券与核销', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
ax.plot(consume_monthly['月份'], consume_monthly['消费数'], 'o-',
        color=C_MAIN, lw=2.2, ms=7)
for m, v in zip(consume_monthly['月份'], consume_monthly['消费数']):
    ax.annotate(f'{v:,}', xy=(m, v), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9)
ax.set_xlabel('月份')
ax.set_ylabel('消费次数')
ax.set_title('各月消费趋势', fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / '02_monthly_trend.png', dpi=140)
plt.show()

# %% [markdown]
# ## 6. 关键洞察汇总
#
# | # | 洞察 | 对建模的影响 |
# |---|---|---|
# | 1 | **`gap` 分布是长尾 + 7 天周期，15 天无特殊结构** | 「15 天」只能作为业务约定，不能包装成数据边界 |
# | 2 | **day9→10 (−41%) / day14→15 (−40%) 才是最陡断层，day15→16 反而平滑** | 切点选择必须靠业务论证，数据只能做「确认」 |
# | 3 | **`gap<=15` 覆盖 84.96% 的核销** | 15 天不丢主要转化，这是它最强的一条数据证据 |
# | 4 | **正样本仅 6.02%，不平衡比 1:15.6** | 必须用 AUC 评估 + 类别权重 |
# | 5 | **缺失即业务**（`coupon_id` 42.55% / `date` 53.38%） | 不能删不能填，要按语义分别处理 |
# | 6 | **距离越近核销率越高** | `distance` 应保留为特征 |
# | 7 | **领券集中在 1/5 月（年初与五一），6 月开始下滑；`test` 是 7 月** | 时间切分（用历史预测未来）在业务上成立 |

# %%
insights = pd.DataFrame([
    {'#': 1, '洞察': 'gap 分布为长尾 + 7 天周期(day7/14/21/28 是峰)，15 天无特殊结构',
     '对建模的影响': '15 天只能作业务约定，不能包装成数据边界'},
    {'#': 2, '洞察': f'最陡断层在 day9->10 ({pct_change.loc[10]:.0f}%) 与 day14->15 ({pct_change.loc[15]:.0f}%)，day15->16 仅 {pct_change.loc[16]:.0f}%',
     '对建模的影响': '切点靠业务论证，数据只做确认'},
    {'#': 3, '洞察': f'gap<=15 覆盖 {cover:.2f}% 的核销',
     '对建模的影响': '15 天不丢主要转化（最强数据证据）'},
    {'#': 4, '洞察': f'正样本仅 {pos_rate:.2f}%，不平衡比 1:{neg / pos:.1f}',
     '对建模的影响': '必须用 AUC 评估 + 类别权重'},
    {'#': 5, '洞察': 'coupon_id 缺失 42.55%、date 缺失 53.38%，缺失即业务行为',
     '对建模的影响': '不能删不能填，按语义分别处理'},
    {'#': 6, '洞察': '距离越近核销率越高；折扣率与核销率有区分度',
     '对建模的影响': 'distance / discount_rate 应保留'},
    {'#': 7, '洞察': '领券集中 1、5 月，6 月下滑；test 为 7 月',
     '对建模的影响': '时间切分（历史预测未来）业务上成立'},
])
insights.to_csv(TAB_DIR / '02_key_insights.csv', index=False, encoding='utf-8-sig')

print('=' * 70)
print('EDA 结论已汇总至 results/tables/02_key_insights.csv')
print('=' * 70)
print(insights.to_string(index=False))

# %% [markdown]
# ## 本节小结
#
# EDA 最重要的收获**不是发现 15 天是边界，而是确认它不是**——
# 数据里挑不出任何特殊点，7/14/21/28 天反而是周期峰，真正的断崖在 day9→10 和 day14→15。
#
# 因此「15 天」的正确位置是：**业务上可解释、被数据确认不违背、且不丢主要转化的人为约定**。
#
# 下一步 → **03 特征工程（v1）**：开始构建多粒度聚合特征。
# 注意——**v1 的特征工程将埋下数据泄漏**，第 06 节会把它揪出来。
