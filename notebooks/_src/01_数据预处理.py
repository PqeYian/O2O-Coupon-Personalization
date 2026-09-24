# %% [markdown]
# # 01 · 数据预处理
#
# > O2O 优惠券个性化投放项目 —— 第 1 步
#
# ---
#
# ## 本节目标
#
# 把两份原始 CSV 洗成**结构统一、类型正确、可直接建模**的干净表，并完成两件定义性工作：
#
# 1. **正负样本定义**（`gap <= 15` → `class = 1`）
# 2. **建模样本范围**（只保留 `coupon_id` 非空的领券记录）
#
# ## 预处理的两层含义
#
# 需要特别说明：本项目流程里「预处理」出现在 EDA 之前，它承担的是**基础清洗**职能——不先做这一步，
# `date_received` 还是浮点数 `20160528.0`、`discount_rate` 还是 `'150:20'` 字符串，EDA 根本无从下手。
#
# 而另一层「建模用预处理」——**时间切分**与**标签时效窗口**——同样在本节完成定义，但其**切分口径的修正**
# 是第 07 节（解决数据泄漏）的核心内容。
#
# | 层次 | 内容 | 位置 |
# |---|---|---|
# | 基础清洗 | `'null'` → NaN、日期解析、折扣率统一 | 本节 3~4 |
# | 标签定义 | `gap <= 15`、样本范围 | 本节 5~6 |
# | 时间切分 | 按领券月份切训练/验证/测试 | 本节 7（口径在第 07 节修正） |

# %% [markdown]
# ## 1. 环境与路径
#
# 全部使用**相对路径**：从当前目录向上查找含 `.git` 的仓库根目录。
# 这样 notebook 不绑定任何绝对路径，克隆到任何机器都能直接跑。

# %%
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 160)
pd.set_option('display.unicode.east_asian_width', True)


def find_root(start=None):
    """向上查找仓库根目录(以 .git 为标志)，使 notebook 不依赖绝对路径"""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / '.git').exists():
            return cand
    return p


ROOT = find_root()
DATA_DIR = ROOT / 'data'
RESULTS = ROOT / 'results'
FIG_DIR = RESULTS / 'figures'
TAB_DIR = RESULTS / 'tables'
OUT_DIR = ROOT / 'data_out'          # 中间产物，gitignore 不入库
for d in (FIG_DIR, TAB_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 绘图统一配置
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 110
matplotlib.rcParams['savefig.bbox'] = 'tight'

# 统一配色（全项目共用）
C_MAIN, C_ALT, C_WARN, C_GREY = '#2F6F9F', '#E08A3C', '#C0504D', '#8C8C8C'

print(f'仓库根目录 : {ROOT}')
print(f'数据目录   : {DATA_DIR}')
print(f'中间产物   : {OUT_DIR}')

# %% [markdown]
# ## 2. 读取原始数据
#
# | 文件 | 行数 | 列数 | 说明 |
# |---|---|---|---|
# | `train.csv` | 1,648,881 | 7 | 含 `date`（到店消费日期），用于构建标签 |
# | `test.csv` | 100,669 | 6 | **无** `date` 列，是真正待预测的 7 月数据 |

# %%
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f'未找到原始数据: {TRAIN_PATH}\n'
        '请从天池 O2O 赛题下载 train.csv / test.csv 放入 data/ 目录。')

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)

print(f'train.csv : {train_raw.shape[0]:,} 行 x {train_raw.shape[1]} 列  列名={list(train_raw.columns)}')
print(f'test.csv  : {test_raw.shape[0]:,} 行 x {test_raw.shape[1]} 列  列名={list(test_raw.columns)}')
print()
print('train 前 5 行：')
print(train_raw.head())

# %% [markdown]
# ## 3. 基础清洗
#
# 原始数据有三处「类型陷阱」，都不是脏数据，而是**业务语义的编码方式**：
#
# | 问题 | 原始形态 | 处理 |
# |---|---|---|
# | 缺失被编码成字符串 | `'null'`（而非空值） | 统一转成 `NaN` / `NaT` |
# | 日期是浮点数 | `20160528.0` | 还原成 `datetime64` |
# | 折扣率两种量纲混用 | `'0.8'`（折扣）与 `'150:20'`（满减） | 满减折算为折扣率 |

# %%
def clean_nulls(df):
    """把字符串 'null' 统一还原为缺失值"""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(
                lambda x: np.nan if isinstance(x, str) and x.strip().lower() == 'null' else x)
    return df


def to_date(x):
    """20160528.0 -> Timestamp('2016-05-28')"""
    if pd.isna(x):
        return pd.NaT
    s = str(x).split('.')[0].strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')


def to_discount(x):
    """'0.8' -> 0.8 ; '150:20' -> 0.87 (满 150 减 20) ; 'null' -> NaN"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if ':' in s:
        a, b = s.split(':')
        return round((float(a) - float(b)) / float(a), 2)
    try:
        return float(s)
    except ValueError:
        return np.nan


train = clean_nulls(train_raw)
test = clean_nulls(test_raw)

for df in (train, test):
    df['date_received'] = df['date_received'].map(to_date)
    if 'date' in df.columns:
        df['date'] = df['date'].map(to_date)
    df['discount_rate'] = df['discount_rate'].map(to_discount)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')

train['dataset'] = 'train'
test['dataset'] = 'test'
# test 没有 date 列，补一个空列，方便两表上下拼接
if 'date' not in test.columns:
    test['date'] = pd.NaT

print('清洗后 train 前 5 行：')
print(train.head())
print()
print('清洗后各列类型：')
print(train.dtypes.to_string())

# %% [markdown]
# ## 4. 派生时间字段与缺失结构
#
# ### 缺失结构：这里的缺失不是「数据质量差」，而是业务行为的编码
#
# | 字段 | 缺失含义 |
# |---|---|
# | `coupon_id` 为空 | 用户**没领券**直接到店消费 |
# | `date` 为空 | 领了券但**没核销** |
# | 两者都为空 | 既没领券也没消费（无效记录） |
#
# 所以**不能删、不能填**，只能按语义分别处理。

# %%
for df in (train, test):
    # gap = 核销间隔天数（未核销为 NaN）
    df['gap'] = (df['date'] - df['date_received']).dt.days
    df['receive_year'] = df['date_received'].dt.year
    df['receive_month'] = df['date_received'].dt.month
    df['receive_weekday'] = df['date_received'].dt.weekday
    df['consume_month'] = df['date'].dt.month

print('train 派生字段后形状:', train.shape)
print()
print('gap 描述统计：')
print(train['gap'].describe().round(2).to_string())

# %%
# 缺失结构表 -> results/tables
miss_rows = []
for name, df in [('train', train), ('test', test)]:
    for c in ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
              'date_received', 'date']:
        if c in df.columns:
            n = int(df[c].isna().sum())
            miss_rows.append({'数据集': name, '字段': c, '缺失数': n,
                              '缺失率(%)': round(n / len(df) * 100, 2), '总行数': len(df)})
missing_df = pd.DataFrame(miss_rows)
missing_df.to_csv(TAB_DIR / '01_missing_structure.csv', index=False, encoding='utf-8-sig')

print('缺失结构（train）：')
print(missing_df[missing_df['数据集'] == 'train'].to_string(index=False))

# %% [markdown]
# ## 5. 记录类型分布 —— 标签映射的依据
#
# 把每一行按「有没有领券」×「有没有核销」映射到四类业务行为，这是后续所有建模决策的地基：

# %%
has_coupon = train['coupon_id'].notna()
has_date = train['date'].notna()
is_valid_consume = has_coupon & has_date

record_type = pd.Series('无效记录', index=train.index)
record_type[~has_coupon & has_date] = '未领券(直接消费)'
record_type[has_coupon & ~has_date] = '领券未核销'
record_type[has_coupon & has_date] = '领券且核销'

rt = (record_type.value_counts().rename_axis('记录类型').reset_index(name='数量'))
rt['占比(%)'] = (rt['数量'] / len(train) * 100).round(2)
rt.to_csv(TAB_DIR / '01_record_types.csv', index=False, encoding='utf-8-sig')

print(f'train 总行数: {len(train):,}')
print(rt.to_string(index=False))
print()
print(f'有效消费记录 : {has_date.sum():,}')
print(f'有效领券记录 : {has_coupon.sum():,}')
print(f'领券后核销   : {is_valid_consume.sum():,}')

# %% [markdown]
# ## 6. 标签定义
#
# ### 正负样本
#
# ```
# gap = (date - date_received).days         # 核销间隔（未核销为 NaN）
# class = 1  if  gap <= 15                  # 领券后 15 天内到店核销
# class = 0  其他                            # 未核销 或 15 天后才核销
# 样本  = 仅 coupon_id 非空的行               # 未领券记录不构成样本
# ```
#
# ### 为什么是 15 天：这是**业务约定，而非数据边界**
#
# 论证分三层，主次必须分清：
#
# **① 主位（业务，规范性）** —— 两周是公认的转化黄金窗口。领券后核销概率随时间衰减，两周后的边际转化价值大幅下降，
# 划为「无效慢转化」在业务上可解释，且与天池赛题官方口径（15 天内核销）一致。
#
# **② 确认（数据，弱）** —— `gap` 分布在 15 天附近**平滑连续、无反常结构**。
# 注意这条证据的方向：它只能说明「取 15 不是一个被数据打脸的硬切点」，**不能**用来「发现」边界。
#
# **③ 确认（数据，强）** —— `gap <= 15` 覆盖了绝大多数有效核销，切在这里不丢主要转化（具体覆盖率见第 02 节的 gap 分布实证）。
#
# > ⚠️ **必须澄清的常见错误论证**：分布中**并不存在**「第 15 天之后券失效」的业务断层。
# > 真正的陡降点在 **day9→10** 和 **day14→15**，而 **day15→16 反而是平滑的**；
# > 且第 16 天起仍有上万笔核销、`gap` 最远延伸到 96 天。详见第 02 节实测。
#
# ### 为什么不用「核销即正」
#
# 标签是**投放决策的判据**（多久算「及时」），不是对「券是否起作用」的裁决。三条理由：
#
# 1. **原始数据没有券有效期字段** —— 7 列里不含任何「过期日」，所以「第 16 天仍有核销」只能说明数据集未设过期，
#    测不出券的真实生命周期；「15 天」本就不是对券有效期的刻画。
# 2. **「券能被核销」≠「算有效转化」** —— `gap>15` 的核销确实发生、券确实起了作用（物理事实），
#    但投放是**当期决策**：让利成本当月付出，消费却发生在活动周期之外，对「这期该不该给他发券」没有指导意义。
# 3. **反馈闭环的可操作性** —— 若按「核销即正」，`gap` 最长 96 天，5 月领券的样本要等到 8 月才能确定标签，
#    训练与投放的反馈周期被拉到 3 个月，不可操作；15 天窗口让标签两周内即可确定。

# %%
for df in (train, test):
    df['class'] = 0
    df.loc[df['gap'] <= 15, 'class'] = 1

# 建模样本：只保留领券记录
train_model = train[train['coupon_id'].notna()].copy()
test_model = test[test['coupon_id'].notna()].copy()

print(f'建模样本(train) : {len(train_model):,} 行')
print(f'  正样本(class=1): {int((train_model["class"] == 1).sum()):,}  '
      f'({(train_model["class"] == 1).mean() * 100:.2f}%)')
print(f'  负样本(class=0): {int((train_model["class"] == 0).sum()):,}  '
      f'({(train_model["class"] == 0).mean() * 100:.2f}%)')
print()
print('负样本内部拆解（这是标签口径的关键）：')
neg = train_model[train_model['class'] == 0]
neg_unused = int(neg['date'].isna().sum())
neg_late = int(neg['date'].notna().sum())
print(f'  领券未核销      : {neg_unused:,}')
print(f'  15 天后才核销   : {neg_late:,}')
print()
print(f'测试样本(test)  : {len(test_model):,} 行（无标签，待预测 7 月）')

# %%
# 标签与记录类型对照表 -> results/tables
label_map = pd.DataFrame([
    {'记录类型': '未领券(仅消费)', 'coupon_id': '空', 'date': '有值', 'class': '—(不进样本)',
     '数量': int((~has_coupon & has_date).sum())},
    {'记录类型': '领券未核销', 'coupon_id': '有值', 'date': '空', 'class': '0',
     '数量': int((has_coupon & ~has_date).sum())},
    {'记录类型': '领券·15 天内核销', 'coupon_id': '有值', 'date': '有值', 'class': '1',
     '数量': int((has_coupon & has_date & (train['gap'] <= 15)).sum())},
    {'记录类型': '领券·15 天后核销', 'coupon_id': '有值', 'date': '有值', 'class': '0',
     '数量': int((has_coupon & has_date & (train['gap'] > 15)).sum())},
])
label_map['占比(%)'] = (label_map['数量'] / len(train) * 100).round(2)
label_map.to_csv(TAB_DIR / '01_label_mapping.csv', index=False, encoding='utf-8-sig')
print(label_map.to_string(index=False))

# %% [markdown]
# ## 7. 时间切分口径（初步）
#
# 领券日期覆盖 `2016-01-01 ~ 2016-06-15`，而 `test.csv` 是 **2016 年 7 月**的待预测数据。
# 天然的时间切分方案：
#
# ```
# train.csv : 1-5 月  ->  训练池
#             6 月    ->  验证集（模拟「用历史预测未来」）
# test.csv  : 7 月    ->  最终测试集
# ```
#
# > 本节只**定义**切分口径。第 03~05 节的 v1 版本会**故意使用随机切分**（这正是泄漏来源之一），
# > 第 07 节再把它改回时间切分并说明为什么必须如此。

# %%
train_pool = train_model[train_model['receive_month'] <= 5]
val_seg = train_model[train_model['receive_month'] == 6]

print('按领券月份分布（train_model）：')
print(train_model['receive_month'].value_counts().sort_index().to_string())
print()
print(f'训练池(1-5 月) : {len(train_pool):,} 行   正样本率 {train_pool["class"].mean() * 100:.2f}%')
print(f'验证段(6 月)   : {len(val_seg):,} 行   正样本率 {val_seg["class"].mean() * 100:.2f}%')
print(f'测试段(test)   : {len(test_model):,} 行  （7 月，无标签）')

# %% [markdown]
# ## 8. 导出清洗结果
#
# 输出到 `data_out/`（gitignore，不入库）：

# %%
cols_keep = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance',
             'date_received', 'date', 'gap', 'class', 'dataset',
             'receive_year', 'receive_month', 'receive_weekday', 'consume_month']

train_model[cols_keep].to_csv(OUT_DIR / 'clean_train.csv', index=False)
test_model[cols_keep].to_csv(OUT_DIR / 'clean_test.csv', index=False)

print('已导出：')
for f in ['clean_train.csv', 'clean_test.csv']:
    p = OUT_DIR / f
    print(f'  {f}  ({p.stat().st_size / 1024 / 1024:.1f} MB)')

# %% [markdown]
# ## 本节小结
#
# | 结论 | 内容 |
# |---|---|
# | **样本范围** | 建模样本 = 947,279 条领券记录（未领券的 70 万条不构成样本） |
# | **标签定义** | `gap <= 15` → 正样本，正样本率约 6%，**严重类别不平衡** |
# | **缺失即业务** | `coupon_id` / `date` 的高缺失对应「未领券」「未核销」，不能删也不能填 |
# | **切分口径** | 已定义 1-5 月 / 6 月 / 7 月三段，但 v1 将故意偏离为随机切分 |
#
# 下一步 → **02 数据探索 EDA**：用实证数据检验上面的标签口径，并可视化 gap 分布的真实结构。
