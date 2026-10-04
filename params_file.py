# -*- coding: utf-8 -*-
"""参数文件读写模块（TOML 格式）。

为什么选 TOML 作为参数文件格式
-----------------------------
* 支持 ``#`` 注释：每个参数都能在文件里写清含义与单位，方便直接改；
* 用 ``[分组]`` 组织参数，平面 / 翼型 / 展向厚度 / 小翼 / 离散化一目了然；
* Python 3.11+ 标准库自带 ``tomllib``，**无需额外安装依赖**；
* 比 JSON 可读、比 YAML 更少歧义，且能表达浮点、整数、布尔、字符串和数组。

用法
----
    from params_file import ensure_param_file, load_param_dict

    path = ensure_param_file("bwb_params.toml")   # 缺失时写出默认文件
    data = load_param_dict(path)                  # -> 扁平字典 {参数名: 值}
"""

from __future__ import annotations

import tomllib
from pathlib import Path

__all__ = [
    "DEFAULT_TOML",
    "PARAM_FILE_NAME",
    "FLAT_GROUPS",
    "load_param_dict",
    "flatten_param_dict",
    "save_param_file",
    "ensure_param_file",
]

PARAM_FILE_NAME = "bwb_params.toml"

# --------------------------------------------------------------------------- #
# 默认参数文件内容（与 main.BWBParameters 的默认值一致）
# 用户直接编辑本文件即可，无需改动 Python 代码。
# --------------------------------------------------------------------------- #
DEFAULT_TOML = '''# =============================================================================
# 翼身融合（BWB）飞机参数文件
# 依据：戴浩, 余雄庆. 翼身融合飞机参数化几何模型[J]. 飞机设计, 2012.
# 说明：长度单位 mm，角度单位 度(deg)。修改后直接运行  python main.py  即可。
#      未在文件中出现的参数将使用程序内置默认值；写错参数名会在运行时提示并忽略。
# =============================================================================

[general]
# 预设型号："A" 或 "B"。预设仅用于填充本文件中**未列出**的参数；
# 本文件已列出全部参数，因此要切换型号请直接修改下面各组的数值
# （文件末尾附有模型 B 的参考值，复制过来覆盖即可）。
model = "A"
# 模型名称（用于输出 CSV 与 CATIA 特征命名）
name = "ModelA"


# -----------------------------------------------------------------------------
# (1) 平面参数：展长、弦长、后掠角、上反角、扭转角
# -----------------------------------------------------------------------------
[planform]
root_chord     = 4500.0   # 根部弦长
mid_chord      = 2500.0   # 中部（内外翼交界）弦长
tip_chord      = 500.0    # 翼尖弦长
inner_span     = 1150.0   # 内段展长（半模）
semi_span      = 4000.0   # 半展长
inner_sweep    = 60.0     # 内段前缘后掠角
outer_sweep    = 35.0     # 外段前缘后掠角
dihedral       = 0.0      # 上反角
incidence_root = 0.0      # 根部安装角（正值为抬头）
incidence_tip  = -2.0     # 翼尖安装角（负值为下扭）


# -----------------------------------------------------------------------------
# (2) 翼型参数：各剖面用 CST 方法（分类函数 + 形状函数）描述
# -----------------------------------------------------------------------------
[airfoil]
root_thickness = 0.20     # 根部相对厚度 t/c（融合体较厚）
mid_thickness  = 0.14     # 中部相对厚度 t/c
tip_thickness  = 0.10     # 翼尖相对厚度 t/c
camber_ratio   = 0.02     # 弯度比
camber_pos     = 0.40     # 最大弯度位置（0~1 的弦向比例）
reflex         = 1.0      # 反弯系数 0~1：1.0 = 后缘弯度斜率归零（完全反弯，
                          #   弯度大幅减小、利于配平）；0.5 = 部分反弯；0 = 不反弯
cst_order      = 8        # 伯恩斯坦多项式阶数（论文取 8，形状参数 = 阶数+1）
cst_n1         = 0.5      # 分类函数 N1：0.5 为圆头前缘
cst_n2         = 1.0      # 分类函数 N2：1.0 为尖形后缘
te_thickness   = 0.0025   # 后缘相对厚度（0 = 尖后缘）


# -----------------------------------------------------------------------------
# (3) 融合段展向厚度变化控制参数（三次 Hermite 曲线的端点斜率 d(t/c)/dy）
#     厚度曲线精确通过 根部/中部/翼尖 三点，斜率决定中间段的走向。
# -----------------------------------------------------------------------------
[span_thickness]
thickness_slope_root = 0.0        # 根部斜率（0 = 根部为最大厚度处）
thickness_slope_tip  = -0.00002   # 翼尖斜率


# -----------------------------------------------------------------------------
# (4) 翼梢小翼参数
# -----------------------------------------------------------------------------
[winglet]
winglet_height     = 540.0   # 小翼高度
winglet_taper      = 0.4     # 小翼尖削比（翼尖弦长 / 根弦）
winglet_sweep      = 55.0    # 小翼后掠角
winglet_cant       = 45.0    # 小翼倾斜角（相对铅垂线，0 = 竖直，90 = 水平）
winglet_incidence  = 5.0     # 小翼安装角
winglet_transition = 0.30    # 过渡段占小翼高度的比例


# -----------------------------------------------------------------------------
# (5) CST 形状函数参数 A_i —— 翼型形状的设计变量（气动优化用）
#     上表面权重 = 弯度权重 + 厚度权重；下表面 = 弯度权重 - 厚度权重（线性，可精确互换）
#     厚度权重按各剖面 t/c 缩放，弯度权重按 camber_ratio 缩放，后缘厚度保持一致
# -----------------------------------------------------------------------------
[cst_shape]
# "naca"    = 由 [airfoil] 的厚度/弯度参数自动生成基准翼型（下面的权重仅作参考）
# "weights" = 直接采用下面的 A_i 作为设计变量（气动优化迭代用这一档）
mode = "weights"
# 各剖面是否共用同一套形状参数：
#   "same"        = 共用（默认），仅按展向缩放厚度与弯度
#   "per_station" = 使用 [cst_shape.sections] 中逐剖面的权重（未列出的剖面回退到这里）
shape_mode = "same"
# 厚度形状参数 A_t（个数 = cst_order + 1）；决定厚度分布形状
thickness_weights = [0.292060, 0.237860, 0.333760, 0.155480, 0.340400, 0.182480, 0.258190, 0.232260, 0.241260]
# 弯度形状参数 A_c（个数 = cst_order + 1）；决定弯度线形状
camber_weights = [0.001270, 0.023890, -0.011140, 0.058750, -0.040480, 0.032960, -0.007060, 0.003280, -0.000110]

[cst_shape.sections]
# shape_mode = "same" 时本组留空；需要逐剖面形状控制时把上面改为 "per_station"，
# 并按下面格式为各剖面单独给出权重（未列出的剖面自动回退到共用权重）：
# root  = { thickness_weights = [ ... ], camber_weights = [ ... ] }
# inner = { thickness_weights = [ ... ], camber_weights = [ ... ] }
# tip   = { thickness_weights = [ ... ], camber_weights = [ ... ] }


# -----------------------------------------------------------------------------
# (6) 三次 Hermite 曲线控制参数 —— 平面外形与展向分布的设计变量
#     points = [[展向 y, 值], ...]   结点
#     slopes = [各结点切矢量斜率, ...]  （前缘/后缘为 dx/dy，展向厚度为 d(t/c)/dy）
#     留空（写成 []）时按 [planform] / [span_thickness] 的常规参数自动生成
# -----------------------------------------------------------------------------
[hermite.nose]
# 机头段前缘（俯视轮廓）：结点 [[展向 y, 前缘 x], ...]，y 从 0（机头顶点）开始向外交接。
# 给出后会自动合并到前缘 Hermite 曲线中（覆盖同展向区间），从而独立控制机头形状；
# 斜率留空则该曲线整体改用 Catmull-Rom 自动估计切矢量。
#   points = [[0.0, 0.0], [400.0, 420.0]]
#   slopes = [0.60, 1.20]
points = []
slopes = []

[hermite.leading_edge]
# 机翼段前缘结点 [[展向 y, 前缘 x], ...]，控制内外段后掠转折（与上面的机头段自动合并）
points = []
# 各结点 dx/dy（等于 tan 后掠角）；留空则用 [planform] 的内段/外段后掠角
slopes = []

[hermite.trailing_edge]
# 结点 [[展向 y, 后缘 x], ...]，直接控制后缘曲线形状与弦长分布
points = []
slopes = []

[hermite.span_thickness]
# 结点 [[展向 y, 相对厚度 t/c], ...]，控制融合段厚度变化
points = []
# 各结点 d(t/c)/dy；留空则用 [span_thickness] 的端点斜率
slopes = []


# -----------------------------------------------------------------------------
# 离散化与输出控制
# -----------------------------------------------------------------------------
[discretization]
n_section_pts   = 81     # 每个剖面的总点数（沿上下表面闭合一圈，越大越精细）
n_le_pts        = 61     # 前缘 / 后缘引导线采样点数（引导线按 Hermite 采样这么多点）
n_sections_span = 9      # 展向厚度曲线采样点数（仅用于打印与导出）
n_span_stations = 9      # 展向剖面数（含融合段）；越大越能体现融合段展向厚度变化


# =============================================================================
# 【参考】论文表 1 中模型 B 的取值：需要时把下面几行复制到上面各分组中覆盖。
# =============================================================================
# [general]
# name = "ModelB"
#
# [planform]
# root_chord  = 5000.0
# mid_chord   = 2500.0
# tip_chord   = 500.0
# inner_span  = 1200.0
# semi_span   = 4000.0
# inner_sweep = 60.0
# outer_sweep = 35.0
#
# [winglet]
# winglet_height    = 500.0
# winglet_taper     = 0.3
# winglet_sweep     = 60.0
# winglet_cant      = 30.0
# winglet_incidence = 2.0
'''


# --------------------------------------------------------------------------- #
# 读写函数
# --------------------------------------------------------------------------- #
# 「扁平分组」：这些分组里的键会被提到顶层，与 BWBParameters 的字段一一对应。
# 其余分组（cst_shape / hermite 等含数组或子表的）保持嵌套结构。
FLAT_GROUPS = (
    "general",
    "planform",
    "airfoil",
    "span_thickness",
    "winglet",
    "discretization",
)


def load_param_dict(path: str | Path = PARAM_FILE_NAME) -> dict:
    """读取 TOML 参数文件，返回与文件结构一致的嵌套字典。

    例：``[hermite.leading_edge] points = [...]`` →
    ``{"hermite": {"leading_edge": {"points": [...]}}}``
    """
    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


def flatten_param_dict(raw: dict, flat_groups: tuple = FLAT_GROUPS):
    """拆分为 ``(扁平参数字典, 结构化分组字典)``。

    * 扁平字典：``[planform] root_chord = 4500`` → ``{"root_chord": 4500.0}``，
      可直接用于 ``BWBParameters(**data)`` 式的赋值；
    * 结构化字典：``cst_shape`` / ``hermite`` 等含数组、嵌套子表的分组原样保留。
    """
    flat: dict = {}
    nested: dict = {}
    for key, value in raw.items():
        if isinstance(value, dict) and key in flat_groups:
            flat.update(value)
        else:
            nested[key] = value
    return flat, nested


def _fmt(value) -> str:
    """把 Python 值格式化成 TOML 字面量（支持标量与（嵌套）数组）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(round(float(value), 12))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    raise TypeError(f"无法写入参数文件的类型: {type(value)!r}")


def _write_table(lines: list, prefix: str, table: dict) -> None:
    """递归写出一个 TOML 表（先标量/数组，再子表；空表也保留表头）。"""
    leaves = {k: v for k, v in table.items() if not isinstance(v, dict)}
    subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
    if leaves or not subtables:
        lines.append(f"[{prefix}]")
        for k, v in leaves.items():
            lines.append(f"{k} = {_fmt(v)}")
    for k, v in subtables.items():
        lines.append("")
        _write_table(lines, f"{prefix}.{k}" if prefix else k, v)


def save_param_file(data: dict, path: str | Path = PARAM_FILE_NAME, header: str = "") -> Path:
    """把参数字典写回 TOML 文件（支持多层子表与数组）。

    供气动优化迭代批量生成参数文件：读入 → 修改权重/控制点 → 写出 → 重新建模。
    """
    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)

    lines: list = []
    if header:
        for line in header.rstrip().splitlines():
            lines.append(f"# {line}" if line else "#")

    for key, value in data.items():   # 顶层标量
        if not isinstance(value, dict):
            lines.append(f"{key} = {_fmt(value)}")
    for key, value in data.items():   # 顶层表
        if isinstance(value, dict):
            lines.append("")
            _write_table(lines, key, value)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def ensure_param_file(path: str | Path = PARAM_FILE_NAME) -> Path:
    """参数文件不存在时，按默认值写出一份带注释的文件，返回其路径。"""
    path = Path(path)
    if not path.exists():
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_TOML, encoding="utf-8")
        print(f"[提示] 未找到参数文件，已生成默认参数文件：{path.resolve()}")
    return path


if __name__ == "__main__":
    # 自检：生成（或读取）参数文件、拆分结构、回写并再次读取验证
    p = ensure_param_file(PARAM_FILE_NAME)
    raw = load_param_dict(p)
    flat, nested = flatten_param_dict(raw)
    print(f"参数文件：{p.resolve()}")
    print(f"扁平参数 {len(flat)} 个，结构化分组 {sorted(nested)}")
    for group, content in nested.items():
        subs = [k for k, v in content.items() if isinstance(v, dict)]
        leaves = [k for k, v in content.items() if not isinstance(v, dict)]
        print(f"  [{group}] 键 {leaves}，子表 {subs}")

    out = Path("_params_roundtrip.toml")
    save_param_file(raw, out, header="参数文件回写自检（临时文件）")
    again = load_param_dict(out)
    same = flatten_param_dict(again) == flatten_param_dict(raw)
    print(f"回写后重新读取一致：{same}  -> {out.resolve()}")
    out.unlink(missing_ok=True)
