# -*- coding: utf-8 -*-
"""翼身融合（BWB）飞机参数化几何建模 —— CST 方法 + 三次 Hermite 曲线 + CATIA 二次开发。

实现依据
--------
戴浩, 余雄庆. 翼身融合飞机参数化几何模型[J]. 飞机设计, 2012, 32(2): 11-14.
邓海强, 余雄庆, 尹海莲, 等. 翼身融合无人机参数化建模与气动特性分析[J].
    航空计算技术, 2016, 46(6): 51-55.

建模思路（对应论文第 2 节）
--------------------------
翼身融合飞机外形分解为 翼身融合段 / 机翼 / 翼梢小翼 三部分，参数分四类：

    (1) 平面参数        —— 展长、弦长、后掠角、上反角、扭转角等；
    (2) 翼型参数        —— 各剖面翼型用 CST 方法（分类函数 + 形状函数）描述；
    (3) 融合段展向厚度  —— 用三次 Hermite 曲线描述，起点取根部最大厚度处；
    (4) 翼梢小翼参数    —— 高度、后掠角、尖削比、安装角、倾斜角、过渡段高度。

复杂的前缘/后缘形状用分段三次 Hermite 曲线描述（保持连接处一阶连续），
最后通过三维矩阵变换把各剖面点变换到机体坐标并放样成三维外形。

模块划分
--------
* ``cst_curve.py``     —— CST 分类函数 / 形状函数 / 翼型（可独立运行自检）
* ``hermite_curve.py`` —— 三次 Hermite 曲线与分段 C1 曲线（可独立运行自检）
* ``params_file.py``   —— 参数文件读写（TOML 格式，便于手工编辑）
* ``bwb_params.toml``  —— 参数文件本体：按平面/翼型/展向厚度/小翼/离散分组
* ``main.py``          —— 参数加载、剖面生成、CATIA 建模流程

运行示例
--------
    python main.py                        # 读取 bwb_params.toml（缺失则自动生成）
    python main.py --params my.toml       # 指定参数文件
    python main.py --model B              # 用论文表 1 的模型 B 预设
    python main.py --no-catia             # 只做数值计算并导出 CSV，不连接 CATIA
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import cast

import numpy as np

# 允许直接在本目录下运行（无需安装为包）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cst_curve import (  # noqa: E402
    CSTAirfoil,
    cosine_spacing,
    cst_weights_from_naca,
    make_cst_section,
    make_cst_section_from_weights,
)
from hermite_curve import PiecewiseHermite, tangents_from_points  # noqa: E402
from params_file import (  # noqa: E402
    ensure_param_file,
    flatten_param_dict,
    load_param_dict,
    save_param_file,
)
from pycatia.scripts.vba import vba_nothing  # noqa: E402


# =========================================================================== #
# 1. 三维矩阵变换（论文 1.3 节）
# =========================================================================== #
def rotation_about_axis(axis, angle_deg: float) -> np.ndarray:
    """罗德里格公式：绕任意轴旋转 angle_deg 度的 3x3 旋转矩阵（论文 1.3 节矩阵变换）。"""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    a = np.radians(angle_deg)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) * np.cos(a) + kx * np.sin(a) + np.outer(k, k) * (1.0 - np.cos(a))


def section_frame(e_span, incidence_deg: float = 0.0):
    """由展向轴确定剖面的局部坐标系 (弦向单位矢量, 厚度单位矢量)。

    剖面必须垂直于展向轴，因此厚度方向取 ``弦向 × 展向``；
    这样机翼上反、翼梢小翼倾斜时剖面不会被压扁或扭曲。
    最后绕展向轴施加安装角（正值为抬头）。
    """
    e_span = np.asarray(e_span, dtype=float)
    e_span = e_span / np.linalg.norm(e_span)

    e_chord = np.array([1.0, 0.0, 0.0])
    e_thick = np.cross(e_chord, e_span)
    e_thick = e_thick / np.linalg.norm(e_thick)
    e_chord = np.cross(e_span, e_thick)  # 正交化，保证严格直角

    rot = rotation_about_axis(e_span, incidence_deg)
    return rot @ e_chord, rot @ e_thick


# =========================================================================== #
# 2. 参数定义
# =========================================================================== #
@dataclass
class BWBParameters:
    """翼身融合飞机总体参数（默认值取自论文表 1 的模型 A，单位 mm / 度）。"""

    name: str = "ModelA"

    # ---- (1) 平面参数 --------------------------------------------------- #
    root_chord: float = 4500.0      # 根部弦长
    mid_chord: float = 2500.0       # 中部（内外翼交界）弦长
    tip_chord: float = 500.0        # 翼尖弦长
    inner_span: float = 1150.0      # 内段展长（半模）
    semi_span: float = 4000.0       # 半展长
    inner_sweep: float = 60.0       # 内段前缘后掠角
    outer_sweep: float = 35.0       # 外段前缘后掠角
    dihedral: float = 0.0           # 上反角
    incidence_root: float = 0.0     # 根部安装角
    incidence_tip: float = -2.0     # 翼尖安装角（扭转）

    # ---- (2) 翼型参数（CST） -------------------------------------------- #
    root_thickness: float = 0.20    # 根部相对厚度（融合体厚）
    mid_thickness: float = 0.14     # 中部相对厚度
    tip_thickness: float = 0.10     # 翼尖相对厚度
    camber_ratio: float = 0.02      # 弯度比
    camber_pos: float = 0.40        # 最大弯度位置
    reflex: float = 1.0             # 反弯系数（1 = 后缘弯度斜率归零，利于配平）
    cst_order: int = 8              # 伯恩斯坦多项式阶数（论文取 8）
    cst_n1: float = 0.5             # 分类函数 N1（圆头前缘）
    cst_n2: float = 1.0             # 分类函数 N2（尖形后缘）
    te_thickness: float = 0.0025    # 后缘相对厚度

    # ---- (3) 融合段展向厚度变化控制参数（三次 Hermite 切矢量） ---------- #
    thickness_slope_root: float = 0.0        # 根部 d(t/c)/dy
    thickness_slope_tip: float = -0.00002    # 翼尖 d(t/c)/dy

    # ---- (4) 翼梢小翼参数 ---------------------------------------------- #
    winglet_height: float = 540.0     # 小翼高度
    winglet_taper: float = 0.4        # 小翼尖削比
    winglet_sweep: float = 55.0       # 小翼后掠角
    winglet_cant: float = 45.0        # 小翼倾斜角（相对铅垂线）
    winglet_incidence: float = 5.0    # 小翼安装角
    winglet_transition: float = 0.30  # 过渡段占小翼高度的比例

    # ---- (5) CST 形状函数参数 A_i（翼型形状的设计变量） ------------------ #
    cst_shape_mode: str = "naca"        # "naca" = 由 [airfoil] 生成；"weights" = 直接用 A_i
    cst_shape_scope: str = "same"       # "same" = 各剖面共用；"per_station" = 逐剖面
    cst_thickness_weights: list = field(default_factory=list)   # 厚度形状参数 A_t
    cst_camber_weights: list = field(default_factory=list)      # 弯度形状参数 A_c
    cst_section_weights: dict = field(default_factory=dict)     # {剖面名: {权重}}

    # ---- (6) 三次 Hermite 曲线控制参数（平面外形/展向分布的设计变量） ----- #
    # 机头段前缘（俯视轮廓）：[[展向 y, 前缘 x], ...]，y 从 0（机头顶点）向外交接
    hermite_nose_points: list = field(default_factory=list)
    hermite_nose_slopes: list = field(default_factory=list)      # 各结点 dx/dy
    hermite_le_points: list = field(default_factory=list)        # [[y, x_le], ...]
    hermite_le_slopes: list = field(default_factory=list)        # 各结点 dx/dy
    hermite_te_points: list = field(default_factory=list)        # [[y, x_te], ...]
    hermite_te_slopes: list = field(default_factory=list)        # 各结点 dx/dy
    hermite_thickness_points: list = field(default_factory=list)  # [[y, t/c], ...]
    hermite_thickness_slopes: list = field(default_factory=list)  # 各结点 d(t/c)/dy

    # ---- 离散化控制 ----------------------------------------------------- #
    n_section_pts: int = 81          # 每个剖面总点数（上下表面闭合一圈）
    n_le_pts: int = 61               # 前缘/后缘引导线采样点数（导出与检查用）
    n_sections_span: int = 9         # 展向厚度插值采样点数（打印用）
    n_span_stations: int = 9         # 展向剖面数（含融合段）；越大越能体现展向厚度变化

    def to_model_b(self) -> "BWBParameters":
        """切换为论文表 1 中的模型 B 参数。"""
        self.name = "ModelB"
        self.root_chord = 5000.0
        self.mid_chord = 2500.0
        self.tip_chord = 500.0
        self.inner_span = 1200.0
        self.semi_span = 4000.0
        self.inner_sweep = 60.0
        self.outer_sweep = 35.0
        self.winglet_height = 500.0
        self.winglet_taper = 0.3
        self.winglet_sweep = 60.0
        self.winglet_cant = 30.0
        self.winglet_incidence = 2.0
        return self

    # ---- 参数文件读写 ---------------------------------------------------- #
    @classmethod
    def from_dict(cls, data: dict, nested: dict | None = None) -> "BWBParameters":
        """由参数文件构造参数对象。

        ``data``   ：扁平标量参数（``flatten_param_dict`` 的第 1 个返回值）
        ``nested`` ：结构化分组（``cst_shape`` / ``hermite`` 等）

        先应用 ``model`` 预设（A / B），再用文件中的显式取值覆盖；
        未识别的键会被忽略并提示，避免手工编辑参数文件时拼错字段名直接报错。
        """
        data = dict(data)
        known = {f.name for f in fields(cls)}
        params = cls()

        if str(data.pop("model", "")).upper() == "B":
            params.to_model_b()

        unknown = []
        for key, value in data.items():
            if key in known:
                setattr(params, key, value)
            else:
                unknown.append(key)
        if unknown:
            print(f"[提示] 参数文件中未识别的键已忽略：{', '.join(sorted(unknown))}")

        params._apply_nested(nested or {})
        return params

    def _apply_nested(self, nested: dict) -> None:
        """解析 ``[cst_shape]`` 与 ``[hermite.*]`` 等结构化分组。"""
        cst = nested.get("cst_shape") or {}
        if cst:
            self.cst_shape_mode = str(cst.get("mode", self.cst_shape_mode)).strip().lower()
            self.cst_shape_scope = str(cst.get("shape_mode", self.cst_shape_scope)).strip().lower()
            self.cst_thickness_weights = [float(v) for v in cst.get("thickness_weights", [])]
            self.cst_camber_weights = [float(v) for v in cst.get("camber_weights", [])]
            self.cst_section_weights = {
                str(name): {
                    str(kk): [float(x) for x in vv]
                    for kk, vv in content.items()
                    if isinstance(vv, (list, tuple))
                }
                for name, content in (cst.get("sections") or {}).items()
            }

        herm = nested.get("hermite") or {}

        def _points(group, key):
            content = herm.get(group) or {}
            return [list(map(float, p)) for p in content.get(key, [])]

        def _slopes(group, key):
            content = herm.get(group) or {}
            return [float(s) for s in content.get(key, [])]

        self.hermite_nose_points = _points("nose", "points")
        self.hermite_nose_slopes = _slopes("nose", "slopes")
        self.hermite_le_points = _points("leading_edge", "points")
        self.hermite_le_slopes = _slopes("leading_edge", "slopes")
        self.hermite_te_points = _points("trailing_edge", "points")
        self.hermite_te_slopes = _slopes("trailing_edge", "slopes")
        self.hermite_thickness_points = _points("span_thickness", "points")
        self.hermite_thickness_slopes = _slopes("span_thickness", "slopes")

    def to_dict(self) -> dict:
        """导出为扁平字典（仅标量字段，便于查看/比较）。"""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not isinstance(getattr(self, f.name), (list, dict))
        }

    def to_param_dict(self) -> dict:
        """导出为**参数文件的嵌套结构**，可直接交给 ``save_param_file`` 回写。

        气动优化迭代时可用：读参数 → 改权重 → 回写 → 重新建模。
        """
        return {
            "general": {"model": "A", "name": self.name},
            "planform": {
                "root_chord": self.root_chord,
                "mid_chord": self.mid_chord,
                "tip_chord": self.tip_chord,
                "inner_span": self.inner_span,
                "semi_span": self.semi_span,
                "inner_sweep": self.inner_sweep,
                "outer_sweep": self.outer_sweep,
                "dihedral": self.dihedral,
                "incidence_root": self.incidence_root,
                "incidence_tip": self.incidence_tip,
            },
            "airfoil": {
                "root_thickness": self.root_thickness,
                "mid_thickness": self.mid_thickness,
                "tip_thickness": self.tip_thickness,
                "camber_ratio": self.camber_ratio,
                "camber_pos": self.camber_pos,
                "reflex": self.reflex,
                "cst_order": self.cst_order,
                "cst_n1": self.cst_n1,
                "cst_n2": self.cst_n2,
                "te_thickness": self.te_thickness,
            },
            "span_thickness": {
                "thickness_slope_root": self.thickness_slope_root,
                "thickness_slope_tip": self.thickness_slope_tip,
            },
            "winglet": {
                "winglet_height": self.winglet_height,
                "winglet_taper": self.winglet_taper,
                "winglet_sweep": self.winglet_sweep,
                "winglet_cant": self.winglet_cant,
                "winglet_incidence": self.winglet_incidence,
                "winglet_transition": self.winglet_transition,
            },
            "cst_shape": {
                "mode": self.cst_shape_mode,
                "shape_mode": self.cst_shape_scope,
                "thickness_weights": list(self.cst_thickness_weights),
                "camber_weights": list(self.cst_camber_weights),
                "sections": {k: dict(v) for k, v in self.cst_section_weights.items()},
            },
            "hermite": {
                "nose": {
                    "points": [list(p) for p in self.hermite_nose_points],
                    "slopes": list(self.hermite_nose_slopes),
                },
                "leading_edge": {
                    "points": [list(p) for p in self.hermite_le_points],
                    "slopes": list(self.hermite_le_slopes),
                },
                "trailing_edge": {
                    "points": [list(p) for p in self.hermite_te_points],
                    "slopes": list(self.hermite_te_slopes),
                },
                "span_thickness": {
                    "points": [list(p) for p in self.hermite_thickness_points],
                    "slopes": list(self.hermite_thickness_slopes),
                },
            },
            "discretization": {
                "n_section_pts": self.n_section_pts,
                "n_le_pts": self.n_le_pts,
                "n_sections_span": self.n_sections_span,
                "n_span_stations": self.n_span_stations,
            },
        }


@dataclass
class Station:
    """一个展向控制剖面（携带自身局部坐标系）。"""

    name: str
    span: float            # 展向位置 y (mm)
    chord: float           # 弦长 (mm)
    x_le: float            # 前缘 x 坐标 (mm)
    z_ref: float           # 参考高度（上反 / 小翼抬升）
    thickness_ratio: float
    incidence: float = 0.0
    te_half: float = 0.0   # 后缘相对厚度的一半（相对弦长）：上表面 +te/2、下表面 -te/2
    e_span: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    e_chord: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    e_thick: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    airfoil: CSTAirfoil | None = None
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def le_point(self) -> np.ndarray:
        """剖面最前缘点（psi = 0，zeta = 0），即局部原点。"""
        return self.origin.copy()

    @property
    def te_point(self) -> np.ndarray:
        """后缘**中弧线**上的点（上下表面后缘的中间）。"""
        return self.origin + self.e_chord * self.chord

    @property
    def te_point_upper(self) -> np.ndarray:
        """后缘上表面点（psi = 1，zeta = +te/2）——闭合剖面的一个角点。"""
        return self.te_point + self.e_thick * (self.te_half * self.chord)

    @property
    def te_point_lower(self) -> np.ndarray:
        """后缘下表面点（psi = 1，zeta = -te/2）——闭合剖面的另一个角点。"""
        return self.te_point - self.e_thick * (self.te_half * self.chord)


# =========================================================================== #
# 3. 平面外形：分段三次 Hermite 前缘 / 后缘
# =========================================================================== #
def hermite_from_slopes(nodes, slopes=None, method: str = "catmull-rom") -> PiecewiseHermite:
    """由结点 ``[[x, y], ...]`` 与各结点斜率 dy/dx 构造分段三次 Hermite。

    给出 ``slopes`` 时切矢量取 ``(Δx, Δx*slope)``（Δx 为相邻结点间距），
    因此 slope 的物理含义严格等于 dy/dx，与结点疏密无关——这正是论文中
    "改变连接处切矢量的大小即可控制曲线形状"的做法。
    ``slopes`` 留空时用 Catmull-Rom 自动估计切矢量。
    """
    pts = np.atleast_2d(np.asarray(nodes, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("结点必须为 [[x, y], ...] 形式")
    if pts.shape[0] < 2:
        raise ValueError("至少需要 2 个结点")
    if slopes is None or len(slopes) == 0:
        return PiecewiseHermite.from_points(pts, method=method)

    slopes = np.asarray(slopes, dtype=float).ravel()
    if slopes.size != pts.shape[0]:
        raise ValueError(f"斜率个数({slopes.size})必须等于结点个数({pts.shape[0]})")

    dx = np.diff(pts[:, 0])
    span = np.empty(pts.shape[0])
    span[0], span[-1] = dx[0], dx[-1]
    if pts.shape[0] > 2:
        span[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return PiecewiseHermite(pts, np.column_stack([span, span * slopes]))


def interpolate_piecewise(curve: PiecewiseHermite, x: float) -> float:
    """在分段 Hermite 曲线上按第 1 个坐标插值第 2 个坐标。"""
    xs = curve.points[:, 0]
    for i in range(curve.num_segments):
        if x <= xs[i + 1] or i == curve.num_segments - 1:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return float(curve.segments[i].evaluate(t)[1])
    return float(curve.points[-1, 1])


def _key_planform_points(params: BWBParameters) -> np.ndarray:
    """三个关键平面点：根部、内外翼交界、翼尖 (展向 y, 前缘 x)。"""
    y_in = params.inner_span
    y_tip = params.semi_span
    x_in = y_in * np.tan(np.radians(params.inner_sweep))
    x_tip = x_in + (y_tip - y_in) * np.tan(np.radians(params.outer_sweep))
    return np.array([(0.0, 0.0), (y_in, x_in), (y_tip, x_tip)], dtype=float)


def leading_edge_nodes(params: BWBParameters):
    """合并**机头段**与**机翼段**的前缘结点，返回 ``(nodes, slopes)``。

    结点来源与优先级：
        1. ``[hermite.nose]``         —— 机头段俯视轮廓（y 从 0 的机头顶点向外交接）；
        2. ``[hermite.leading_edge]`` —— 机翼段前缘结点；
        3. 若上述结点未覆盖到半展长，自动补上由后掠角算得的关键点。

    结点按展向递增排列、重复站位自动跳过。只有**所有**结点都给出了斜率时才采用显式
    切矢量，否则整条曲线统一用 Catmull-Rom 自动估计（避免半显式半自动造成折点）。
    """
    merged: list[list[float]] = []
    slopes: list[float] = []
    complete = True

    for pts, slp in (
        (params.hermite_nose_points, params.hermite_nose_slopes),
        (params.hermite_le_points, params.hermite_le_slopes),
    ):
        pts = list(pts or [])
        for k, p in enumerate(pts):
            y = float(p[0])
            if merged and y <= merged[-1][0] + 1e-9:
                continue
            merged.append([y, float(p[1])])
            if slp and len(slp) == len(pts):
                slopes.append(float(slp[k]))
            else:
                complete = False

    if not merged:                                    # 未给任何结点 -> 用后掠角默认
        slopes = [
            np.tan(np.radians(params.inner_sweep)),
            np.tan(np.radians(params.outer_sweep)),
            np.tan(np.radians(params.outer_sweep)),
        ]
        return _key_planform_points(params), slopes

    if merged[-1][0] < params.semi_span - 1e-9:       # 未覆盖到翼尖 -> 补默认关键点
        default_slopes = [
            np.tan(np.radians(params.inner_sweep)),
            np.tan(np.radians(params.outer_sweep)),
            np.tan(np.radians(params.outer_sweep)),
        ]
        for j, (y, x) in enumerate(_key_planform_points(params)):
            if y > merged[-1][0] + 1e-9:
                merged.append([float(y), float(x)])
                # 补点也补上与之对应的默认斜率，避免用户已给的斜率被整体丢弃
                if complete:
                    slopes.append(float(default_slopes[j]))

    if not (complete and len(slopes) == len(merged)):
        slopes = []
    return np.asarray(merged, dtype=float), slopes


def leading_edge_curve(params: BWBParameters) -> PiecewiseHermite:
    """前缘曲线（含机头段，分段三次 Hermite）。

    前缘不是"把各站位端点用样条连起来"，而是由分段三次 Hermite 描述（论文 1.2 节）：
    机头段与机翼段各有自己的结点与切矢量（dx/dy = tan 后掠角），
    交界处共享切矢量，因此天然一阶连续，机头形状也能独立控制。
    """
    nodes, slopes = leading_edge_nodes(params)
    return hermite_from_slopes(nodes, slopes)


def leading_edge_x(params: BWBParameters, y: float) -> float:
    """由分段 Hermite 前缘曲线插值任意展向位置处的前缘 x 坐标。"""
    return interpolate_piecewise(leading_edge_curve(params), y)


def _te_curve(params: BWBParameters):
    """``[hermite.trailing_edge]`` 给定且覆盖 0~半展长时返回后缘 Hermite 曲线。"""
    pts = params.hermite_te_points
    if not pts:
        return None
    nodes = np.asarray(pts, dtype=float)
    if nodes[0, 0] > 1e-9 or nodes[-1, 0] < params.semi_span - 1e-9:
        print("[提示] [hermite.trailing_edge] 的结点未覆盖 0~半展长，已忽略该分组")
        return None
    return hermite_from_slopes(nodes, params.hermite_te_slopes)


def trailing_edge_curve(params: BWBParameters, num: int = 61) -> np.ndarray:
    """后缘曲线（3D 点列）。

    论文中用三次 Hermite 描述后缘曲线形状：若 ``[hermite.trailing_edge]`` 给了
    结点则以该曲线为准，否则由前缘曲线 + 展向弦长分布得到。
    """
    te = _te_curve(params)
    tan_dih = np.tan(np.radians(params.dihedral))
    ys = np.linspace(0.0, params.semi_span, num)
    pts = []
    for y in ys:
        x_le = leading_edge_x(params, y)
        x_te = interpolate_piecewise(te, y) if te is not None else x_le + chord_at_span(params, y)
        pts.append((x_te, y, y * tan_dih))
    return np.asarray(pts, dtype=float)


def leading_edge_points_3d(params: BWBParameters, num: int = 61) -> np.ndarray:
    """前缘曲线（3D 点列），用于 CATIA 引导线与 CSV 导出。"""
    tan_dih = np.tan(np.radians(params.dihedral))
    ys = np.linspace(0.0, params.semi_span, num)
    return np.asarray([(leading_edge_x(params, y), y, y * tan_dih) for y in ys], dtype=float)


# =========================================================================== #
# 4. 弦长 / 厚度 / 安装角沿展向分布
# =========================================================================== #
def chord_at_span(params: BWBParameters, y: float) -> float:
    """展向弦长分布。

    默认按 根部→中部→翼尖 三点分段线性；
    若 ``[hermite.trailing_edge]`` 给出了后缘结点，则弦长 = 后缘 x − 前缘 x。
    """
    te = _te_curve(params)
    if te is not None:
        return interpolate_piecewise(te, y) - leading_edge_x(params, y)

    y_in, y_tip = params.inner_span, params.semi_span
    if y <= y_in:
        t = y / y_in
        return params.root_chord + t * (params.mid_chord - params.root_chord)
    t = (y - y_in) / (y_tip - y_in)
    return params.mid_chord + t * (params.tip_chord - params.mid_chord)


def thickness_curve(params: BWBParameters, num: int = 61) -> tuple[np.ndarray, np.ndarray]:
    """融合段展向相对厚度变化曲线（分段三次 Hermite，论文 2(3) 节）。

    默认：取 根部 / 中部 / 翼尖 三个结点，切矢量由 ``[span_thickness]`` 的
    端点斜率与中点中心差分给出，起点取在根部最大厚度处。
    覆盖：若 ``[hermite.span_thickness]`` 给出了 ``points``，则以它为结点，
    可给多个结点做更精细的展向厚度控制。
    """
    if params.hermite_thickness_points:
        nodes = np.asarray(params.hermite_thickness_points, dtype=float)
        if nodes[0, 0] > 1e-9 or nodes[-1, 0] < params.semi_span - 1e-9:
            print("[提示] [hermite.span_thickness] 的结点未覆盖 0~半展长，已忽略该分组")
            nodes = None
    else:
        nodes = None

    if nodes is not None:
        curve = hermite_from_slopes(nodes, params.hermite_thickness_slopes)
    else:
        nodes = np.column_stack([
            [0.0, params.inner_span, params.semi_span],
            [params.root_thickness, params.mid_thickness, params.tip_thickness],
        ])
        slope_mid = 0.5 * (
            (params.mid_thickness - params.root_thickness) / params.inner_span
            + (params.tip_thickness - params.mid_thickness) / (params.semi_span - params.inner_span)
        )
        curve = hermite_from_slopes(
            nodes, [params.thickness_slope_root, slope_mid, params.thickness_slope_tip]
        )

    per_seg = max(num // max(curve.num_segments, 1), 2)
    pts = curve.sample(per_seg)
    order = np.argsort(pts[:, 0])
    return pts[order, 0], pts[order, 1]


def thickness_at_span(params: BWBParameters, y: float) -> float:
    """融合段展向厚度变化：三次 Hermite 曲线插值。"""
    y_arr, tc_arr = thickness_curve(params)
    return float(np.interp(y, y_arr, tc_arr))


def incidence_at_span(params: BWBParameters, y: float) -> float:
    """安装角沿展向线性变化（根部→翼尖）。"""
    t = min(max(y / params.semi_span, 0.0), 1.0)
    return params.incidence_root + t * (params.incidence_tip - params.incidence_root)


# =========================================================================== #
# 5. 剖面与整机点云
# =========================================================================== #
def make_section_airfoil(params: BWBParameters, thickness_ratio: float, name: str) -> CSTAirfoil:
    """按给定相对厚度生成 CST 剖面翼型。

    两种模式（由参数文件 ``[cst_shape] mode`` 决定）：
        ``"naca"``    —— 由 ``[airfoil]`` 的厚度/弯度参数生成基准翼型；
        ``"weights"`` —— 直接使用 ``[cst_shape]`` 的厚度/弯度形状参数 A_i
                         （气动优化用），并支持 ``[cst_shape.sections]`` 逐剖面覆盖。
    """
    t_w = list(params.cst_thickness_weights)
    c_w = list(params.cst_camber_weights)

    if params.cst_shape_mode.strip().lower() == "weights" and t_w and c_w:
        if params.cst_shape_scope.strip().lower() == "per_station":
            override = params.cst_section_weights.get(name) or {}
            t_w = list(override.get("thickness_weights", t_w))
            c_w = list(override.get("camber_weights", c_w))

        if len(t_w) != len(c_w):
            raise ValueError(
                f"剖面 {name}: 厚度权重({len(t_w)} 个)与弯度权重({len(c_w)} 个)个数不一致"
            )
        if len(t_w) != params.cst_order + 1:
            print(f"[提示] 剖面 {name}: 权重个数 {len(t_w)} 与 cst_order+1="
                  f"{params.cst_order + 1} 不一致，将按 {len(t_w) - 1} 阶伯恩斯坦多项式计算")

        return make_cst_section_from_weights(
            t_w, c_w,
            thickness_ratio=thickness_ratio,
            camber_ratio=params.camber_ratio,
            n1=params.cst_n1,
            n2=params.cst_n2,
            te_thickness=params.te_thickness,
            name=name,
        )

    return make_cst_section(
        thickness_ratio=thickness_ratio,
        camber_ratio=params.camber_ratio,
        camber_pos=params.camber_pos,
        reflex=params.reflex,
        order=params.cst_order,
        n1=params.cst_n1,
        n2=params.cst_n2,
        te_thickness=params.te_thickness,
        num=201,
        name=name,
    )


def span_station_positions(params: BWBParameters) -> list[tuple[str, float]]:
    """给出展向剖面站位（含融合段），使展向厚度变化真正体现在几何中。

    关键站位（root / inner / tip）保留名字，便于 ``[cst_shape.sections]`` 逐剖面覆盖；
    其余站位按**余弦分布**加密（根部与翼尖附近更密），数目由 ``n_span_stations`` 控制。
    只有把站位铺满展向，Hermite 厚度曲线才会被真实地"采样"成一系列剖面，
    否则放样只会在 3 个剖面之间插值，展向厚度变化无从体现。
    """
    n = max(int(params.n_span_stations), 3)
    key = {"root": 0.0, "inner": float(params.inner_span), "tip": float(params.semi_span)}

    s = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    ys = (params.semi_span * s).tolist()

    tol = 0.05 * params.semi_span
    for ky in key.values():                     # 关键站位替换最近的分布点
        if any(abs(y - ky) < 1e-6 for y in ys):
            continue
        near = [i for i, y in enumerate(ys) if abs(y - ky) < tol]
        if near:
            ys[near[0]] = ky

    ys = sorted({round(float(y), 6) for y in ys})
    used: set[str] = set()
    named: list[tuple[str, float]] = []
    for y in ys:
        name = ""
        for k, ky in key.items():
            if k not in used and abs(y - ky) < 1e-6:
                name, used = k, used | {k}
                break
        named.append((name, y))

    result: list[tuple[str, float]] = []
    idx = 0
    for name, y in named:
        if not name:
            name = f"sec{idx:02d}"
            idx += 1
        result.append((name, y))
    return result


def station_at(params: BWBParameters, y: float) -> Station:
    """按展向位置 y 生成"虚拟剖面"的几何信息（不含翼型）。

    引导线必须与剖面使用**同一套**几何定义，否则会出现剖面端点不与引导线相交的问题：
    例如剖面有安装角（扭转）时，后缘点会偏离"前缘 + 弦长"的平面外形曲线
    （翼尖 -2° 扭转即可造成十几毫米的偏差）。因此引导线与站位都走这个函数。
    """
    dihedral = np.radians(params.dihedral)
    wing_axis = np.array([0.0, np.cos(dihedral), np.sin(dihedral)])
    x_le = leading_edge_x(params, y)
    z_ref = float(y) * np.tan(dihedral)
    incidence = incidence_at_span(params, y)
    e_chord, e_thick = section_frame(wing_axis, incidence)
    return Station(
        name=f"y{y:g}",
        span=float(y),
        chord=chord_at_span(params, y),
        x_le=x_le,
        z_ref=z_ref,
        thickness_ratio=thickness_at_span(params, y),
        incidence=incidence,
        te_half=0.5 * float(params.te_thickness),
        e_span=wing_axis,
        e_chord=e_chord,
        e_thick=e_thick,
        airfoil=None,
        origin=np.array([x_le, float(y), z_ref]),
    )


def build_stations(params: BWBParameters) -> list[Station]:
    """建立展向控制剖面：融合段/机翼多个剖面 + 翼梢小翼 2 个（过渡段 + 主段）。

    展向站位由 ``span_station_positions`` 给出（数目 = ``n_span_stations``），
    每个剖面的相对厚度来自 Hermite 展向厚度曲线，因此融合段的展向厚度变化
    会真实地体现在一系列剖面上；机头（根部）剖面可由 ``[hermite.nose]`` 控制。
    """
    stations: list[Station] = []

    # ---- 融合段 / 机翼剖面：与引导线共用 station_at 的几何定义 ------------- #
    for name, y in span_station_positions(params):
        st = station_at(params, y)
        st.name = name
        st.airfoil = make_section_airfoil(params, st.thickness_ratio, name)
        stations.append(st)

    # ---- 翼梢小翼剖面：展向轴沿小翼倾斜方向 ------------------------------ #
    h = params.winglet_height
    cant = np.radians(params.winglet_cant)
    sweep = np.radians(params.winglet_sweep)
    winglet_axis = np.array([0.0, np.sin(cant), np.cos(cant)])
    tip_station = stations[-1]

    for frac, label in ((params.winglet_transition, "winglet_trans"), (1.0, "winglet_tip")):
        # 沿小翼展向轴移动 + 沿弦向的后掠偏移
        origin = (
            tip_station.origin
            + h * frac * winglet_axis
            + np.array([h * frac * np.tan(sweep), 0.0, 0.0])
        )
        chord = tip_station.chord * (1.0 + frac * (params.winglet_taper - 1.0))
        thickness = params.tip_thickness * (1.0 - 0.30 * frac)
        incidence = params.winglet_incidence
        e_chord, e_thick = section_frame(winglet_axis, incidence)
        stations.append(
            Station(
                name=label,
                span=float(origin[1]),
                chord=chord,
                x_le=float(origin[0]),
                z_ref=float(origin[2]),
                thickness_ratio=thickness,
                incidence=incidence,
                te_half=0.5 * float(params.te_thickness),
                e_span=winglet_axis,
                e_chord=e_chord,
                e_thick=e_thick,
                airfoil=make_section_airfoil(params, thickness, label),
                origin=origin,
            )
        )
    return stations


def _to_world(station: Station, psi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """把归一化 (弦向, 厚度) 坐标映射到机体坐标系。"""
    x = np.asarray(psi, dtype=float) * station.chord
    z = np.asarray(zeta, dtype=float) * station.chord
    return (
        station.origin.reshape(1, 3)
        + np.outer(x, station.e_chord)
        + np.outer(z, station.e_thick)
    )


def section_surface_curves(station: Station, n_surface: int = 41):
    """返回剖面上、下表面点列（3D）。

    上表面：后缘 → 前缘；下表面：前缘 → 后缘。
    两者**都包含前缘点 (psi = 0)**，因此 ``upper[-1] == lower[0] == station.origin``，
    上下表面在前缘严格重合，才能生成光滑的圆头（否则前缘会错开一段，曲线严重失真）。
    """
    foil = station.airfoil
    if foil is None:
        raise ValueError(f"剖面 {station.name} 未定义翼型")

    psi = cosine_spacing(n_surface)
    _, zu = foil.upper_ordinates(n_surface)
    _, zl = foil.lower_ordinates(n_surface)

    upper = _to_world(station, psi[::-1], zu[::-1])   # TE -> LE
    lower = _to_world(station, psi, zl)               # LE -> TE
    return upper, lower


def section_coordinates(station: Station, num: int = 81) -> np.ndarray:
    """闭合剖面点列（上表面 TE→LE，再下表面 LE→TE，前缘点只保留一次）。"""
    n_surface = max(num // 2 + 1, 6)
    upper, lower = section_surface_curves(station, n_surface)
    return np.vstack([upper, lower[1:]])


def build_all_sections(params: BWBParameters) -> list[tuple[str, np.ndarray]]:
    """返回全部（名称，点列）剖面，供 CSV 导出使用。"""
    result = []
    for st in build_stations(params):
        result.append((f"section_{st.name}", section_coordinates(st, num=params.n_section_pts)))
    return result


# =========================================================================== #
# 6. 导出与打印（无 CATIA 时的数值验证）
# =========================================================================== #
def export_csv(params: BWBParameters, out_dir: str | Path = "output") -> list[Path]:
    """把各剖面与前缘/后缘曲线导出为 CSV，便于脱离 CATIA 校验几何。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, coords in build_all_sections(params):
        path = out_dir / f"{params.name}_{name}.csv"
        np.savetxt(path, coords, delimiter=",", header="x_mm,y_mm,z_mm", comments="")
        written.append(path)

    path = out_dir / f"{params.name}_leading_edge.csv"
    np.savetxt(path, leading_edge_points_3d(params, params.n_le_pts), delimiter=",",
               header="x_mm,y_mm,z_mm", comments="")
    written.append(path)

    path = out_dir / f"{params.name}_trailing_edge.csv"
    np.savetxt(path, trailing_edge_curve(params, params.n_le_pts), delimiter=",",
               header="x_mm,y_mm,z_mm", comments="")
    written.append(path)
    return written


def print_controls(params: BWBParameters) -> None:
    """打印当前生效的 CST / Hermite 控制参数，便于确认气动优化的设计变量。"""
    mode = params.cst_shape_mode.strip().lower()
    print("-" * 72)
    print("曲线控制参数（气动优化的设计变量）")
    print(f"  CST  模式={mode}  形状范围={params.cst_shape_scope}  阶数={params.cst_order}  "
          f"N1={params.cst_n1}  N2={params.cst_n2}")
    if mode == "weights":
        print(f"       厚度权重 A_t({len(params.cst_thickness_weights)}) = "
              + np.array2string(np.asarray(params.cst_thickness_weights, dtype=float), precision=5))
        print(f"       弯度权重 A_c({len(params.cst_camber_weights)}) = "
              + np.array2string(np.asarray(params.cst_camber_weights, dtype=float), precision=5))
        if params.cst_section_weights:
            print(f"       逐剖面覆盖：{sorted(params.cst_section_weights)}")

    le = leading_edge_curve(params)
    if params.hermite_nose_points and params.hermite_le_points:
        src_le = "（机头段 + 机翼段，来自参数文件）"
    elif params.hermite_nose_points:
        src_le = "（机头段来自参数文件，其余按后掠角生成）"
    elif params.hermite_le_points:
        src_le = "（机翼段来自参数文件，其余按后掠角生成）"
    else:
        src_le = "（按后掠角生成）"
    print(f"  Hermite 前缘    : {le.num_segments} 段，结点 y={np.round(le.points[:, 0], 1).tolist()} "
          f"x_le={np.round(le.points[:, 1], 1).tolist()}{src_le}")
    print(f"                    切矢量 dx/dy="
          f"{np.round(le.tangents[:, 1] / le.tangents[:, 0], 4).tolist()}")
    if params.hermite_nose_points:
        print(f"                    其中机头段结点 y={np.round(np.asarray(params.hermite_nose_points)[:, 0], 1).tolist()}"
              f"（机头俯视轮廓独立控制）")

    te = _te_curve(params)
    if te is not None:
        print(f"  Hermite 后缘    : {te.num_segments} 段，结点 y={np.round(te.points[:, 0], 1).tolist()} "
              f"x_te={np.round(te.points[:, 1], 1).tolist()}（来自参数文件，弦长由它反推）")
    else:
        print("  Hermite 后缘    : 按 前缘 + 弦长分布 生成")

    if params.hermite_thickness_points:
        nodes = np.asarray(params.hermite_thickness_points, dtype=float)
        print(f"  Hermite 展向厚度: 结点 y={np.round(nodes[:, 0], 1).tolist()} "
              f"t/c={np.round(nodes[:, 1], 5).tolist()}（来自参数文件）")
    else:
        print(f"  Hermite 展向厚度: 结点 y=[0, {params.inner_span:g}, {params.semi_span:g}] "
              f"t/c=[{params.root_thickness:g}, {params.mid_thickness:g}, {params.tip_thickness:g}]"
              f"（根部/中部/翼尖）")
    print("-" * 72)


def print_summary(params: BWBParameters) -> None:
    print("=" * 72)
    print(f"翼身融合飞机参数化模型 —— {params.name}")
    print("=" * 72)
    print(f"{'剖面':<16}{'展向y/mm':>12}{'弦长/mm':>12}{'前缘x/mm':>12}{'t/c':>10}")
    for st in build_stations(params):
        print(f"{st.name:<16}{st.span:>12.1f}{st.chord:>12.1f}{st.x_le:>12.1f}"
              f"{st.thickness_ratio:>10.4f}")
    print_controls(params)
    print("=" * 72)


# =========================================================================== #
# 7. CATIA 二次开发
# =========================================================================== #
def connect_catia():
    """连接 CATIA 并新建 Part 文档，返回 (catia, part)。"""
    import pycatia
    from pycatia.mec_mod_interfaces.part_document import PartDocument

    catia = pycatia.catia()
    try:
        doc = catia.documents.add("Part")
    except Exception:  # 已有零件文档时回退到活动文档
        doc = catia.active_document
    part = cast(PartDocument, doc).part
    return catia, part


def add_points(hsf, coords: np.ndarray):
    """批量创建点，返回点对象列表。"""
    coord_list = [[float(a), float(b), float(c)] for a, b, c in np.atleast_2d(coords)]
    return hsf.add_new_point_coords(coord_list)


def add_spline(hsf, points, name: str):
    """由点对象列表创建样条（按给定顺序）。"""
    spline = hsf.add_new_spline()
    for pt in points:
        spline.add_point(pt)
    spline.name = name
    return spline


def _add_le_point_with_tangency(spline, point, direction, tension: float = 1.0, invert: int = 1) -> bool:
    """把点作为样条末控制点加入并施加切矢约束，使前缘光滑过渡。

    CATIA 不同版本的行为略有差异，若约束设置失败则退化为普通控制点，
    曲线仍可生成（只是前缘为 C0），避免整个流程中断。
    """
    try:
        spline.add_point_with_constraint_explicit(point, direction, tension, invert, vba_nothing, 0.0)
        return True
    except Exception as exc:
        print(f"[提示] 前缘切矢约束未生效（{exc}），该点退化为普通控制点")
        spline.add_point(point)
        return False


def add_airfoil_section(hsf, part, hb, station: Station, num: int, name: str):
    """生成闭合剖面曲线：上、下表面各一条样条 + 后缘直线 + 接合。

    两个关键点：
    1. 上、下表面**共享同一个前缘点对象**，前缘严格闭合，不会错开；
    2. 两条样条在前缘都施加沿厚度方向（垂直于弦）的切矢约束——
       圆头前缘在最前缘处的切线本来就是垂直弦向的，这样才能得到光滑圆头。
       若用一条样条绕过前缘（方向折返），会形成尖点并引起抖动。
    """
    n_surface = max(num // 2 + 1, 6)
    upper_xyz, lower_xyz = section_surface_curves(station, n_surface)

    # 前缘共享点（upper[-1] 与 lower[0] 坐标相同，这里只建一个点对象）
    le_point = add_points(hsf, upper_xyz[-1].reshape(1, 3))[0]
    upper_pts = add_points(hsf, upper_xyz[:-1]) + [le_point]
    lower_pts = [le_point] + add_points(hsf, lower_xyz[1:])

    # 前缘切矢方向 = -厚度方向（圆头前缘最前缘点的切线）
    le_dir = hsf.add_new_direction_by_coord(*(-float(v) for v in station.e_thick))
    le_dir.name = f"{name}_le_dir"
    hb.append_hybrid_shape(le_dir)

    # 上表面样条：前缘位于末端
    spline_up = hsf.add_new_spline()
    spline_up.name = f"{name}_upper"
    for pt in upper_pts[:-1]:
        spline_up.add_point(pt)
    _add_le_point_with_tangency(spline_up, upper_pts[-1], le_dir, 1.0, 1)

    # 下表面样条：前缘位于首端
    spline_lo = hsf.add_new_spline()
    spline_lo.name = f"{name}_lower"
    _add_le_point_with_tangency(spline_lo, lower_pts[0], le_dir, 1.0, 1)
    for pt in lower_pts[1:]:
        spline_lo.add_point(pt)

    # 后缘直线：保留尖后缘
    te_line = hsf.add_new_line_pt_pt(upper_pts[0], lower_pts[-1])
    te_line.name = f"{name}_te"

    for obj in (spline_up, spline_lo, te_line):
        hb.append_hybrid_shape(obj)

    # 接合为一条闭合剖面曲线
    join = hsf.add_new_join(
        part.create_reference_from_object(spline_up),
        part.create_reference_from_object(te_line),
    )
    join.add_element(part.create_reference_from_object(spline_lo))
    join.name = name
    hb.append_hybrid_shape(join)
    return join


def guide_points(params: BWBParameters, stations: list[Station]) -> dict:
    """生成引导线点列，**机翼段与翼梢小翼段分开**，后缘再分上/下两条。

    返回字典：
        wing_le / winglet_le
            —— 前缘：按 Hermite 曲线采样（含各站位 y），再取该处剖面的真实前缘点；
        wing_te_up / wing_te_lo / winglet_te_up / winglet_te_lo
            —— 后缘**上、下各一条**：由**同一组** ``[hermite.trailing_edge]`` 参数
               （后缘中弧线）确定弦长与后缘 x，再沿剖面厚度方向偏移 +te/2 与 -te/2
               得到两条穿过剖面实际角点的曲线，这样多截面曲面才能正常放样。

    翼梢小翼段以"翼尖剖面"为起点，与机翼段在翼尖处天然衔接。
    """
    n = max(int(params.n_le_pts), 20)
    ys = set(np.round(np.linspace(0.0, params.semi_span, n), 6).tolist())
    ys |= {round(float(st.span), 6) for st in stations if st.span <= params.semi_span + 1e-9}
    ys = sorted(y for y in ys if y <= params.semi_span + 1e-9)

    wing_le, wing_tu, wing_tl = [], [], []
    for y in ys:
        st = station_at(params, y)          # 与剖面同一套几何定义（含扭转）
        wing_le.append(tuple(float(v) for v in st.le_point))
        wing_tu.append(tuple(float(v) for v in st.te_point_upper))
        wing_tl.append(tuple(float(v) for v in st.te_point_lower))

    tip = [st for st in stations if abs(st.span - params.semi_span) < 1e-6]
    outboard = [st for st in stations if st.span > params.semi_span + 1e-9]
    chain = tip + outboard                  # 翼尖 -> 过渡段 -> 主段
    if len(chain) >= 2:
        m = max(n // 4, 8)
        winglet_le = PiecewiseHermite.from_points(
            np.asarray([st.le_point for st in chain], dtype=float)
        ).sample(m)
        # 后缘上/下两条：同一组参数（同一条中弧线 + 同一套剖面），仅偏移方向不同
        winglet_tu = PiecewiseHermite.from_points(
            np.asarray([st.te_point_upper for st in chain], dtype=float)
        ).sample(m)
        winglet_tl = PiecewiseHermite.from_points(
            np.asarray([st.te_point_lower for st in chain], dtype=float)
        ).sample(m)
    else:
        winglet_le = winglet_tu = winglet_tl = np.empty((0, 3))

    return {
        "wing_le": np.asarray(wing_le, dtype=float),
        "wing_te_up": np.asarray(wing_tu, dtype=float),
        "wing_te_lo": np.asarray(wing_tl, dtype=float),
        "winglet_le": winglet_le,
        "winglet_te_up": winglet_tu,
        "winglet_te_lo": winglet_tl,
    }


def add_guides(hsf, part, hb, params: BWBParameters, stations: list[Station]) -> dict:
    """创建引导线，机翼与翼梢小翼分开，共六条：前缘各一条、后缘各上/下两条。

    后缘之所以要两条：多截面曲面的引导线必须穿过剖面的实际角点，
    而闭合剖面的后缘是"上表面点 + 下表面点"两个角点，只给中弧线一条会无法正常放样。

    返回 ``{键: 样条对象}``。
    """
    pts = guide_points(params, stations)
    made: dict = {}
    for key, name in (
        ("wing_le", "leading_edge"),
        ("wing_te_up", "trailing_edge_upper"),
        ("wing_te_lo", "trailing_edge_lower"),
        ("winglet_le", "winglet_leading_edge"),
        ("winglet_te_up", "winglet_trailing_edge_upper"),
        ("winglet_te_lo", "winglet_trailing_edge_lower"),
    ):
        arr = pts[key]
        if arr.shape[0] < 2:
            continue
        spline = add_spline(hsf, add_points(hsf, arr), name)
        hb.append_hybrid_shape(spline)
        made[key] = spline

    if params.hermite_nose_points:          # 机头段轮廓参考曲线
        nose = np.asarray(
            [(float(p[1]), float(p[0]), 0.0) for p in params.hermite_nose_points], dtype=float
        )
        if nose.shape[0] >= 2:
            hb.append_hybrid_shape(add_spline(hsf, add_points(hsf, nose), "nose_outline"))
    return made


def create_loft(hsf, part, hb, sections: list, guides: tuple, name: str):
    """多截面曲面：按顺序填入各剖面并加入引导线。

    注意：CATIA 的自动化接口是 ``AddSectionToLoft``（pycatia 对应
    ``add_section_to_loft``），并不存在 ``add_section``，
    之前调用不存在的接口导致异常被吞掉、多截面曲面里没有任何元素。
    """
    loft = hsf.add_new_loft()
    for sec in sections:
        ref = part.create_reference_from_object(sec)
        # i_ori = 1 表示保持剖面方向；封闭剖面无需闭合点，传 Nothing
        loft.add_section_to_loft(ref, 1, vba_nothing)
    for guide in guides:
        loft.add_guide(part.create_reference_from_object(guide))

    loft.section_coupling = 1   # 按弧长比例耦合，最稳健
    loft.canonical_detection = 0
    loft.name = name
    hb.append_hybrid_shape(loft)
    return loft


def build_catia_model(catia, part, params: BWBParameters) -> None:
    """在 CATIA 中生成 BWB 三维外形：剖面 → 引导线 → 多截面曲面 → 镜像。

    机翼与翼梢小翼的引导线**分开创建**、放样也分成两个曲面，避免相互影响；
    两者在"翼尖剖面"处衔接，因此共用该剖面。
    """
    hsf = part.hybrid_shape_factory
    hb = part.hybrid_bodies.add()
    hb.name = f"BWB_{params.name}"

    stations = build_stations(params)

    # ---- 1) 各控制剖面（CST 翼型，上下表面分开做样条） ------------------- #
    pairs: list[tuple[Station, object]] = []
    for st in stations:
        curve = add_airfoil_section(hsf, part, hb, st, params.n_section_pts, f"section_{st.name}")
        pairs.append((st, curve))
    print(f"[1/4] 已创建 {len(pairs)} 个控制剖面（CST 上下表面样条）")
    part.update()

    # ---- 2) 引导线：机翼与小翼各两条，共四条独立曲线 --------------------- #
    made = add_guides(hsf, part, hb, params, stations)
    print(f"[2/4] 已创建引导线：{', '.join(sorted(made))}")
    part.update()

    # ---- 3) 多截面曲面：机翼与小翼分别放样 ------------------------------- #
    wing_pairs = [(st, c) for st, c in pairs if st.span <= params.semi_span + 1e-9]
    tip_pairs = [(st, c) for st, c in pairs if abs(st.span - params.semi_span) < 1e-6]
    winglet_pairs = [(st, c) for st, c in pairs if st.span > params.semi_span + 1e-9]

    lofts = []

    if {"wing_le", "wing_te_up", "wing_te_lo"} <= made.keys():
        loft_wing = create_loft(
            hsf, part, hb, [c for _, c in wing_pairs],
            (made["wing_le"], made["wing_te_up"], made["wing_te_lo"]),
            f"BWB_wing_{params.name}",
        )
        print(f"[3/4] 机翼多截面曲面：剖面 {len(wing_pairs)} 个，"
              f"引导线 {loft_wing.get_nb_of_guides()} 条")
        lofts.append(loft_wing)

    winglet_chain = tip_pairs + winglet_pairs          # 翼尖 → 过渡段 → 主段
    if len(winglet_chain) >= 2 and {"winglet_le", "winglet_te_up", "winglet_te_lo"} <= made.keys():
        loft_winglet = create_loft(
            hsf, part, hb, [c for _, c in winglet_chain],
            (made["winglet_le"], made["winglet_te_up"], made["winglet_te_lo"]),
            f"BWB_winglet_{params.name}",
        )
        print(f"      翼梢小翼多截面曲面：剖面 {len(winglet_chain)} 个，"
              f"引导线 {loft_winglet.get_nb_of_guides()} 条")
        lofts.append(loft_winglet)

    if not lofts:
        raise RuntimeError("未能创建任何多截面曲面，请检查剖面与引导线是否匹配")
    part.update()

    # ---- 4) 关于对称面镜像，得到全机外形 --------------------------------- #
    plane = part.origin_elements.plane_zx  # 机身对称面（x-z 平面）
    for i, loft in enumerate(lofts, start=1):
        sym = hsf.add_new_symmetry(part.create_reference_from_object(loft), plane)
        sym.name = f"BWB_mirror{i}_{params.name}"
        hb.append_hybrid_shape(sym)
    part.update()
    print(f"[4/4] 已关于机身对称面镜像 {len(lofts)} 个曲面")

    try:
        catia.refresh_display()
    except Exception:
        pass

    try:
        catia.refresh_display()
    except Exception:
        pass


# =========================================================================== #
# 8. 入口
# =========================================================================== #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="翼身融合飞机参数化几何建模（CST + 三次 Hermite + CATIA）"
    )
    parser.add_argument("--params", default="bwb_params.toml",
                        help="参数文件路径（TOML，默认 ./bwb_params.toml，缺失时自动生成）")
    parser.add_argument("--model", choices=["A", "B", "a", "b"], default=None,
                        help="覆盖参数文件中的 model 预设（A 或 B）")
    parser.add_argument("--no-catia", action="store_true",
                        help="仅做数值计算并导出 CSV，不连接 CATIA")
    parser.add_argument("--csv-dir", default="output",
                        help="CSV 导出目录（默认 ./output）")
    parser.add_argument("--export-params", default=None, metavar="PATH",
                        help="把当前生效的完整参数（含 CST 权重与 Hermite 控制点）"
                             "写出为 TOML，便于气动优化迭代使用")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---- 参数文件：不存在则按默认值生成，用户可直接编辑其中的数值 --------- #
    param_path = ensure_param_file(args.params)
    flat, nested = flatten_param_dict(load_param_dict(param_path))
    if args.model is not None:
        flat["model"] = args.model
    params = BWBParameters.from_dict(flat, nested)
    print(f"参数文件：{param_path.resolve()}")

    print_summary(params)

    # ---- 可选：导出完整参数（含设计变量），供优化迭代批量生成参数文件 ------ #
    if args.export_params:
        out = save_param_file(
            params.to_param_dict(),
            args.export_params,
            header=f"由 main.py 导出的完整参数（含 CST 权重与 Hermite 控制点）\n模型：{params.name}",
        )
        print(f"已导出参数文件：{Path(out).resolve()}")

    if args.no_catia:
        written = export_csv(params, args.csv_dir)
        print(f"已导出 {len(written)} 个 CSV 到 {Path(args.csv_dir).resolve()}")
        return 0

    catia, part = connect_catia()
    build_catia_model(catia, part, params)
    print("CATIA 模型生成完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

