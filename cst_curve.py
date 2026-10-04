# -*- coding: utf-8 -*-
"""CST（Class-Shape Transformation，分类函数 / 形状函数变换）参数化曲线与翼型模块。

参考文献
--------
戴浩, 余雄庆. 翼身融合飞机参数化几何模型[J]. 飞机设计, 2012, 32(2): 11-14.
Kulfan B. Universal parametric geometry representation method[J].
    Journal of Aircraft, 2008, 45(1): 142-158.

基本表达式（对应论文式 (1)~(3)）
-------------------------------
    zeta(psi) = C(psi) * S(psi) + T(psi)                      ......(1)
    C(psi)    = psi**N1 * (1 - psi)**N2                       ......(2) 分类函数
    S(psi)    = sum_{i=0}^{n} A_i * B_i^n(psi)                ......(3) 形状函数
    B_i^n(psi)= C(n, i) * psi**i * (1 - psi)**(n - i)         ......(4) 伯恩斯坦多项式
    T(psi)    = psi * dzeta_TE                                ......(5) 厚度（后缘）函数

要点
----
* N1 = 0.5、N2 = 1.0 时描述“圆头前缘 / 尖形后缘”类翼型，常用翼型与超临界翼型均可此取值。
* 取 n 阶伯恩斯坦多项式时，一条曲线共需 n + 4 个参数；对某类几何形状 N1、N2、dzeta_TE
  是确定的，于是只剩 n + 1 个形状参数 A_i。实际计算中只需取曲线上 n + 1 个型值点，
  求解 n + 1 阶线性方程组即可确定全部 A_i（点数多于 n + 1 时用最小二乘逼近）。
* 论文中伯恩斯坦多项式阶数取 8（即 9 个形状参数），生成的 NASA SC(2)0518 与原始翼型吻合。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

__all__ = [
    "cosine_spacing",
    "class_function",
    "bernstein_basis",
    "shape_function",
    "cst_ordinates",
    "fit_cst",
    "leading_edge_radius",
    "naca_half_thickness",
    "naca_camber_line",
    "CSTAirfoil",
    "make_cst_section",
    "cst_weights_from_naca",
    "make_cst_section_from_weights",
    "smoothness_report",
]


# --------------------------------------------------------------------------- #
# 1. 基础函数
# --------------------------------------------------------------------------- #
def cosine_spacing(num: int) -> np.ndarray:
    """余弦分布结点：前缘密、后缘疏，符合翼型曲率变化特点。返回 [0, 1]。"""
    if num < 2:
        raise ValueError("num 至少为 2")
    theta = np.linspace(0.0, np.pi, num)
    return 0.5 * (1.0 - np.cos(theta))


def class_function(psi, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """分类函数 C(psi) = psi**N1 * (1 - psi)**N2  （论文式 (2)）。"""
    psi = np.asarray(psi, dtype=float)
    return np.power(psi, n1) * np.power(1.0 - psi, n2)


def bernstein_basis(psi, order: int) -> np.ndarray:
    """伯恩斯坦多项式基 B_i^n(psi)（论文式 (4)）。

    返回形状为 ``(len(psi), order + 1)`` 的矩阵，第 i 列为 B_i^n(psi)。
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    n = int(order)
    basis = np.empty((psi.size, n + 1), dtype=float)
    one_minus = 1.0 - psi
    for i in range(n + 1):
        basis[:, i] = comb(n, i) * np.power(psi, i) * np.power(one_minus, n - i)
    return basis


def shape_function(psi, weights, order: int | None = None) -> np.ndarray:
    """形状函数 S(psi) = sum_i A_i * B_i^n(psi)（论文式 (3)）。"""
    weights = np.asarray(weights, dtype=float).ravel()
    n = len(weights) - 1 if order is None else int(order)
    if n != len(weights) - 1:
        raise ValueError("order 与形状参数个数不一致")
    return bernstein_basis(psi, n) @ weights


def cst_ordinates(
    psi,
    weights,
    n1: float = 0.5,
    n2: float = 1.0,
    te_thickness: float = 0.0,
) -> np.ndarray:
    """由形状参数计算 CST 曲线高度 zeta(psi)（论文式 (1)）。

    参数
    ----
    psi        : 归一化弦向坐标（0 = 前缘，1 = 后缘）
    weights    : 形状参数 A_0 ... A_n（n + 1 个）
    n1, n2     : 分类函数参数
    te_thickness : 后缘处的 zeta 值 dzeta_TE（厚度函数 T = psi * dzeta_TE）
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    return class_function(psi, n1, n2) * shape_function(psi, weights) + psi * te_thickness


def fit_cst(
    psi,
    zeta,
    order: int,
    n1: float = 0.5,
    n2: float = 1.0,
    te_thickness: float = 0.0,
    scaled: bool = True,
) -> np.ndarray:
    """由型值点反求 CST 形状参数 A_i（求解 n + 1 阶线性方程组）。

    CST 表达式为 ``zeta = C(psi) * S(psi) + psi * dzte``，因此

        形状函数目标值：  S(psi) = [zeta(psi) - psi * dzte] / C(psi)

    ``scaled=True``（默认）直接对 S 做最小二乘，这是标准做法，也是本模块
    曲线质量的关键：分类函数 C(psi) 在前后缘趋于 0，若改对 ``C*S`` 做拟合
    （``scaled=False``），两端的目标会被 C 的微小权重淹没，
    最小二乘只能用正负交替的大幅权重去凑 → **曲线出现振荡**。
    所以只要保证后缘自洽（``zeta(1) == dzte``），除以 C 的拟合就是良态的。

    注意：前缘（psi = 0）与后缘（psi = 1）处 C = 0，方程退化，
    因此自动剔除端点，仅用内部型值点构造方程组。
    """
    psi = np.asarray(psi, dtype=float).ravel()
    zeta = np.asarray(zeta, dtype=float).ravel()
    if psi.size != zeta.size:
        raise ValueError("psi 与 zeta 长度必须一致")

    mask = (psi > 0.0) & (psi < 1.0)
    psi_i, zeta_i = psi[mask], zeta[mask]
    if psi_i.size < order + 1:
        raise ValueError(
            f"内部型值点不足：需要至少 {order + 1} 个 (0, 1) 区间内的点，实得 {psi_i.size}"
        )

    residual = zeta_i - psi_i * te_thickness
    basis = bernstein_basis(psi_i, order)

    if scaled:
        # S(psi) = residual / C(psi)，直接拟合形状函数（良态、无振荡）
        target = residual / class_function(psi_i, n1, n2)
    else:
        # 旧做法：拟合 C*S，前后缘权重极小，易产生振荡（仅保留用于对比）
        basis = basis * class_function(psi_i, n1, n2)[:, None]
        target = residual

    if psi_i.size == order + 1:
        return np.linalg.solve(basis, target)
    weights, *_ = np.linalg.lstsq(basis, target, rcond=None)
    return weights


def leading_edge_radius(a0: float, n1: float = 0.5) -> float:
    """由首个形状参数估算前缘半径：R_le = A_0**2 / 2 （N1 = 0.5，单位弦长）。"""
    return 0.5 * float(a0) ** 2


# --------------------------------------------------------------------------- #
# 2. NACA 基准外形（用于生成基准厚度/弯度并验证 CST 拟合精度）
# --------------------------------------------------------------------------- #
def naca_half_thickness(psi, thickness_ratio: float = 1.0, closed_te: bool = True) -> np.ndarray:
    """NACA 四位翼型的半厚度分布 yt(psi)。

    thickness_ratio = 1.0 时最大半厚度为 0.5（即最大相对厚度 1.0）。
    closed_te = True 时末项系数取 -0.1036，使后缘封闭。
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5.0 * thickness_ratio * (
        0.2969 * np.sqrt(psi)
        - 0.1260 * psi
        - 0.3516 * psi**2
        + 0.2843 * psi**3
        + a4 * psi**4
    )
    return yt


def naca_camber_line(
    psi,
    camber_ratio: float = 0.0,
    camber_pos: float = 0.4,
    reflex: float = 0.0,
) -> np.ndarray:
    """NACA 四位翼型弯度线，可选反弯（reflex）修正。

    反弯修正叠加 ``reflex * slope_TE * psi * (1 - psi)``：

    * 该项在前、后缘均为 0，因此 **始终满足 zc(0) = zc(1) = 0**，
      不会把后缘抬高（这一点至关重要：后缘若不为 0，
      CST 的 psi*dzte 项与之不自洽，拟合会剧烈振荡）；
    * 叠加后后缘弯度斜率变为 ``slope_TE * (1 - reflex)``，
      故 ``reflex = 1`` 时后缘弯度斜率归零（完全反弯），
      ``reflex = 0.5`` 时为部分反弯。
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    m, p = float(camber_ratio), float(camber_pos)
    if m == 0.0:
        return np.zeros_like(psi)
    if not 0.0 < p < 1.0:
        raise ValueError("camber_pos 必须在 (0, 1) 区间内")

    zc = np.where(
        psi < p,
        m / p**2 * (2.0 * p * psi - psi**2),
        m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * psi - psi**2),
    )
    if reflex:
        slope_te = -2.0 * m / (1.0 - p)   # 基准弯度线在后缘的斜率
        zc = zc + float(reflex) * slope_te * psi * (1.0 - psi)
    return zc


# --------------------------------------------------------------------------- #
# 3. CST 翼型对象
# --------------------------------------------------------------------------- #
@dataclass
class CSTAirfoil:
    """由 CST 形状参数定义的翼型（上下表面各一组 A_i）。

    upper / lower      : 上、下表面形状参数（n + 1 个）
    n1, n2             : 分类函数参数（圆头前缘/尖形后缘取 0.5 / 1.0）
    te_upper/te_lower  : 上、下表面在 psi = 1 处的 zeta 值，即各自的后缘项

    后缘项为什么上下分开给：弯度线在后缘一般不为零，因此上下表面的后缘值
    并不等大反号（``zeta_u(1) = zc(1) + t/2``，``zeta_l(1) = zc(1) - t/2``）。
    用 ``te_upper`` / ``te_lower`` 表达才能保证与目标后缘严格自洽。
    """

    upper: np.ndarray
    lower: np.ndarray
    n1: float = 0.5
    n2: float = 1.0
    te_upper: float = 0.0
    te_lower: float = 0.0
    name: str = "cst"

    def __post_init__(self) -> None:
        self.upper = np.asarray(self.upper, dtype=float).ravel()
        self.lower = np.asarray(self.lower, dtype=float).ravel()
        if self.upper.size != self.lower.size:
            raise ValueError("上/下表面形状参数个数必须一致")

    # ---- 基本属性 -------------------------------------------------------- #
    @property
    def order(self) -> int:
        """伯恩斯坦多项式阶数 n（形状参数个数为 n + 1）。"""
        return len(self.upper) - 1

    @property
    def te_thickness(self) -> float:
        """后缘相对厚度 = 上后缘 - 下后缘。"""
        return float(self.te_upper - self.te_lower)

    @te_thickness.setter
    def te_thickness(self, value: float) -> None:
        """按上下对称分配设置后缘厚度（兼容旧用法）。"""
        self.te_upper = 0.5 * float(value)
        self.te_lower = -0.5 * float(value)

    def upper_ordinates(self, num: int = 101):
        psi = cosine_spacing(num)
        return psi, cst_ordinates(psi, self.upper, self.n1, self.n2, self.te_upper)

    def lower_ordinates(self, num: int = 101):
        psi = cosine_spacing(num)
        return psi, cst_ordinates(psi, self.lower, self.n1, self.n2, self.te_lower)

    def coordinates(self, num: int = 101) -> np.ndarray:
        """闭合翼型坐标 ``(m, 2)``：后缘 → 上表面 → 前缘 → 下表面 → 后缘。"""
        psi, zu = self.upper_ordinates(num)
        _, zl = self.lower_ordinates(num)
        upper_half = np.column_stack([psi[::-1], zu[::-1]])   # TE -> LE
        lower_half = np.column_stack([psi[1:], zl[1:]])        # LE -> TE
        return np.vstack([upper_half, lower_half])

    def thickness(self, num: int = 201) -> tuple[np.ndarray, np.ndarray]:
        psi, zu = self.upper_ordinates(num)
        _, zl = self.lower_ordinates(num)
        return psi, zu - zl

    def camber(self, num: int = 201) -> tuple[np.ndarray, np.ndarray]:
        psi, zu = self.upper_ordinates(num)
        _, zl = self.lower_ordinates(num)
        return psi, 0.5 * (zu + zl)

    @property
    def max_thickness_ratio(self) -> float:
        _, t = self.thickness(401)
        return float(t.max())

    @property
    def max_thickness_position(self) -> float:
        psi, t = self.thickness(401)
        return float(psi[int(np.argmax(t))])

    @property
    def leading_edge_radius(self) -> float:
        return leading_edge_radius(float(self.upper[0]), self.n1)

    # ---- 构造方法 -------------------------------------------------------- #
    @classmethod
    def from_points(
        cls,
        psi,
        zeta_upper,
        zeta_lower,
        order: int = 8,
        n1: float = 0.5,
        n2: float = 1.0,
        te_upper: float = 0.0,
        te_lower: float = 0.0,
        name: str = "cst",
    ) -> "CSTAirfoil":
        """由上下表面型值点拟合 CST 形状参数。

        ``te_upper`` / ``te_lower`` 必须等于目标曲线在 psi = 1 处的值，
        否则后缘不自洽、拟合会产生振荡（详见 make_cst_section 的说明）。
        """
        upper = fit_cst(psi, zeta_upper, order, n1, n2, te_upper)
        lower = fit_cst(psi, zeta_lower, order, n1, n2, te_lower)
        return cls(upper, lower, n1, n2, te_upper, te_lower, name)

    def scaled_thickness(self, thickness_ratio: float) -> "CSTAirfoil":
        """保持弯度、按目标相对厚度线性缩放厚度分量（返回重新拟合后的对象）。"""
        psi, t = self.thickness(401)
        factor = thickness_ratio / float(t.max())
        zc = self.camber(401)[1]
        zu = zc + 0.5 * t * factor
        zl = zc - 0.5 * t * factor
        return CSTAirfoil.from_points(
            psi, zu, zl, self.order, self.n1, self.n2,
            self.te_upper * factor, self.te_lower * factor, self.name,
        )

    def scaled(self, scale: float) -> "CSTAirfoil":
        """整体等比缩放（对无量纲翼型通常不用；保留用于完整性）。"""
        return CSTAirfoil(
            self.upper * scale, self.lower * scale, self.n1, self.n2,
            self.te_upper * scale, self.te_lower * scale, self.name,
        )


# --------------------------------------------------------------------------- #
# 4. 由设计参数直接生成 CST 翼型剖面
# --------------------------------------------------------------------------- #
def make_cst_section(
    thickness_ratio: float = 0.12,
    camber_ratio: float = 0.02,
    camber_pos: float = 0.40,
    reflex: float = 0.0,
    order: int = 8,
    n1: float = 0.5,
    n2: float = 1.0,
    te_thickness: float = 0.0025,
    num: int = 201,
    name: str = "section",
) -> CSTAirfoil:
    """按设计参数生成 CST 翼型剖面（厚度/弯度权重模式的便捷入口）。

    流程（对应论文 2(2) 节“应用 CST 方法参数化描述翼型”）：
        1. 由 NACA 厚度分布与弯度线（可选反弯）构造目标半厚度与弯度线；
        2. 分别拟合成 CST 厚度权重 A_t 与弯度权重 A_c（后缘严格自洽，
           即目标在 psi = 1 处的值必须等于 CST 的 psi*dzte 项）；
        3. 按 thickness_ratio / camber_ratio 缩放并组合：
               A_upper = A_c + A_t,   A_lower = A_c - A_t

    关于后缘自洽（曲线质量的关键）
    -----------------------------
    分类函数 C(psi) = psi^0.5 (1-psi) 在前后缘为 0，故 CST 在 psi = 1 的取值
    完全由后缘项 dzte 决定。若目标后缘与 dzte 不一致，最小二乘只能用大幅
    正负交替的权重去硬凑，曲线就会抖动。因此这里保证：
        zc(1) = 0（反弯项在两端皆为 0）且 半厚度(1) = te/2，
    于是 目标后缘 = ±te/2 = dzte，严格自洽。

    关于 ``camber_ratio`` 的含义
    ---------------------------
    这里统一为**实际最大弯度**（已包含反弯修正的影响）。反弯会把弯度线压成
    S 形、最大弯度明显减小，本函数按最大弯度归一化，因此要 2% 的实际弯度
    直接填 0.02 即可；若希望复现普通 NACA 弯度线（不减弱弯度），把 reflex 设 0。
    """
    t_w, c_w = cst_weights_from_naca(
        thickness_ratio=thickness_ratio,
        camber_ratio=camber_ratio,
        camber_pos=camber_pos,
        reflex=reflex,
        order=order,
        n1=n1,
        n2=n2,
        te_thickness=te_thickness,
        num=num,
    )
    return make_cst_section_from_weights(
        t_w, c_w, thickness_ratio, camber_ratio, n1, n2, te_thickness, name
    )


# --------------------------------------------------------------------------- #
# 4b. 厚度 / 弯度形式：便于把 CST 形状参数直接作为气动优化的设计变量
# --------------------------------------------------------------------------- #
def cst_weights_from_naca(
    thickness_ratio: float = 0.20,
    camber_ratio: float = 0.02,
    camber_pos: float = 0.40,
    reflex: float = 1.0,
    order: int = 8,
    n1: float = 0.5,
    n2: float = 1.0,
    te_thickness: float = 0.0025,
    num: int = 201,
) -> tuple[np.ndarray, np.ndarray]:
    """把默认 NACA 基准翼型写成 CST 形状参数（厚度 / 弯度各一组）。

    返回 ``(thickness_weights, camber_weights)``。

    为什么要拆成厚度 + 弯度：
        ``upper = camber + thickness``、``lower = camber - thickness`` 是线性关系，
        而 CST 对形状参数也是线性的，因此两组权重可以**精确**互换：

            A_upper = A_camber + A_thickness
            A_lower = A_camber - A_thickness

        好处是展向缩放厚度时只需缩放 ``A_thickness``，弯度保持不变——
        这也让"厚度权重 / 弯度权重"成为最适合气动优化的设计变量。
    """
    psi = cosine_spacing(num)
    half_thickness = (
        naca_half_thickness(psi, thickness_ratio=thickness_ratio, closed_te=True)
        + 0.5 * te_thickness * psi**2
    )
    camber = naca_camber_line(psi, camber_ratio, camber_pos, reflex)

    # 厚度：半厚度在 psi=1 处为 te/2；弯度：zc(1) = 0（后缘自洽）
    t_weights = fit_cst(psi, half_thickness, order, n1, n2, 0.5 * te_thickness)
    c_weights = fit_cst(psi, camber, order, n1, n2, 0.0)
    return t_weights, c_weights


def make_cst_section_from_weights(
    thickness_weights,
    camber_weights,
    thickness_ratio: float = 0.12,
    camber_ratio: float = 0.02,
    n1: float = 0.5,
    n2: float = 1.0,
    te_thickness: float = 0.0025,
    name: str = "section",
) -> CSTAirfoil:
    """由厚度 / 弯度形状参数生成剖面，并按目标相对厚度与最大弯度缩放。

    形状参数是无量纲的“单位形状”；每个剖面只需按展向的 ``thickness_ratio``
    与 ``camber_ratio`` 缩放即可，因此**同一组权重就能控制整机所有剖面**，
    这也正是把 A_i 作为气动优化设计变量的原因。
    缩放对 CST 是精确的（C*S 与 psi*dzte 都随权重线性缩放）：

        zeta(psi; s*A, s*dzte) = s * zeta(psi; A, dzte)

    约定：
        ``thickness_ratio`` = 最大相对厚度（各剖面独立指定，实现展向厚度分布）；
        ``camber_ratio``    = 实际最大弯度；
        ``te_thickness``    = 后缘相对厚度，**不随厚度缩放**（所有剖面保持一致）。
    """
    t_w = np.asarray(thickness_weights, dtype=float).ravel()
    c_w = np.asarray(camber_weights, dtype=float).ravel()
    if t_w.size != c_w.size:
        raise ValueError("厚度权重与弯度权重个数必须一致")

    # 密集均匀网格上搜索最大值，保证 t/c 与弯度缩放到位
    psi = np.linspace(0.0, 1.0, 401)

    # --- 厚度：只缩放形状部分，后缘厚度保持参数值不变 --- #
    thick_shape = cst_ordinates(psi, t_w, n1, n2, 0.0)
    base_max = 2.0 * float(np.max(thick_shape))
    s_t = (thickness_ratio / base_max) if base_max > 1e-12 else 0.0
    for _ in range(3):   # 叠加固定后缘项后最大厚度略变，迭代修正
        t_now = s_t * thick_shape + 0.5 * te_thickness * psi
        now = 2.0 * float(np.max(t_now))
        if now <= 1e-12:
            break
        s_t *= thickness_ratio / now

    # --- 弯度：按最大弯度缩放（弯度权重的后缘项为 0） --- #
    zc = cst_ordinates(psi, c_w, n1, n2, 0.0)
    c_max = float(np.max(np.abs(zc)))
    s_c = (camber_ratio / c_max) if c_max > 1e-12 else 0.0

    # 上下表面权重 = 弯度权重 ± 缩放后的厚度权重；
    # 后缘项 = 弯度后缘(0) ± te/2，由 CST 的线性性质自动精确合成。
    return CSTAirfoil(
        s_c * c_w + s_t * t_w,
        s_c * c_w - s_t * t_w,
        n1,
        n2,
        0.5 * te_thickness,
        -0.5 * te_thickness,
        name,
    )


# --------------------------------------------------------------------------- #
# 5. 自检 / 演示
# --------------------------------------------------------------------------- #
def smoothness_report(zeta, label: str = "", tol: float = 1e-6):
    """检查一条曲线是否振荡：统计二阶差分（离散曲率）的符号变化次数。

    解析光滑的翼型表面（前缘上凸、后缘附近单调）符号变化应 ≲ 2 次；
    若出现 4 次及以上，说明曲线在抖动——这是 CST 拟合不自洽的典型症状。
    返回 (符号变化次数, 二阶差分绝对值最大处的相对幅值)。
    """
    zeta = np.asarray(zeta, dtype=float).ravel()
    d2 = np.diff(zeta, 2)
    scale = np.abs(d2).max()
    if scale == 0.0:
        return 0, 0.0
    signif = d2[np.abs(d2) > tol * scale]
    flips = int(np.sum(np.diff(np.sign(signif)) != 0)) if signif.size > 1 else 0
    print(f"  光滑性[{label}]: 二阶差分符号变化 {flips} 次 (≥4 视为振荡); "
          f"最大二阶差分 {scale:.3e}")
    return flips, float(scale)


if __name__ == "__main__":
    print("=" * 72)
    print("CST 曲线模块自检")
    print("=" * 72)

    # ---- 1) 与解析 NACA 2412 对比（验证拟合精度） ------------------------ #
    psi_fit = cosine_spacing(201)
    half = naca_half_thickness(psi_fit, thickness_ratio=0.12, closed_te=True)
    cam = naca_camber_line(psi_fit, 0.02, 0.4, reflex=0.0)
    foil = CSTAirfoil.from_points(psi_fit, cam + half, cam - half, order=8, name="NACA2412")
    psi_chk = np.linspace(0.0, 1.0, 401)
    print("[1] 以 NACA 2412 为基准，用 8 阶伯恩斯坦多项式拟合")
    print(f"    形状参数个数 : {foil.upper.size} (阶数 n = {foil.order})")
    print(f"    最大相对厚度 : {foil.max_thickness_ratio:.5f} (目标 0.12000)")
    print(f"    前缘半径     : {foil.leading_edge_radius:.5f}")
    err = (
        cst_ordinates(psi_chk, foil.upper, foil.n1, foil.n2, foil.te_upper)
        - np.interp(psi_chk, psi_fit, cam + half)
    )
    print(f"    拟合最大误差 : {np.abs(err).max():.3e}")

    # ---- 2) 光滑性（振荡）检查：这是曲线质量的核心指标 ------------------- #
    print("[2] 光滑性检查（二阶差分符号变化；≥4 表示曲线抖动）")
    smoothness_report(cst_ordinates(psi_chk, foil.upper, foil.n1, foil.n2, foil.te_upper),
                      "NACA2412 上表面")

    # ---- 3) 后缘自洽性检查（曲线质量的关键） ----------------------------- #
    print("[3] 后缘自洽性检查（BWB 参数：t/c=0.20, 弯度 0.02, 反弯 1.0）")
    sec = make_cst_section(
        thickness_ratio=0.20, camber_ratio=0.02, camber_pos=0.40, reflex=1.0,
        order=8, te_thickness=0.0025, num=201, name="BWB",
    )
    psi_s = cosine_spacing(201)
    half_s = naca_half_thickness(psi_s, 0.20, closed_te=True) + 0.5 * 0.0025 * psi_s**2
    cam_s = naca_camber_line(psi_s, 0.02, 0.40, 1.0)
    print(f"    弯度线 zc(0)={cam_s[0]:+.6f}  zc(1)={cam_s[-1]:+.6f}  (后缘必须为 0)")
    print(f"    目标半厚度(1)={half_s[-1]:+.6f}  目标 te/2={0.5 * 0.0025:+.6f}  "
          f"差={half_s[-1] - 0.5 * 0.0025:+.2e}")
    print(f"    模型 dzte_上={sec.te_upper:+.6f}  dzte_下={sec.te_lower:+.6f}  "
          f"后缘厚度={sec.te_thickness:.6f}")
    print(f"    实测 t/c={sec.max_thickness_ratio:.5f}  实测最大弯度={sec.camber(401)[1].max():+.5f}")
    smoothness_report(cst_ordinates(psi_chk, sec.upper, 0.5, 1.0, sec.te_upper), "BWB 上表面")
    smoothness_report(cst_ordinates(psi_chk, sec.lower, 0.5, 1.0, sec.te_lower), "BWB 下表面")

    # ---- 4) 厚度/弯度权重模式（气动优化用的设计变量） --------------------- #
    print("[4] 厚度/弯度权重模式（A_i 即气动优化的设计变量）")
    t_w, c_w = cst_weights_from_naca(
        thickness_ratio=0.20, camber_ratio=0.02, camber_pos=0.40, reflex=1.0,
        order=8, te_thickness=0.0025, num=201,
    )
    print("    厚度权重 A_t =", np.array2string(t_w, precision=5))
    print("    弯度权重 A_c =", np.array2string(c_w, precision=5))
    for tc, cm in ((0.20, 0.02), (0.14, 0.02), (0.10, 0.01), (0.07, 0.00)):
        s = make_cst_section_from_weights(t_w, c_w, thickness_ratio=tc, camber_ratio=cm,
                                          n1=0.5, n2=1.0, te_thickness=0.0025, name="w")
        print(f"    目标 t/c={tc:.2f} 弯度={cm:.3f} -> 实测 t/c={s.max_thickness_ratio:.5f}  "
              f"实测弯度={s.camber(401)[1].max():+.5f}  后缘厚度={s.te_thickness:.5f}")

    # 线性分解恒等式：upper = 弯度 + 半厚度，lower = 弯度 - 半厚度（应精确成立）
    psi_l = np.linspace(0.0, 1.0, 401)
    zu = cst_ordinates(psi_l, sec.upper, sec.n1, sec.n2, sec.te_upper)
    zl = cst_ordinates(psi_l, sec.lower, sec.n1, sec.n2, sec.te_lower)
    t_half = cst_ordinates(psi_l, 0.5 * (sec.upper - sec.lower), sec.n1, sec.n2,
                           0.5 * sec.te_thickness)
    zc = cst_ordinates(psi_l, 0.5 * (sec.upper + sec.lower), sec.n1, sec.n2,
                       0.5 * (sec.te_upper + sec.te_lower))
    print(f"    分解恒等式误差: max|zu-(zc+t)|={np.abs(zu - (zc + t_half)).max():.2e}  "
          f"max|zl-(zc-t)|={np.abs(zl - (zc - t_half)).max():.2e}")

    # 厚度形状仍与解析 NACA 厚度一致
    err_t = np.abs(
        cst_ordinates(psi_s, 0.5 * (sec.upper - sec.lower), 0.5, 1.0, 0.5 * sec.te_thickness)
        - half_s * (sec.max_thickness_ratio / 0.20)
    ).max()
    print(f"    厚度形状与解析 NACA 的最大偏差 = {err_t:.3e}")
