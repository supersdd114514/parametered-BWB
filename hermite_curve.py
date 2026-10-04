# -*- coding: utf-8 -*-
"""三次 Hermite（施密特）曲线参数化模块。

参考文献
--------
戴浩, 余雄庆. 翼身融合飞机参数化几何模型[J]. 飞机设计, 2012, 32(2): 11-14.
论文 1.2 节：用三次 Hermite 曲线描述飞机沿展向的厚度变化与后缘曲线形状，
用分段曲线描述复杂的前缘形状；在保持连接处一阶连续的条件下，
仅改变连接处切矢量的大小即可生成光滑的分段曲线。

数学表达式（论文式 (3)）
-----------------------
    P(t) = h00(t) * P0 + h10(t) * R0 + h01(t) * P1 + h11(t) * R1 ,  t ∈ [0, 1]

其中 P0、P1 为端点坐标，R0、R1 为端点切矢量，混合函数为
    h00(t) =  2t^3 - 3t^2 + 1
    h10(t) =    t^3 - 2t^2 + t
    h01(t) = -2t^3 + 3t^2
    h11(t) =    t^3 -   t^2

整条曲线的导数 P'(t) = dh00*P0 + dh10*R0 + dh01*P1 + dh11*R1，
因此切矢量方向决定曲线走向、模长决定“松紧”，可用来控制前缘/后缘弯曲程度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ArrayLike = Any  # 允许传入 list / tuple / ndarray 等任意“类数组”输入

__all__ = [
    "hermite_basis",
    "hermite_basis_derivative",
    "hermite_point",
    "hermite_curve",
    "HermiteSegment",
    "PiecewiseHermite",
    "tangents_from_points",
    "spanwise_thickness_curve",
]


# --------------------------------------------------------------------------- #
# 1. 混合函数与单段曲线
# --------------------------------------------------------------------------- #
def hermite_basis(t):
    """三次 Hermite 混合函数 (h00, h10, h01, h11)。"""
    t = np.asarray(t, dtype=float)
    t2 = t * t
    t3 = t2 * t
    return (
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + t,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
    )


def hermite_basis_derivative(t):
    """混合函数对参数 t 的一阶导数 (dh00, dh10, dh01, dh11)。"""
    t = np.asarray(t, dtype=float)
    return (
        6.0 * t * t - 6.0 * t,
        3.0 * t * t - 4.0 * t + 1.0,
        -6.0 * t * t + 6.0 * t,
        3.0 * t * t - 2.0 * t,
    )


def hermite_point(p0, p1, r0, r1, t):
    """计算三次 Hermite 曲线上的点（支持 2D / 3D 矢量）。

    p0, p1, r0, r1 : 长度 dim 的端点坐标 / 切矢量
    t              : 参数，标量或数组
    返回           : t 为标量时返回 (dim,)，否则返回 (len(t), dim)
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)

    scalar = np.isscalar(t) or np.asarray(t).ndim == 0
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    h00, h10, h01, h11 = hermite_basis(t_arr)

    pts = (
        h00[:, None] * p0
        + h10[:, None] * r0
        + h01[:, None] * p1
        + h11[:, None] * r1
    )
    return pts[0] if scalar else pts


def hermite_curve(p0, p1, r0, r1, num: int = 51) -> np.ndarray:
    """按等参数采样一条三次 Hermite 曲线，返回 ``(num, dim)`` 数组。"""
    t = np.linspace(0.0, 1.0, num)
    return hermite_point(p0, p1, r0, r1, t)


@dataclass
class HermiteSegment:
    """单段三次 Hermite 曲线（式 (3)）。"""

    p0: ArrayLike
    p1: ArrayLike
    r0: ArrayLike
    r1: ArrayLike

    def __post_init__(self) -> None:
        self.p0 = np.asarray(self.p0, dtype=float).ravel()
        self.p1 = np.asarray(self.p1, dtype=float).ravel()
        self.r0 = np.asarray(self.r0, dtype=float).ravel()
        self.r1 = np.asarray(self.r1, dtype=float).ravel()
        if not (self.p0.size == self.p1.size == self.r0.size == self.r1.size):
            raise ValueError("端点与切矢量维数必须一致")

    @property
    def dim(self) -> int:
        return self.p0.size

    def evaluate(self, t) -> np.ndarray:
        return hermite_point(self.p0, self.p1, self.r0, self.r1, t)

    def sample(self, num: int = 51, include_start: bool = True) -> np.ndarray:
        t = np.linspace(0.0, 1.0, num)
        return self.evaluate(t if include_start else t[1:])

    def derivative(self, t) -> np.ndarray:
        """一阶导数 dP/dt（切矢量方向），用于验证首末端切矢量与 C1 连续性。"""
        scalar = np.isscalar(t) or np.asarray(t).ndim == 0
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        dh00, dh10, dh01, dh11 = hermite_basis_derivative(t_arr)
        d = (
            dh00[:, None] * self.p0
            + dh10[:, None] * self.r0
            + dh01[:, None] * self.p1
            + dh11[:, None] * self.r1
        )
        return d[0] if scalar else d

    def length(self, num: int = 201) -> float:
        """按折线近似计算弧长。"""
        pts = self.sample(num)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# --------------------------------------------------------------------------- #
# 2. 分段三次 Hermite 曲线（C1 连续）
# --------------------------------------------------------------------------- #
@dataclass
class PiecewiseHermite:
    """分段三次 Hermite 曲线链：给定 k 个结点与其切矢量即得 k-1 段曲线。

    连接处切矢量唯一，因此天然保证一阶连续（C1），
    正对应论文“保持连接处一阶连续，仅改变连接处切矢量大小”的描述。
    """

    points: np.ndarray
    tangents: np.ndarray

    def __post_init__(self) -> None:
        self.points = np.atleast_2d(np.asarray(self.points, dtype=float))
        self.tangents = np.atleast_2d(np.asarray(self.tangents, dtype=float))
        if self.points.shape != self.tangents.shape:
            raise ValueError("points 与 tangents 形状必须一致")
        if self.points.shape[0] < 2:
            raise ValueError("至少需要 2 个结点")

    # ---- 构造 ------------------------------------------------------------ #
    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        tangents: ArrayLike | None = None,
        method: str = "catmull-rom",
        tension: float = 0.5,
    ) -> "PiecewiseHermite":
        """由结点自动生成切矢量（Catmull-Rom / cardinal / natural）。"""
        pts = np.asarray(points, dtype=float)
        if tangents is None:
            tangents = tangents_from_points(pts, method=method, tension=tension)
        return cls(pts, tangents)

    @classmethod
    def from_slopes(
        cls,
        points: ArrayLike,
        slopes: ArrayLike,
    ) -> "PiecewiseHermite":
        """二维平面曲线：由结点与各点“斜率 dy/dx”构造切矢量（参数 t 按结点序号）。

        切矢量取 (1, slope)，相当于以弦向为参数，适合描述前缘/后缘曲线。
        """
        pts = np.asarray(points, dtype=float)
        slopes = np.asarray(slopes, dtype=float).ravel()
        if pts.shape[0] != slopes.size:
            raise ValueError("结点数与斜率数必须一致")
        unit_x = np.ones_like(slopes)
        tangents = np.column_stack([unit_x, slopes])
        if pts.shape[1] > 2:  # 扩展到 3D
            tangents = np.column_stack([tangents, np.zeros_like(slopes)])
        return cls(pts, tangents)

    # ---- 基本属性 -------------------------------------------------------- #
    @property
    def num_segments(self) -> int:
        return self.points.shape[0] - 1

    @property
    def segments(self) -> list[HermiteSegment]:
        return [
            HermiteSegment(
                self.points[i], self.points[i + 1], self.tangents[i], self.tangents[i + 1]
            )
            for i in range(self.num_segments)
        ]

    # ---- 求值 ------------------------------------------------------------ #
    def evaluate(self, u) -> np.ndarray:
        """在全局参数 u ∈ [0, num_segments] 上求值。"""
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        out = np.empty((u_arr.size, self.points.shape[1]), dtype=float)
        for idx, ui in enumerate(np.clip(u_arr, 0.0, self.num_segments)):
            seg_idx = min(int(ui), self.num_segments - 1)
            local_t = ui - seg_idx
            out[idx] = self.segments[seg_idx].evaluate(local_t)
        return out

    def sample(self, per_segment: int = 20, include_start: bool = True) -> np.ndarray:
        """等参数采样，结点只出现一次（后续分段的起始点自动跳过，避免重复点）。"""
        chunks = []
        for i, seg in enumerate(self.segments):
            # i > 0 时跳过分段起点（即上一段的终点），避免结点重复
            pts = seg.sample(per_segment, include_start=(i == 0 and include_start))
            chunks.append(pts)
        return np.vstack(chunks)

    def tangent_at(self, u: float) -> np.ndarray:
        seg_idx = min(int(u), self.num_segments - 1)
        return self.segments[seg_idx].derivative(u - seg_idx)

    def length(self, per_segment: int = 100) -> float:
        pts = self.sample(per_segment)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def rescaled_tangents(self, scales: ArrayLike) -> "PiecewiseHermite":
        """返回切矢量按给定比例缩放后的新曲线（不改变结点位置与 C1 连续性）。"""
        scales = np.asarray(scales, dtype=float).ravel()
        if scales.size == 1:
            scales = np.full(self.points.shape[0], float(scales[0]))
        if scales.size != self.points.shape[0]:
            raise ValueError("缩放系数个数必须为 1 或等于结点数")
        return PiecewiseHermite(self.points, self.tangents * scales[:, None])


# --------------------------------------------------------------------------- #
# 3. 自动切矢量计算
# --------------------------------------------------------------------------- #
def tangents_from_points(
    points,
    method: str = "catmull-rom",
    tension: float = 0.5,
) -> np.ndarray:
    """由结点自动生成切矢量。

    method:
        ``catmull-rom``          中心差分，端点用单侧差分（默认，最稳健）
        ``cardinal``             带张力参数的基数样条切矢量
        ``finite-difference``    与 catmull-rom 相同的有限差分
        ``natural``              自然三次样条切矢量（二阶导数端点为零，C2）
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    k, dim = pts.shape
    if k < 2:
        raise ValueError("至少需要 2 个结点")

    if method in ("catmull-rom", "finite-difference"):
        r = np.empty_like(pts)
        r[0] = pts[1] - pts[0]
        r[-1] = pts[-1] - pts[-2]
        if k > 2:
            r[1:-1] = 0.5 * (pts[2:] - pts[:-2])
        return r

    if method == "cardinal":
        s = (1.0 - float(tension)) / 2.0
        r = np.empty_like(pts)
        r[0] = s * (pts[1] - pts[0]) * 2.0
        r[-1] = s * (pts[-1] - pts[-2]) * 2.0
        if k > 2:
            r[1:-1] = s * (pts[2:] - pts[:-2])
        return r

    if method == "natural":
        # 均匀参数化下的自然三次样条一阶导数方程组
        dim = pts.shape[1]
        n = k
        a = np.zeros((n, n))
        b = np.zeros((n, dim))
        a[0, 0], a[0, 1] = 2.0, 1.0
        b[0] = 3.0 * (pts[1] - pts[0])
        for i in range(1, n - 1):
            a[i, i - 1] = 1.0
            a[i, i] = 4.0
            a[i, i + 1] = 1.0
            b[i] = 3.0 * (pts[i + 1] - pts[i - 1])
        a[-1, -2], a[-1, -1] = 1.0, 2.0
        b[-1] = 3.0 * (pts[-1] - pts[-2])
        return np.linalg.solve(a, b)

    raise ValueError(f"未知切矢量计算方法: {method}")


# --------------------------------------------------------------------------- #
# 4. 翼身融合体专用：融合段展向厚度变化
# --------------------------------------------------------------------------- #
def spanwise_thickness_curve(
    y_root: float,
    y_tip: float,
    thickness_root: float,
    thickness_tip: float,
    slope_root: float,
    slope_tip: float,
    num: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    """用三次 Hermite 曲线描述融合段展向相对厚度变化（论文 2(3) 节）。

    起点取在根部最大厚度处，控制参数即首末端点切矢量（斜率）。

    返回
    ----
    y        : 展向坐标序列, 形状 (num,)
    tc       : 对应相对厚度 t/c 序列, 形状 (num,)
    """
    p0 = np.array([float(y_root), float(thickness_root)])
    p1 = np.array([float(y_tip), float(thickness_tip)])
    # 以展向为参数，切矢量 x 分量取 1，y 分量即 d(t/c)/dy
    r0 = np.array([1.0, float(slope_root)])
    r1 = np.array([1.0, float(slope_tip)])
    curve = hermite_curve(p0, p1, r0, r1, num)
    return curve[:, 0], curve[:, 1]


# --------------------------------------------------------------------------- #
# 5. 自检 / 演示
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 68)
    print("三次 Hermite 曲线模块自检")
    print("=" * 68)

    seg = HermiteSegment(
        p0=(0.0, 0.0), p1=(1.0, 1.0), r0=(1.0, 0.0), r1=(1.0, 0.0)
    )
    print(f"单段曲线: 起点 {seg.evaluate(0.0)}, 终点 {seg.evaluate(1.0)}")
    print(f"         t=0 切矢量 {seg.derivative(0.0)}  (应等于 R0 = [1. 0.])")
    print(f"         t=1 切矢量 {seg.derivative(1.0)}  (应等于 R1 = [1. 0.])")
    print(f"         弧长 ≈ {seg.length():.6f}")

    nodes = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5), (3.0, 1.2)]
    curve = PiecewiseHermite.from_points(nodes, method="catmull-rom")
    pts = curve.sample(10)
    l = curve.segments[0].derivative(1.0)
    r = curve.segments[1].derivative(0.0)
    print(f"分段曲线: {curve.num_segments} 段, 采样 {pts.shape[0]} 点, 弧长 ≈ {curve.length():.4f}")
    print(f"         连接处 C1 连续性: 左导数 {l} / 右导数 {r}  -> 相等 {np.allclose(l, r)}")

    y, tc = spanwise_thickness_curve(0.0, 4000.0, 0.20, 0.10, 0.0, -0.00002, 51)
    print(f"展向厚度分布: y∈[{y[0]:.0f}, {y[-1]:.0f}] mm, "
          f"t/c: {tc[0]:.4f} -> {tc[-1]:.4f}")
