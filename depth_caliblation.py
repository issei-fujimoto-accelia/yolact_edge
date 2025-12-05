import numpy as np

def build_bias_map(H, W,
                   d_top, d_center, d_bottom,
                   d_left, d_right):
    # 正規化座標
    y = np.linspace(-1.0, 1.0, H)
    x = np.linspace(-1.0, 1.0, W)

    # 縦方向係数
    a_v = (d_bottom + d_top) / 2.0 - d_center
    b_v = (d_bottom - d_top) / 2.0

    # 横方向係数
    a_h = (d_right + d_left) / 2.0 - d_center
    b_h = (d_right - d_left) / 2.0

    # 縦のバイアス（H×1）
    bias_v = (a_v * (y**2) + b_v * y)[:, None]
    # 横のバイアス（1×W）
    bias_h = (a_h * (x**2) + b_h * x)[None, :]

    # 全体のバイアス（H×W）
    bias = bias_v + bias_h
    return bias

def correct_depth(depth_raw, bias):
    # depth_raw, bias は同じ H×W
    return depth_raw - bias



H, W = 640, 480
bias = build_bias_map(H, W,
                      d_top=790,
                      d_center=795,
                      d_bottom=806,
                      d_left=797,
                      d_right=793)
# print(bias)