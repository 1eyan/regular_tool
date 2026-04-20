"""
OVT (Offset Vector Tile) domain transformation module.

从炮检域 (shot-receiver) 转换到 OVT 域 (midpoint + half-offset)。
主要流程：
    1. 读取 SEG-Y 道头（复用 Segy2H5.py 的读取逻辑）
    2. 还原坐标（复用 _scale_coords 缩放逻辑）
    3. 计算 midpoint / half-offset
    4. 对 mx, my, hx, hy 离散分箱
    5. 按 OVT cell 聚合 trace
    6. 输出 trace 级几何表 + OVT gather 索引

用法示例：
    python ovt_domain.py <segy_file> [--mx_bin 50] [--my_bin 50] [--hx_bin 25] [--hy_bin 25]

或作为模块导入：
    from ovt_domain import read_trace_headers, restore_coordinates,
                           compute_midpoint_offset, build_ovt_bins,
                           group_traces_by_ovt, build_ovt_geometry_table
"""

from pathlib import Path
import os
import struct
import argparse
import json

import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 复用 Segy2H5.py 的常量与辅助函数（通过直接复制保证独立性）
# ---------------------------------------------------------------------------

BYTE_POS = {
    'shot_line':   17,
    'shot_no':     25,
    'recv_line':   61,
    'recv_stake':  65,
    'shot_x':      73,
    'shot_y':      77,
    'rec_x':       81,
    'rec_y':       85,
    'shot_stake':  21,
    'recv_no':     41,
    'cmp':         193,
    'cmp_line':    189,
    'offset':      37,
}
BYTE_POS_ = {
    'shot_x': 73,
    'shot_y': 77,
    'rec_x':  81,
    'rec_y':  85,
}


def _read_bin_header_format_and_ns(f):
    """读取 SEG-Y 二进制头，返回 (fmt_code, ns_from_bin)。"""
    f.seek(3200, 0)
    binhdr = f.read(400)
    ns_from_bin = struct.unpack('>H', binhdr[20:22])[0]
    fmt_code    = struct.unpack('>H', binhdr[24:26])[0]
    return fmt_code, ns_from_bin


def _bps_from_fmt(fmt: int) -> int:
    """format code → bytes per sample。"""
    if fmt in (1, 2, 5): return 4
    if fmt == 3: return 2
    if fmt == 8: return 1
    return 4


def _scale_coords(values, scalars):
    """
    坐标缩放（SEGY scalco 规则）。
    - scalco > 0: real = raw * scalco
    - scalco < 0: real = raw / abs(scalco)
    - scalco == 0: real = raw
    """
    v = np.asarray(values, dtype=np.float64)
    if scalars is None:
        return v
    s = np.asarray(scalars, dtype=np.int64)
    out = v.astype(np.float64, copy=True)
    pos = s > 0
    neg = s < 0
    out[pos] = out[pos] * s[pos]
    out[neg] = out[neg] / np.abs(s[neg])
    return out


# ---------------------------------------------------------------------------
# Step 1: 读取道头
# ---------------------------------------------------------------------------

def read_trace_headers(path: Path, mode: str = 'self_computed'):
    """
    从 SEG-Y 读取道头。

    Parameters
    ----------
    path : Path
        SEG-Y 文件路径。
    mode : str
        - 'self_computed': 只读 sx, sy, rx, ry（使用 segyio 的 scalar 自动缩放）
        - 'fixed': 读全部 BYTE_POS 字段，使用 i32be 手动解析

    Returns
    -------
    headers : dict
        键为字段名，值为 list（trace 数组成员）。
    """
    if mode == 'self_computed':
        # 使用 segyio 读取 scalar + 坐标，再手动应用 scalco 规则
        with segyio.open(path, ignore_geometry=True) as f:
            scalar = np.abs(
                f.attributes(segyio.TraceField.SourceGroupScalar)[:].astype(np.float32)
            )
            scalar[scalar == 0] = 1.0  # scalar=0 等价于不缩放

        out = {'trace': [], 'shot_x': [], 'shot_y': [], 'rec_x': [], 'rec_y': []}
        with open(path, 'rb') as f:
            fmt, _ = _read_bin_header_format_and_ns(f)
            bps = _bps_from_fmt(fmt)
            f.seek(3600, 0)
            t = 0
            while True:
                hdr = f.read(240)
                if len(hdr) < 240:
                    break

                def i32be(pos1b):
                    j0 = pos1b - 1
                    return struct.unpack('>i', hdr[j0:j0 + 4])[0]

                out['trace'].append(t)
                out['shot_x'].append(i32be(BYTE_POS_['shot_x']))
                out['shot_y'].append(i32be(BYTE_POS_['shot_y']))
                out['rec_x'].append(i32be(BYTE_POS_['rec_x']))
                out['rec_y'].append(i32be(BYTE_POS_['rec_y']))
                ns = struct.unpack('>H', hdr[114:116])[0]
                f.seek(ns * bps, 1)
                t += 1

        df = pd.DataFrame(out)
        # 用 segyio 的 scalar 做统一缩放（与 Segy2H5.py 的 self_computed 模式一致）
        df['shot_x'] = _scale_coords(df['shot_x'].to_numpy(), scalar)
        df['shot_y'] = _scale_coords(df['shot_y'].to_numpy(), scalar)
        df['rec_x']  = _scale_coords(df['rec_x'].to_numpy(),  scalar)
        df['rec_y']  = _scale_coords(df['rec_y'].to_numpy(),  scalar)
        return df

    elif mode == 'fixed':
        out = {'trace': []}
        for key in BYTE_POS.keys():
            out[key] = []
        with open(path, 'rb') as f:
            fmt, _ = _read_bin_header_format_and_ns(f)
            bps = _bps_from_fmt(fmt)
            f.seek(3600, 0)
            t = 0
            while True:
                hdr = f.read(240)
                if len(hdr) < 240:
                    break

                def i32be(pos1b):
                    j0 = pos1b - 1
                    return struct.unpack('>i', hdr[j0:j0 + 4])[0]

                out['trace'].append(t)
                out['shot_line'].append(i32be(BYTE_POS['shot_line']))
                out['shot_no'].append(i32be(BYTE_POS['shot_no']))
                out['recv_line'].append(i32be(BYTE_POS['recv_line']))
                out['recv_no'].append(i32be(BYTE_POS['recv_no']))
                out['shot_x'].append(i32be(BYTE_POS['shot_x']))
                out['shot_y'].append(i32be(BYTE_POS['shot_y']))
                out['rec_x'].append(i32be(BYTE_POS['rec_x']))
                out['rec_y'].append(i32be(BYTE_POS['rec_y']))
                out['shot_stake'].append(i32be(BYTE_POS['shot_stake']))
                out['recv_stake'].append(i32be(BYTE_POS['recv_stake']))
                out['cmp'].append(i32be(BYTE_POS['cmp']))
                out['cmp_line'].append(i32be(BYTE_POS['cmp_line']))
                out['offset'].append(i32be(BYTE_POS['offset']))
                ns = struct.unpack('>H', hdr[114:116])[0]
                f.seek(ns * bps, 1)
                t += 1

        # 对 fixed 模式的坐标也做 scalco 缩放（从 offset 字段推断 scalar）
        # 注：fixed 模式不使用 segyio 的 scalar，这里 scalar=1
        df = pd.DataFrame(out)
        for coord in ['shot_x', 'shot_y', 'rec_x', 'rec_y']:
            df[coord] = _scale_coords(df[coord].to_numpy(), np.ones(len(df), dtype=np.int64))
        return df

    else:
        raise ValueError("mode must be 'self_computed' or 'fixed'")


# ---------------------------------------------------------------------------
# Step 2: 坐标还原（已在 read_trace_headers 中完成，此处保留作独立接口）
# ---------------------------------------------------------------------------

def restore_coordinates(df: pd.DataFrame, scalar=None):
    """
    对坐标列应用 scalco 缩放。

    Parameters
    ----------
    df : pd.DataFrame
        包含 shot_x, shot_y, rec_x, rec_y 的 DataFrame。
    scalar : array-like, optional
        缩放因子数组，长度与 df 行数相同。
        若为 None，则不缩放。

    Returns
    -------
    df : pd.DataFrame
        缩放后的 DataFrame（copy）。
    """
    df = df.copy()
    for coord in ['shot_x', 'shot_y', 'rec_x', 'rec_y']:
        df[coord] = _scale_coords(df[coord].to_numpy(), scalar)
    return df


# ---------------------------------------------------------------------------
# Step 3: 计算 midpoint / half-offset
# ---------------------------------------------------------------------------

def compute_midpoint_offset(df: pd.DataFrame,
                             return_full_offset: bool = False):
    """
    从炮点、检波点坐标计算 midpoint 和 half-offset。

    Parameters
    ----------
    df : pd.DataFrame
        包含 sx/sy/rx/ry（或 shot_x/shot_y/rec_x/rec_y）列。
    return_full_offset : bool
        若为 True，同时计算全偏移量 ox/oy 和偏移距 magnitude/azimuth。

    Returns
    -------
    df : pd.DataFrame
        新增列：mx, my, hx, hy，
        可选：ox, oy, offset_mag, azimuth。
    """
    df = df.copy()

    sx = df['shot_x'].to_numpy() if 'shot_x' in df.columns else df['sx'].to_numpy()
    sy = df['shot_y'].to_numpy() if 'shot_y' in df.columns else df['sy'].to_numpy()
    rx = df['rec_x'].to_numpy()  if 'rec_x'  in df.columns else df['rx'].to_numpy()
    ry = df['rec_y'].to_numpy()  if 'rec_y'  in df.columns else df['ry'].to_numpy()

    df['mx'] = 0.5 * (sx + rx)
    df['my'] = 0.5 * (sy + ry)
    df['hx'] = 0.5 * (rx - sx)
    df['hy'] = 0.5 * (ry - sy)

    if return_full_offset:
        df['ox'] = rx - sx
        df['oy'] = ry - sy
        df['offset_mag'] = np.sqrt(df['ox']**2 + df['oy']**2)
        df['azimuth'] = np.arctan2(df['oy'], df['ox'])

    return df


# ---------------------------------------------------------------------------
# Step 4: OVT 分箱
# ---------------------------------------------------------------------------

def build_ovt_bins(df: pd.DataFrame,
                   mx_bin: float = None,
                   my_bin: float = None,
                   hx_bin: float = None,
                   hy_bin: float = None,
                   mx_origin: float = None,
                   my_origin: float = None,
                   hx_origin: float = None,
                   hy_origin: float = None):
    """
    对 mx, my, hx, hy 进行离散分箱，生成 OVT 整数索引。

    Parameters
    ----------
    df : pd.DataFrame
        包含 mx, my, hx, hy 列。
    mx_bin, my_bin, hx_bin, hy_bin : float
        各维 bin size（米）。若为 None，则自动从数据范围估计：
        - mx_bin / my_bin: 使用各自 range 的 1/100，最小 1.0
        - hx_bin / hy_bin: 使用各自 range 的 1/50，最小 0.5
    mx_origin, my_origin, hx_origin, hy_origin : float
        各维起始参考点。若为 None，则自动对齐到数据的最小值。

    Returns
    -------
    df : pd.DataFrame
        新增列：imx, imy, ihx, ihy（整数 bin 索引），
        以及可选的 bin_center 列。
    """
    df = df.copy()

    # 自动估计 bin size
    def _auto_bin(arr, n_div=100, min_val=1.0):
        r = arr.max() - arr.min()
        if r <= 0:
            return min_val
        return max(r / n_div, min_val)

    if mx_bin is None:
        mx_bin = _auto_bin(df['mx'].to_numpy())
    if my_bin is None:
        my_bin = _auto_bin(df['my'].to_numpy())
    if hx_bin is None:
        hx_bin = _auto_bin(df['hx'].to_numpy(), n_div=50, min_val=0.5)
    if hy_bin is None:
        hy_bin = _auto_bin(df['hy'].to_numpy(), n_div=50, min_val=0.5)

    # 自动对齐 origin 到最小值
    if mx_origin is None:
        mx_origin = df['mx'].min()
    if my_origin is None:
        my_origin = df['my'].min()
    if hx_origin is None:
        hx_origin = df['hx'].min()
    if hy_origin is None:
        hy_origin = df['hy'].min()

    df['imx'] = np.floor((df['mx'] - mx_origin) / mx_bin).astype(np.int64)
    df['imy'] = np.floor((df['my'] - my_origin) / my_bin).astype(np.int64)
    df['ihx'] = np.floor((df['hx'] - hx_origin) / hx_bin).astype(np.int64)
    df['ihy'] = np.floor((df['hy'] - hy_origin) / hy_bin).astype(np.int64)

    # 保留 bin center（可选，便于后续检查）
    df['mx_center'] = mx_origin + (df['imx'].to_numpy() + 0.5) * mx_bin
    df['my_center'] = my_origin + (df['imy'].to_numpy() + 0.5) * my_bin
    df['hx_center'] = hx_origin + (df['ihx'].to_numpy() + 0.5) * hx_bin
    df['hy_center'] = hy_origin + (df['ihy'].to_numpy() + 0.5) * hy_bin

    return df


# ---------------------------------------------------------------------------
# Step 5: 按 OVT cell 聚合 trace
# ---------------------------------------------------------------------------

def group_traces_by_ovt(df: pd.DataFrame,
                       fold_threshold: int = None):
    """
    将 trace 按 OVT cell (imx, imy, ihx, ihy) 聚合。

    Parameters
    ----------
    df : pd.DataFrame
        包含 imx, imy, ihx, ihy, trace 列的 DataFrame。
    fold_threshold : int, optional
        若指定，则只返回 fold >= fold_threshold 的 cell。

    Returns
    -------
    ovt_gathers : dict
        key = (imx, imy, ihx, ihy)
        value = np.array([trace_id, ...])，按 trace 列排序
    fold_dict : dict
        key = (imx, imy, ihx, ihy)
        value = fold（该 cell 的 trace 数）
    """
    # 按 OVT key 排序，确保输出顺序确定
    sort_keys = ['imx', 'imy', 'ihx', 'ihy', 'trace']
    df_sorted = df.sort_values(by=sort_keys).reset_index(drop=True)

    ovt_gathers = {}
    fold_dict = {}

    # 预分配：分组循环用向量化起步，groupby 保持简洁
    for key, group in df_sorted.groupby(['imx', 'imy', 'ihx', 'ihy'], sort=True):
        trace_ids = group['trace'].to_numpy(dtype=np.int64)
        fold = len(trace_ids)
        if fold_threshold is not None and fold < fold_threshold:
            continue
        ovt_gathers[key] = trace_ids
        fold_dict[key] = fold

    return ovt_gathers, fold_dict


# ---------------------------------------------------------------------------
# 汇总：一条主流程
# ---------------------------------------------------------------------------

def build_ovt_geometry_table(segy_path: Path,
                              mode: str = 'self_computed',
                              mx_bin: float = None,
                              my_bin: float = None,
                              hx_bin: float = None,
                              hy_bin: float = None,
                              return_full_offset: bool = False,
                              fold_threshold: int = None):
    """
    端到端：SEG-Y → trace 级几何表 + OVT gather 索引。

    Parameters
    ----------
    segy_path : Path
        SEG-Y 文件路径。
    mode : str
        'self_computed' 或 'fixed'，传至 read_trace_headers。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        分箱 bin size。
    return_full_offset : bool
        是否计算全偏移量字段。
    fold_threshold : int, optional
        OVT gather 最小 fold 阈值。

    Returns
    -------
    trace_table : pd.DataFrame
        每条 trace 一行，包含原始坐标、midpoint/offset、OVT 索引。
    ovt_gathers : dict
        key = (imx, imy, ihx, ihy)，value = trace_id array。
    fold_dict : dict
        key = (imx, imy, ihx, ihy)，value = fold。
    bin_params : dict
        实际使用的 bin 参数（便于复现）。
    """
    # 1. 读道头
    df = read_trace_headers(segy_path, mode=mode)

    # 2. 计算 midpoint / half-offset
    df = compute_midpoint_offset(df, return_full_offset=return_full_offset)

    # 3. 分箱
    df = build_ovt_bins(df,
                        mx_bin=mx_bin, my_bin=my_bin,
                        hx_bin=hx_bin, hy_bin=hy_bin)

    # 4. 聚合 OVT gather
    ovt_gathers, fold_dict = group_traces_by_ovt(df, fold_threshold=fold_threshold)

    def _infer_bin_size(center_values):
        unique_vals = np.unique(center_values.to_numpy(dtype=np.float64))
        if unique_vals.size <= 1:
            return 0.0
        diffs = np.diff(unique_vals)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return 0.0
        return float(diffs.min())

    # 5. 收集 bin 参数
    bin_params = {
        'mx_bin': float(mx_bin) if mx_bin is not None else _infer_bin_size(df['mx_center']),
        'my_bin': float(my_bin) if my_bin is not None else _infer_bin_size(df['my_center']),
        'hx_bin': float(hx_bin) if hx_bin is not None else _infer_bin_size(df['hx_center']),
        'hy_bin': float(hy_bin) if hy_bin is not None else _infer_bin_size(df['hy_center']),
        'mx_origin': float(df['mx'].min()),
        'my_origin': float(df['my'].min()),
        'hx_origin': float(df['hx'].min()),
        'hy_origin': float(df['hy'].min()),
    }

    return df, ovt_gathers, fold_dict, bin_params


# ---------------------------------------------------------------------------
# 输出工具
# ---------------------------------------------------------------------------

def export_trace_table(df: pd.DataFrame, output_path: str, fmt: str = 'csv'):
    """
    将 trace 级几何表导出为 CSV 或 Parquet。

    Parameters
    ----------
    df : pd.DataFrame
        build_ovt_geometry_table 返回的 trace 表。
    output_path : str
        输出文件路径。
    fmt : str
        'csv' 或 'parquet'。
    """
    if fmt == 'csv':
        df.to_csv(output_path, index=False)
    elif fmt == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def export_ovt_gathers(ovt_gathers: dict, fold_dict: dict, output_dir: str):
    """
    将 OVT gather 索引保存为 numpy 文件。

    生成：
        output_dir/ovt_gathers.npz   — npz，key=(imx,imy,ihx,ihy) 用元组字符串作 key
        output_dir/ovt_fold.npz      — fold 数字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 用 (imx,imy,ihx,ihy) 组成的复合数组 + 辅助索引表
    keys_arr = np.array(list(ovt_gathers.keys()), dtype=np.int32)  # (N, 4)
    traces_arr = np.concatenate(list(ovt_gathers.values()), axis=0)
    # 构建偏移量：每个 cell 的起始位置和长度
    n_cells = len(ovt_gathers)
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    for i, v in enumerate(ovt_gathers.values()):
        offsets[i + 1] = offsets[i] + len(v)

    np.savez(
        os.path.join(output_dir, 'ovt_gathers.npz'),
        keys=keys_arr,
        traces=traces_arr,
        offsets=offsets,
    )
    np.savez(
        os.path.join(output_dir, 'ovt_fold.npz'),
        keys=keys_arr,
        folds=np.array([fold_dict[k] for k in ovt_gathers.keys()], dtype=np.int32),
    )


def import_ovt_gathers(npz_path: str) -> dict:
    """
    从 export_ovt_gathers 生成的 npz 恢复 gather 字典。

    Returns
    -------
    ovt_gathers : dict
    """
    data = np.load(npz_path)
    keys = data['keys']  # (N, 4)
    traces = data['traces']
    offsets = data['offsets']
    result = {}
    for i, k in enumerate(keys):
        result[tuple(k)] = traces[offsets[i]:offsets[i + 1]]
    return result


# ---------------------------------------------------------------------------
# H5 衔接：将 OVT 字段追加到已有的 H5 文件
# 复用 Segy2H5.py 的 _scale_coords，不破坏现有 segy2h5 流程
# ---------------------------------------------------------------------------

def compute_ovt_fields(sx, sy, rx, ry,
                        mx_bin=None, my_bin=None,
                        hx_bin=None, hy_bin=None,
                        mx_origin=None, my_origin=None,
                        hx_origin=None, hy_origin=None):
    """
    从炮点、检波点坐标计算 OVT 域字段。

    Parameters
    ----------
    sx, sy, rx, ry : array-like
        炮点和检波点坐标（已缩放）。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        各维 bin size（米），None 则自动从数据范围估计。
    mx_origin, my_origin, hx_origin, hy_origin : float, optional
        各维起始参考点，None 则对齐到数据最小值。

    Returns
    -------
    dict with keys:
        mx, my, hx, hy             — continuous midpoint / half-offset
        imx, imy, ihx, ihy        — binned integer indices
        mx_center, my_center,
        hx_center, hy_center       — bin centers
        mx_bin, my_bin, hx_bin, hy_bin — actual bin sizes used
        mx_origin, my_origin, hx_origin, hy_origin — actual origins used
        fold                      — 每条 trace 所在 OVT cell 的 fold
    """
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    rx = np.asarray(rx, dtype=np.float64)
    ry = np.asarray(ry, dtype=np.float64)

    mx = 0.5 * (sx + rx)
    my = 0.5 * (sy + ry)
    hx = 0.5 * (rx - sx)
    hy = 0.5 * (ry - sy)

    def _auto_bin(arr, n_div=100, min_val=1.0):
        r = arr.max() - arr.min()
        return max(r / n_div, min_val) if r > 0 else float(min_val)

    if mx_bin is None:
        mx_bin = _auto_bin(mx)
    if my_bin is None:
        my_bin = _auto_bin(my)
    if hx_bin is None:
        hx_bin = _auto_bin(hx, n_div=50, min_val=0.5)
    if hy_bin is None:
        hy_bin = _auto_bin(hy, n_div=50, min_val=0.5)

    if mx_origin is None:
        mx_origin = mx.min()
    if my_origin is None:
        my_origin = my.min()
    if hx_origin is None:
        hx_origin = hx.min()
    if hy_origin is None:
        hy_origin = hy.min()

    imx = np.floor((mx - mx_origin) / mx_bin).astype(np.int64)
    imy = np.floor((my - my_origin) / my_bin).astype(np.int64)
    ihx = np.floor((hx - hx_origin) / hx_bin).astype(np.int64)
    ihy = np.floor((hy - hy_origin) / hy_bin).astype(np.int64)

    mx_center = mx_origin + (imx.astype(np.float64) + 0.5) * mx_bin
    my_center = my_origin + (imy.astype(np.float64) + 0.5) * my_bin
    hx_center = hx_origin + (ihx.astype(np.float64) + 0.5) * hx_bin
    hy_center = hy_origin + (ihy.astype(np.float64) + 0.5) * hy_bin

    # 计算 fold：每个 OVT cell 有多少条 trace
    keys = np.column_stack((imx, imy, ihx, ihy))
    unique_keys = np.unique(keys, axis=0)
    fold = np.zeros(len(imx), dtype=np.int32)
    for k in tqdm(unique_keys):
        mask = (keys[:, 0] == k[0]) & (keys[:, 1] == k[1]) & \
               (keys[:, 2] == k[2]) & (keys[:, 3] == k[3])
        fold[mask] = mask.sum()

    return {
        'mx': mx.astype(np.float32),
        'my': my.astype(np.float32),
        'hx': hx.astype(np.float32),
        'hy': hy.astype(np.float32),
        'imx': imx.astype(np.int32),
        'imy': imy.astype(np.int32),
        'ihx': ihx.astype(np.int32),
        'ihy': ihy.astype(np.int32),
        'mx_center': mx_center.astype(np.float32),
        'my_center': my_center.astype(np.float32),
        'hx_center': hx_center.astype(np.float32),
        'hy_center': hy_center.astype(np.float32),
        'mx_bin': float(mx_bin),
        'my_bin': float(my_bin),
        'hx_bin': float(hx_bin),
        'hy_bin': float(hy_bin),
        'mx_origin': float(mx_origin),
        'my_origin': float(my_origin),
        'hx_origin': float(hx_origin),
        'hy_origin': float(hy_origin),
        'fold': fold,
    }


def add_ovt_to_h5(h5_path: str,
                  group_name: str = None,
                  mx_bin: float = None,
                  my_bin: float = None,
                  hx_bin: float = None,
                  hy_bin: float = None,
                  overwrite: bool = False):
    """
    读取已有 H5 文件（由 Segy2H5.py 生成），追加 OVT 字段。

    H5 中的 sx/sy/rx/ry 已经是缩放后的坐标（Segy2H5.py 的 organize_traces
    已经处理过 scalco），此处直接使用，不再重复缩放。

    追加的字段（写入 group 属性）：
        mx, my, hx, hy                — continuous midpoint / half-offset
        imx, imy, ihx, ihy           — binned integer indices
        mx_center, my_center,
        hx_center, hy_center          — bin centers
        fold                          — 每条 trace 所在 OVT cell 的 fold
        bin 参数 (mx_bin, ...)        — 写入 group attrs，便于复现

    Parameters
    ----------
    h5_path : str
        H5 文件路径（由 segy2h5 生成）。
    group_name : str, optional
        H5 中的 group 名。若为 None，则取 H5 中第一个含 'data' 的 group。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        分箱 bin size（米）。None 则自动从数据范围估计。
    overwrite : bool
        若为 True，删除同名字段后重新写入；False（默认）则跳过已存在的字段。

    Returns
    -------
    bin_params : dict
        实际使用的 bin 参数。
    """
    import h5py as h5

    with h5.File(h5_path, 'r') as hf:
        # 自动找到含有 'data' 的 group
        if group_name is None:
            for key in hf.keys():
                node = hf[key]
                if hasattr(node, 'keys') and 'data' in node:
                    group_name = key
                    break
            if group_name is None:
                raise ValueError(f"H5 file {h5_path} 中未找到含有 'data' 的 group")
        g = hf[group_name]

        sx = g['sx'][:]
        sy = g['sy'][:]
        rx = g['rx'][:]
        ry = g['ry'][:]
        n_traces = g['data'].shape[0]

    # 计算 OVT 字段（H5 中的坐标已是缩放后坐标，直接用）
    ovt = compute_ovt_fields(
        sx, sy, rx, ry,
        mx_bin=mx_bin, my_bin=my_bin,
        hx_bin=hx_bin, hy_bin=hy_bin,
    )

    # 写入 H5
    with h5.File(h5_path, 'r+') as hf:
        g = hf[group_name]

        dataset_keys = (
            'mx', 'my', 'hx', 'hy',
            'imx', 'imy', 'ihx', 'ihy',
            'mx_center', 'my_center', 'hx_center', 'hy_center',
            'fold',
        )

        existing_keys = [key for key in dataset_keys if key in g]
        if existing_keys and not overwrite:
            raise RuntimeError(
                f"字段 {existing_keys} 已存在于 {h5_path}/{group_name}，"
                "请删除后重试或设置 overwrite=True"
            )

        if overwrite:
            for key in existing_keys:
                del g[key]

        for dkey in dataset_keys:
            g.create_dataset(dkey, data=ovt[dkey], compression='gzip')

        # bin 参数写入 group 属性
        for bkey in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                     'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
            g.attrs[bkey] = ovt[bkey]

    print(f"[add_ovt_to_h5] OVT fields written to {h5_path}/{group_name}")
    print(f"  Traces: {n_traces}, OVT cells: {len(np.unique(np.column_stack((ovt['imx'], ovt['imy'], ovt['ihx'], ovt['ihy'])), axis=0))}")
    print(f"  mx_bin={ovt['mx_bin']:.2f}, my_bin={ovt['my_bin']:.2f}, "
          f"hx_bin={ovt['hx_bin']:.2f}, hy_bin={ovt['hy_bin']:.2f}")

    return {
        'mx_bin': ovt['mx_bin'],
        'my_bin': ovt['my_bin'],
        'hx_bin': ovt['hx_bin'],
        'hy_bin': ovt['hy_bin'],
        'mx_origin': ovt['mx_origin'],
        'my_origin': ovt['my_origin'],
        'hx_origin': ovt['hx_origin'],
        'hy_origin': ovt['hy_origin'],
    }


# ---------------------------------------------------------------------------
# H5 衔接：直接从 SEG-Y 生成带 OVT 的 H5（两阶段，顺序调用）
# 阶段1: python Segy2H5.py       → 生成原始 H5（sx, sy, rx, ry, data...）
# 阶段2: python ovt_domain.py --h5 → 追加 OVT 字段到同一 H5
# ---------------------------------------------------------------------------

def segy_to_h5_with_ovt(segy_path: str,
                         h5_path: str,
                         group_name: str = None,
                         mode: str = 'self_computed',
                         mx_bin: float = None,
                         my_bin: float = None,
                         hx_bin: float = None,
                         hy_bin: float = None,
                         **seg2h5_kwargs):
    """
    顺序调用 Segy2H5 的 segy2h5，再追加 OVT 字段。

    等价于依次执行：
        segy2h5(h5_path, segy_path, ...)
        add_ovt_to_h5(h5_path, group_name, ...)

    Parameters
    ----------
    segy_path : str
        SEG-Y 文件路径。
    h5_path : str
        输出 H5 路径。
    group_name : str
        H5 group 名。
    mode : str
        'self_computed' 或 'fixed'，传给 Segy2H5.segy2h5。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        OVT 分箱 bin size。
    seg2h5_kwargs : dict
        其他传给 Segy2H5.segy2h5 的参数（如 sort_keys 等）。

    Returns
    -------
    bin_params : dict
        OVT bin 参数。
    """
    # 阶段1：调用 Segy2H5.py 的 segy2h5（动态 import 避免循环）
    import sys
    from pathlib import Path as P
    _pkg_root = P(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))

    from generate_py.Segy2H5 import segy2h5

    if group_name is None:
        group_name = P(segy_path).stem

    segy2h5(
        h5_file=h5_path,
        input_segy=segy_path,
        group_name=group_name,
        mode=mode,
        **seg2h5_kwargs,
    )
    print(f"[segy_to_h5_with_ovt] Stage 1 done: {h5_path}")

    # 阶段2：追加 OVT 字段
    bin_params = add_ovt_to_h5(
        h5_path=h5_path,
        group_name=group_name,
        mx_bin=mx_bin,
        my_bin=my_bin,
        hx_bin=hx_bin,
        hy_bin=hy_bin,
        overwrite=True,
    )
    print(f"[segy_to_h5_with_ovt] Stage 2 done: OVT fields added")

    return bin_params


def summarize_ovt_cells(df: pd.DataFrame):
    """
    汇总 midpoint、half-offset 和 4D OVT cell 的 fold 信息。

    Returns
    -------
    midpoint_cells : pd.DataFrame
        列包含 imx, imy, mx_center, my_center, fold
    offset_cells : pd.DataFrame
        列包含 ihx, ihy, hx_center, hy_center, fold
    ovt_cells : pd.DataFrame
        列包含 imx, imy, ihx, ihy, fold
    """
    midpoint_cells = (
        df.groupby(['imx', 'imy'], as_index=False)
          .agg(
              mx_center=('mx_center', 'first'),
              my_center=('my_center', 'first'),
              fold=('trace', 'size'),
          )
          .sort_values(['imx', 'imy'])
          .reset_index(drop=True)
    )
    offset_cells = (
        df.groupby(['ihx', 'ihy'], as_index=False)
          .agg(
              hx_center=('hx_center', 'first'),
              hy_center=('hy_center', 'first'),
              fold=('trace', 'size'),
          )
          .sort_values(['ihx', 'ihy'])
          .reset_index(drop=True)
    )
    ovt_cells = (
        df.groupby(['imx', 'imy', 'ihx', 'ihy'], as_index=False)
          .agg(fold=('trace', 'size'))
          .sort_values(['imx', 'imy', 'ihx', 'ihy'])
          .reset_index(drop=True)
    )
    return midpoint_cells, offset_cells, ovt_cells


def visualize_ovt_partition(df: pd.DataFrame,
                            output_dir: str,
                            max_points: int = 50000,
                            dpi: int = 200):
    """
    可视化 OVT 划分结果，输出总览图和 fold 统计图。

    Parameters
    ----------
    df : pd.DataFrame
        build_ovt_geometry_table 返回的 trace_table。
    output_dir : str
        图片输出目录。
    max_points : int
        原始散点最多采样点数，避免大数据绘图过慢。
    dpi : int
        图片分辨率。

    Returns
    -------
    plot_paths : dict
        生成图片的路径字典。
    """
    if df.empty:
        raise ValueError("trace_table is empty, nothing to visualize")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("visualize_ovt_partition 需要 matplotlib，请先安装该依赖") from exc

    os.makedirs(output_dir, exist_ok=True)

    midpoint_cells, offset_cells, ovt_cells = summarize_ovt_cells(df)

    if len(df) > max_points:
        sample_df = df.sample(n=max_points, random_state=42)
    else:
        sample_df = df

    overview_path = os.path.join(output_dir, 'ovt_partition_overview.png')
    fold_hist_path = os.path.join(output_dir, 'ovt_fold_histogram.png')

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.scatter(sample_df['mx'], sample_df['my'], s=3, alpha=0.25, c='tab:blue', edgecolors='none')
    ax1.set_title(f'Midpoint scatter (sample={len(sample_df)})')
    ax1.set_xlabel('mx')
    ax1.set_ylabel('my')
    ax1.grid(True, linestyle='--', alpha=0.3)

    midpoint_plot = ax2.scatter(
        midpoint_cells['mx_center'],
        midpoint_cells['my_center'],
        c=midpoint_cells['fold'],
        cmap='viridis',
        s=24,
        marker='s',
        edgecolors='none',
    )
    ax2.set_title(f'Midpoint bin fold (cells={len(midpoint_cells)})')
    ax2.set_xlabel('mx_center')
    ax2.set_ylabel('my_center')
    ax2.grid(True, linestyle='--', alpha=0.3)
    fig.colorbar(midpoint_plot, ax=ax2, shrink=0.85, label='fold')

    offset_plot = ax3.scatter(
        offset_cells['hx_center'],
        offset_cells['hy_center'],
        c=offset_cells['fold'],
        cmap='plasma',
        s=24,
        marker='s',
        edgecolors='none',
    )
    ax3.set_title(f'Half-offset bin fold (cells={len(offset_cells)})')
    ax3.set_xlabel('hx_center')
    ax3.set_ylabel('hy_center')
    ax3.grid(True, linestyle='--', alpha=0.3)
    fig.colorbar(offset_plot, ax=ax3, shrink=0.85, label='fold')

    ovt_folds = ovt_cells['fold'].to_numpy(dtype=np.int64)
    bins = min(60, max(10, int(np.sqrt(len(ovt_folds)))))
    ax4.hist(ovt_folds, bins=bins, color='tab:green', alpha=0.85, edgecolor='black', linewidth=0.4)
    ax4.set_title(f'OVT cell fold histogram (cells={len(ovt_cells)})')
    ax4.set_xlabel('fold')
    ax4.set_ylabel('cell count')
    ax4.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('OVT partition overview', fontsize=15)
    fig.tight_layout()
    fig.savefig(overview_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ovt_folds, bins=bins, color='tab:purple', alpha=0.85, edgecolor='black', linewidth=0.4)
    ax.axvline(float(ovt_folds.mean()), color='red', linestyle='--', linewidth=1.5,
               label=f'mean={ovt_folds.mean():.2f}')
    ax.axvline(float(np.median(ovt_folds)), color='orange', linestyle='--', linewidth=1.5,
               label=f'median={np.median(ovt_folds):.2f}')
    ax.set_title('OVT fold distribution')
    ax.set_xlabel('fold')
    ax.set_ylabel('cell count')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fold_hist_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {
        'overview': overview_path,
        'fold_histogram': fold_hist_path,
    }


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEG-Y → OVT domain transformation")
    parser.add_argument("segy_file", nargs="?", help="Input SEG-Y file path (ignored when --h5 is set)")
    parser.add_argument("--h5", dest="h5_path", default=None,
                        help="已有 H5 文件路径（由 Segy2H5.py 生成），直接追加 OVT 字段")
    parser.add_argument("--h5_group", dest="h5_group", default='1551',
                        help="H5 group 名（当 --h5 指定时使用）")
    parser.add_argument("--mode", choices=['self_computed', 'fixed'], default='self_computed',
                        help="Header read mode (default: self_computed)")
    parser.add_argument("--mx_bin", type=float, default=None, help="mx bin size (m)")
    parser.add_argument("--my_bin", type=float, default=None, help="my bin size (m)")
    parser.add_argument("--hx_bin", type=float, default=None, help="hx bin size (m)")
    parser.add_argument("--hy_bin", type=float, default=None, help="hy bin size (m)")
    parser.add_argument("--fold_threshold", type=int, default=1,
                        help="Minimum fold for OVT cell to be kept")
    parser.add_argument("--output_dir", default=f'./ovt_res',
                        help="Output directory (default: <segy_dir>/ovt_output)")
    parser.add_argument("--trace_fmt", choices=['csv', 'parquet'], default='csv',
                        help="Trace table output format")
    parser.add_argument("--plot",default=False,
                        help="Save OVT partition visualization figures")
    parser.add_argument("--plot_max_points", type=int, default=50000,
                        help="Maximum number of trace points used in raw scatter plot")
    args = parser.parse_args()

    # --h5 模式：从已有 H5 追加 OVT 字段
    if args.h5_path is not None:
        print(f"[ovt_domain] H5 mode: {args.h5_path}")
        bp = add_ovt_to_h5(
            h5_path=args.h5_path,
            group_name=args.h5_group,
            mx_bin=args.mx_bin,
            my_bin=args.my_bin,
            hx_bin=args.hx_bin,
            hy_bin=args.hy_bin,
            overwrite=True,
        )
        print(f"[ovt_domain] bin_params: {bp}")
    else:
        # 原有 SEG-Y 模式
        segy_path = Path(args.segy_file)
        output_dir = os.path.join(args.output_dir, segy_path.stem)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[ovt_domain] Reading: {segy_path}")
        print(f"  mode={args.mode}, mx_bin={args.mx_bin}, my_bin={args.my_bin}, "
              f"hx_bin={args.hx_bin}, hy_bin={args.hy_bin}")

        trace_table, ovt_gathers, fold_dict, bin_params = build_ovt_geometry_table(
            segy_path,
            mode=args.mode,
            mx_bin=args.mx_bin,
            my_bin=args.my_bin,
            hx_bin=args.hx_bin,
            hy_bin=args.hy_bin,
            fold_threshold=args.fold_threshold,
        )

        print(f"[ovt_domain] Trace table shape: {trace_table.shape}")
        print(f"[ovt_domain] OVT cells: {len(ovt_gathers)}, "
              f"total traces: {sum(fold_dict.values())}")
        if fold_dict:
            folds = np.array(list(fold_dict.values()))
            print(f"[ovt_domain] Fold stats — min: {folds.min()}, "
                  f"max: {folds.max()}, mean: {folds.mean():.2f}")

        cols_show = ['trace', 'shot_x', 'shot_y', 'rec_x', 'rec_y',
                     'mx', 'my', 'hx', 'hy', 'imx', 'imy', 'ihx', 'ihy']
        cols_exist = [c for c in cols_show if c in trace_table.columns]
        print(f"\n[ovt_domain] Trace table preview:\n{trace_table[cols_exist].head(10).to_string(index=False)}")

        export_trace_table(trace_table, os.path.join(output_dir, f'trace_geometry.{args.trace_fmt}'), fmt=args.trace_fmt)
        export_ovt_gathers(ovt_gathers, fold_dict, output_dir)

        with open(os.path.join(output_dir, 'bin_params.json'), 'w') as f:
            bp = {k: float(v) if v is not None else None for k, v in bin_params.items()}
            json.dump(bp, f, indent=2)

        if args.plot:
            plot_paths = visualize_ovt_partition(
                trace_table,
                output_dir=output_dir,
                max_points=args.plot_max_points,
            )
            print("\n[ovt_domain] Visualization saved:")
            for plot_name, plot_path in plot_paths.items():
                print(f"  - {plot_name}: {plot_path}")

        print(f"\n[ovt_domain] Outputs saved to: {output_dir}")
        print(f"  - trace_geometry.{args.trace_fmt}")
        print(f"  - ovt_gathers.npz")
        print(f"  - ovt_fold.npz")
        print(f"  - bin_params.json")
        if args.plot:
            print(f"  - ovt_partition_overview.png")
            print(f"  - ovt_fold_histogram.png")
