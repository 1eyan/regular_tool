from pathlib import Path
import h5py as h5
import numpy as np
import segyio
import time
import dataset_config
import os
import struct
import pandas as pd


def get_traces_idx(cfg):
    return np.load(os.path.join(os.path.dirname(info_h5), f"{info_h5.split('/')[-1].split('.')[0]}_info", f"kept_trace_indices_{cfg['domain']}_{cfg['keep_ratio']}.npy"))

###mapping###defined by your own segy data####
BYTE_POS = {
    'shot_line': 17,  # 炮线 137位，4字节
    'shot_no':   25,  # 炮号 197位，4字节
    'recv_line': 61,  # 检波点线 171位，4字节
    'recv_stake': 65,  # 检波点桩号 175位，4字节
    'shot_x':    73,   # 炮点X坐标 73位，4字节
    'shot_y':    77,   # 炮点Y坐标 77位，4字节
    'rec_x':     81,   # 检波点X坐标 81位，4字节
    'rec_y':     85,   # 检波点Y坐标 85位，4字节
    'shot_stake':21,   # 炮桩号
    'recv_no':41,   # 检波点桩号
    'cmp':193,   # 炮桩号
    'cmp_line':189,   # 检波点桩号
    'offset':37,
}
BYTE_POS_ = {
    'shot_x':    73,   # 炮点X坐标 73位，4字节
    'shot_y':    77,   # 炮点Y坐标 77位，4字节
    'rec_x':     81,   # 检波点X坐标 81位，4字节
    'rec_y':     85,   # 检波点Y坐标 85位，4字节
}
##sort keys##
SORT_KEYS = ['recv_line', 'recv_stake', 'shot_line', 'shot_stake']
SORT_KEYS = ['shot_y', 'shot_x', 'rec_y', 'rec_x']
# === 辅助函数 ===
def _read_bin_header_format_and_ns(f):
    f.seek(3200, 0)
    binhdr = f.read(400)
    ns_from_bin = struct.unpack('>H', binhdr[20:22])[0]
    fmt_code    = struct.unpack('>H', binhdr[24:26])[0]
    return fmt_code, ns_from_bin

def _bps_from_fmt(fmt: int) -> int:
    if fmt in (1, 2, 5): return 4
    if fmt == 3: return 2
    if fmt == 8: return 1
    return 4

def _scale_coords(values, scalars):
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

# === sgy-->headers-->pandas dataframe ===
## 道头文件中如果自带炮线炮桩测线检波点
def read_headers_pure_python_fixed(path: Path):
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
                return struct.unpack('>i', hdr[j0:j0+4])[0]
            out['trace'].append(t)
            out['shot_line'].append(i32be(BYTE_POS['shot_line']))
            out['shot_no'].append(i32be(BYTE_POS['shot_no']))
            out['recv_line'].append(i32be(BYTE_POS['recv_line']))
            out['recv_no'].append(i32be(BYTE_POS['recv_no']))
            # 读取坐标信息
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
    return pd.DataFrame(out)

def read_headers_pure_self_computed(path: Path):
    out = {'trace': [],
    'shot_x': [],
    'shot_y': [],
    'rec_x': [],
    'rec_y': [],
    }
    for key in BYTE_POS_.keys():
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
                return struct.unpack('>i', hdr[j0:j0+4])[0]
            out['trace'].append(t)
            out['shot_x'].append(i32be(BYTE_POS_['shot_x']))
            out['shot_y'].append(i32be(BYTE_POS_['shot_y']))
            out['rec_x'].append(i32be(BYTE_POS_['rec_x']))
            out['rec_y'].append(i32be(BYTE_POS_['rec_y']))
            ns = struct.unpack('>H', hdr[114:116])[0]
            f.seek(ns * bps, 1)
            t += 1
    #print(len(out['trace']),len(out['shot_x']),len(out['shot_y']),len(out['rec_x']),len(out['rec_y']))
    return pd.DataFrame(out)

def organize_traces(input_segy, headers_df=None, sort_keys=SORT_KEYS,mode='self_computed'):
    """
    按 headers dataframe 的排序键重排地震道。
    - headers_df 为 None: 自动从 SEG-Y 读取道头并按 sort_keys 排序
    - headers_df 不为 None: 使用传入 dataframe（必须包含 trace 列），再按 sort_keys 排序
    返回:
      {
        'headers': 排序后的 dataframe,
        'data':    排序后的地震道矩阵,
        'sx','sy','rx','ry','inline','crossline','delta','t0': 对齐后的属性
      }
    """
    with segyio.open(input_segy, ignore_geometry=True) as f:
        data = f.trace.raw[:]
        scalar = np.abs(f.attributes(segyio.TraceField.SourceGroupScalar)[:].astype(np.float32))
        delta = f.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[:].astype(np.float32) / 1000.0
        t0 = f.attributes(segyio.TraceField.DelayRecordingTime)[:].astype(np.float32) / 1000.0
    
    headers = read_headers_pure_python_fixed(Path(input_segy)) if mode == 'fixed' else read_headers_pure_self_computed(Path(input_segy)) if headers_df is None else headers_df.copy()
    if 'trace' not in headers.columns:
        raise ValueError("headers_df 必须包含 'trace' 列")
    trace_idx_old = headers['trace'].to_numpy(dtype=np.intp)
    headers = headers.sort_values(by=sort_keys).reset_index(drop=True)
    trace_idx = headers['trace'].to_numpy(dtype=np.intp)
    if np.all(trace_idx == trace_idx_old):
        print('sort_headers is not needed')
    else:
        print('sort_headers is needed')
    n_traces = data.shape[0]
    if trace_idx.min() < 0 or trace_idx.max() >= n_traces:
        raise ValueError(f"trace 索引越界，合法范围是 [0, {n_traces - 1}]")

    scalar = scalar[trace_idx]
    scalar[scalar == 0] = 1.0
    if mode == 'fixed':
        out = {
            'data': data[trace_idx],
            'sx': headers['shot_x'].to_numpy(dtype=np.float32) / scalar,
            'sy': headers['shot_y'].to_numpy(dtype=np.float32) / scalar,
            'rx': headers['rec_x'].to_numpy(dtype=np.float32) / scalar,
            'ry': headers['rec_y'].to_numpy(dtype=np.float32) / scalar,
            'delta':delta[trace_idx],
            't0': t0[trace_idx],
            'shot_line': headers['shot_line'].to_numpy(dtype=np.int32),
            'shot_no': headers['shot_no'].to_numpy(dtype=np.int32),
            'recv_line': headers['recv_line'].to_numpy(dtype=np.int32),
            'recv_no': headers['recv_no'].to_numpy(dtype=np.int32),
            'shot_stake': headers['shot_stake'].to_numpy(dtype=np.int32),
            'recv_stake': headers['recv_stake'].to_numpy(dtype=np.int32),
            'cmp': headers['cmp'].to_numpy(dtype=np.int32),
            'cmp_line': headers['cmp_line'].to_numpy(dtype=np.int32),
            'offset': headers['offset'].to_numpy(dtype=np.int32),
            'trace_idx': headers['trace'].to_numpy(dtype=np.int32),
        }
    elif mode == 'self_computed':
        out = {
            'data': data[trace_idx],
            'sx': _scale_coords(headers['shot_x'].to_numpy(dtype=np.float32), scalar),
            'sy': _scale_coords(headers['shot_y'].to_numpy(dtype=np.float32), scalar),
            'rx': _scale_coords(headers['rec_x'].to_numpy(dtype=np.float32), scalar),
            'ry': _scale_coords(headers['rec_y'].to_numpy(dtype=np.float32), scalar),
            'sx_original': headers['shot_x'].to_numpy(dtype=np.float32),
            'sy_original': headers['shot_y'].to_numpy(dtype=np.float32),
            'rx_original': headers['rec_x'].to_numpy(dtype=np.float32),
            'ry_original': headers['rec_y'].to_numpy(dtype=np.float32),
            'delta':delta[trace_idx],
            't0': t0[trace_idx],
            'shot_line': pd.Series(np.rint(_scale_coords(headers['shot_y'].to_numpy(dtype=np.float32), scalar)), dtype="Int64"),
            'shot_no': pd.Series(np.rint(_scale_coords(headers['shot_x'].to_numpy(dtype=np.float32), scalar)), dtype="Int64"),
            'recv_line': pd.Series(np.rint(_scale_coords(headers['rec_y'].to_numpy(dtype=np.float32), scalar)), dtype="Int64"),
            'recv_no': pd.Series(np.rint(_scale_coords(headers['rec_x'].to_numpy(dtype=np.float32), scalar)), dtype="Int64"),
            'trace_idx': headers['trace'].to_numpy(dtype=np.int32),
        }
    else:
        raise ValueError(f"mode 必须为 'fixed' 或 'self_computed'")
    return out

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
        mx, my, hx, hy            — continuous midpoint / half-offset
        imx, imy, ihx, ihy       — binned integer indices
        mx_center, my_center, hx_center, hy_center — bin centers
        mx_bin, my_bin, hx_bin, hy_bin — actual bin sizes used
        mx_origin, my_origin, hx_origin, hy_origin — actual origins used
        fold                     — 每条 trace 所在 OVT cell 的 fold
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
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    fold = np.zeros(len(imx), dtype=np.int32)
    for i, k in enumerate(unique_keys):
        mask = (keys[:, 0] == k[0]) & (keys[:, 1] == k[1]) & \
               (keys[:, 2] == k[2]) & (keys[:, 3] == k[3])
        cnt = mask.sum()
        fold[mask] = cnt

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


def add_ovt_to_h5(h5_file, group_name='1551',
                  mx_bin=None, my_bin=None,
                  hx_bin=None, hy_bin=None):
    """
    读取已有 H5（由 segy2h5 生成），计算 OVT 字段并写入同一 H5 文件。

    Parameters
    ----------
    h5_file : str
        H5 文件路径。
    group_name : str
        H5 中的 group 名。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        分箱 bin size，None 则自动估计。
    """
    with h5.File(h5_file, 'r+') as h5f:
        g = h5f[group_name]
        sx = g['sx'][:]
        sy = g['sy'][:]
        rx = g['rx'][:]
        ry = g['ry'][:]

        ovt = compute_ovt_fields(
            sx, sy, rx, ry,
            mx_bin=mx_bin, my_bin=my_bin,
            hx_bin=hx_bin, hy_bin=hy_bin,
        )

        for key, val in ovt.items():
            if key not in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                           'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
                g.create_dataset(key, data=val, compression='gzip')

        # bin 参数作为 group 属性存储
        for key in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                    'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
            g.attrs[key] = ovt[key]

    print(f"[add_ovt_to_h5] OVT fields written to {h5_file}/{group_name}")


def segy2h5(h5_file, input_segy, group_name='1551', headers_df=None, sort_keys=SORT_KEYS,mode='self_computed',
            compute_ovt=False, mx_bin=None, my_bin=None, hx_bin=None, hy_bin=None):
    """
    单个 SEG-Y 落盘到 H5，按 sort_keys 组织地震道。

    Parameters
    ----------
    h5_file : str
        输出 H5 路径。
    input_segy : str
        输入 SEG-Y 路径。
    group_name : str
        H5 group 名。
    headers_df : pd.DataFrame, optional
        预读取的道头。
    sort_keys : list
        排序键。
    mode : str
        'self_computed' 或 'fixed'。
    compute_ovt : bool
        若为 True，计算并写入 OVT 字段（mx, my, hx, hy, imx, imy, ihx, ihy, fold）。
    mx_bin, my_bin, hx_bin, hy_bin : float, optional
        OVT 分箱 bin size。
    """
    block = organize_traces(input_segy, headers_df=headers_df, sort_keys=sort_keys,mode=mode)
    with h5.File(h5_file, 'w') as h5f:
        g = h5f.create_group(group_name)
        g.create_dataset('data', data=block['data'], compression='gzip')
        g.create_dataset('sx', data=block['sx'], compression='gzip')
        g.create_dataset('sy', data=block['sy'], compression='gzip')
        g.create_dataset('rx', data=block['rx'], compression='gzip')
        g.create_dataset('ry', data=block['ry'], compression='gzip')
        g.create_dataset('delta', data=block['delta'], compression='gzip')
        g.create_dataset('t0', data=block['t0'], compression='gzip')
        g.create_dataset('shot_line', data=block['shot_line'], compression='gzip')
        g.create_dataset('shot_no', data=block['shot_no'], compression='gzip')
        g.create_dataset('recv_line', data=block['recv_line'], compression='gzip')
        g.create_dataset('recv_no', data=block['recv_no'], compression='gzip')
        g.create_dataset('trace_idx', data=block['trace_idx'], compression='gzip')
        if mode == 'fixed':
            g.create_dataset('shot_stake', data=block['shot_stake'], compression='gzip')
            g.create_dataset('recv_stake', data=block['recv_stake'], compression='gzip')
            g.create_dataset('cmp', data=block['cmp'], compression='gzip')
            g.create_dataset('cmp_line', data=block['cmp_line'], compression='gzip')
            g.create_dataset('offset', data=block['offset'], compression='gzip')
        elif mode == 'self_computed':
            g.create_dataset('sx_original', data=block['sx_original'], compression='gzip')
            g.create_dataset('sy_original', data=block['sy_original'], compression='gzip')
            g.create_dataset('rx_original', data=block['rx_original'], compression='gzip')
            g.create_dataset('ry_original', data=block['ry_original'], compression='gzip')
        else:
            raise ValueError(f"mode 必须为 'fixed' 或 'self_computed'")

        if compute_ovt:
            sx = block['sx'] if isinstance(block['sx'], np.ndarray) else block['sx'].to_numpy()
            sy = block['sy'] if isinstance(block['sy'], np.ndarray) else block['sy'].to_numpy()
            rx = block['rx'] if isinstance(block['rx'], np.ndarray) else block['rx'].to_numpy()
            ry = block['ry'] if isinstance(block['ry'], np.ndarray) else block['ry'].to_numpy()
            ovt = compute_ovt_fields(
                sx, sy, rx, ry,
                mx_bin=mx_bin, my_bin=my_bin,
                hx_bin=hx_bin, hy_bin=hy_bin,
            )
            for key, val in ovt.items():
                if key not in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                               'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
                    g.create_dataset(key, data=val, compression='gzip')
            for key in ('mx_bin', 'my_bin', 'hx_bin', 'hy_bin',
                        'mx_origin', 'my_origin', 'hx_origin', 'hy_origin'):
                g.attrs[key] = ovt[key]



if __name__ == "__main__":
    mode = 'self_computed'
    info_h5 = dataset_config.info_h5
    segyPairs = dataset_config.segyPairs
    os.makedirs(os.path.dirname(info_h5), exist_ok=True)

    first_key = next(iter(segyPairs.keys()))
    input_segy = segyPairs[first_key][0]
    headers_df = read_headers_pure_python_fixed(Path(input_segy)) if mode == 'fixed' else read_headers_pure_self_computed(Path(input_segy))
    print(headers_df.head())
    s_time = time.time()
    segy2h5(
        h5_file=info_h5,
        input_segy=input_segy,
        headers_df=headers_df,
        group_name=first_key,
        mode='self_computed',
    )
    f_time = time.time()
    print(f"cost time: {f_time - s_time:.2f}")
