import argparse
import json
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm, trange
from h5py import File
import pickle
import dataset_config as dataset_config
import os
import sys
from ovt_masking import dispatch_ovt_mask, save_mask_results

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def get_unique_receivers(rx, ry, atol=1e-3):
    """
    从 (rx, ry) 提取唯一检波点，抗浮点误差。
    返回 unique_rx, unique_ry (各长度 n_unique)，以及 inverse (长度 N，inverse[i] 为第 i 道对应的唯一点下标)。
    """
    rx = np.asarray(rx, dtype=np.float64)
    ry = np.asarray(ry, dtype=np.float64)
    scale = 1.0 / max(atol, 1e-12)
    rx_int = np.round(rx * scale).astype(np.int64)
    ry_int = np.round(ry * scale).astype(np.int64)
    keys = np.column_stack((rx_int, ry_int))
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    unique_rx = unique_keys[:, 0].astype(np.float64) / scale
    unique_ry = unique_keys[:, 1].astype(np.float64) / scale
    return unique_rx, unique_ry, inverse


def _irregular_sampling(obs_rx, obs_ry, trace_to_obs, keep_ratio, seed, rx_all, ry_all):
    """不规则采样：保留部分检波点并在半间距内扰动（来自 data_irr.py 的逻辑）。"""
    n_obs = len(obs_rx)
    keep_num = int(round(n_obs * keep_ratio))
    rng = np.random.RandomState(seed)

    # 按原始顺序保留 keep_num 个点
    perm = rng.permutation(n_obs)
    kept_obs_indices = perm[:keep_num]
    obs_rx_kept = obs_rx[kept_obs_indices]
    obs_ry_kept = obs_ry[kept_obs_indices]

    # 计算扰动
    perturbation = np.zeros((keep_num, 2), dtype=np.float64)
    for i in trange(keep_num):
        # 估算局部检波点间距（取最近邻距离的一半）
        dists = np.sqrt((obs_rx_kept - obs_rx_kept[i]) ** 2 + (obs_ry_kept - obs_ry_kept[i]) ** 2)
        dists[i] = np.inf
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        spacing = max(min_dist / 2.0, 1.0)  # 半间距

        while True:
            p = np.round(rng.rand() * spacing - spacing / 2.0, 2)
            if p == 0:
                p = 0.5 * (2 * rng.randint(0, 2) - 1)
            new_x = obs_rx_kept[i] + p
            new_y = obs_ry_kept[i] + p
            if new_x >= 0 and new_y >= 0:
                perturbation[i, 0] = p
                perturbation[i, 1] = p
                break

    # 扰动后的坐标
    obs_rx_perturbed = obs_rx_kept + perturbation[:, 0]
    obs_ry_perturbed = obs_ry_kept + perturbation[:, 1]

    # 找到原始观测点中最接近扰动后位置的点
    mask_obs = np.zeros(n_obs, dtype=bool)
    for i in range(keep_num):
        dists = (obs_rx - obs_rx_perturbed[i]) ** 2 + (obs_ry - obs_ry_perturbed[i]) ** 2
        nearest_idx = np.argmin(dists)
        mask_obs[nearest_idx] = True

    return mask_obs


def _jitter_sampling(obs_rx, obs_ry, trace_to_obs, keep_ratio, seed):
    """抖动采样：k×k 网格，每单元保留 1 个点。"""
    np.random.seed(seed)
    n_obs = len(obs_rx)
    n_cells = max(1, int(keep_ratio * n_obs))
    k = max(1, math.ceil(math.sqrt(n_cells)))
    k = min(k, n_obs)

    rx_min, rx_max = obs_rx.min(), obs_rx.max()
    ry_min, ry_max = obs_ry.min(), obs_ry.max()
    span_rx = max(rx_max - rx_min, 1e-12)
    span_ry = max(ry_max - ry_min, 1e-12)
    cell_w = span_rx / k
    cell_h = span_ry / k

    cell_i = np.clip(np.floor((obs_rx - rx_min) / cell_w).astype(np.int32), 0, k - 1)
    cell_j = np.clip(np.floor((obs_ry - ry_min) / cell_h).astype(np.int32), 0, k - 1)
    cell_linear = cell_i * k + cell_j

    perm = np.random.permutation(n_obs)
    cell_perm = cell_linear[perm]
    _, first_in_perm = np.unique(cell_perm, return_index=True)
    kept_obs_indices = perm[first_in_perm]

    mask_obs = np.zeros(n_obs, dtype=bool)
    mask_obs[kept_obs_indices] = True
    return mask_obs


def _random_sampling(n_obs, keep_ratio, seed):
    """随机独立采样：每个点独立以 keep_ratio 概率保留。"""
    np.random.seed(seed)
    mask_obs = np.random.rand(n_obs) < keep_ratio
    return mask_obs


def _line_sampling(obs_rx, obs_ry, trace_to_obs, lines, keep_ratio, seed):
    """整线缺失：随机保留部分线。"""
    np.random.seed(seed)
    unique_lines = np.unique(lines)
    n_lines = len(unique_lines)
    n_keep = max(1, int(keep_ratio * n_lines))
    kept_lines = set(np.random.choice(unique_lines, n_keep, replace=False))
    mask_obs = np.array([lines[i] in kept_lines for i in range(len(lines))])
    return mask_obs


def receiver_sampling(sx, sy, rx, ry, keep_ratio=0.4, mode="jitter",
                      recv_line=None, shot_line=None, seed=None):
    """
    检波点域采样统一入口，支持多种缺失模式：
    - "irregular": 保留部分检波点并在半间距内扰动（来自 data_irr.py）
    - "jitter":    k×k 抖动网格采样，每单元保留 1 点
    - "random":    每点独立以 keep_ratio 概率保留
    - "line_recv": 整条检波线缺失，随机保留部分检波线
    - "line_shot": 整条炮线缺失，随机保留部分炮线
    - "mixed":     jitter + random 混合（先 jitter 再随机 dropout）
    返回 kept_trace_indices, mask_obs, rx_kept, ry_kept
    """
    rx = np.asarray(rx, dtype=np.float64)
    ry = np.asarray(ry, dtype=np.float64)

    # 观测系统取检波点坐标
    obs_rx, obs_ry, trace_to_obs = get_unique_receivers(rx, ry, atol=1e-3)
    n_obs = len(obs_rx)
    if n_obs == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=bool), np.array([]), np.array([])

    # 根据模式计算 mask_obs
    if mode == "irregular":
        mask_obs = _irregular_sampling(obs_rx, obs_ry, trace_to_obs, keep_ratio, seed, rx, ry)
    elif mode == "jitter":
        mask_obs = _jitter_sampling(obs_rx, obs_ry, trace_to_obs, keep_ratio, seed)
    elif mode == "random":
        mask_obs = _random_sampling(n_obs, keep_ratio, seed)
    elif mode == "line_recv":
        if recv_line is None:
            raise ValueError("line_recv mode requires recv_line array")
        _, _, line_to_obs = get_unique_receivers(rx, ry, atol=1e-3)
        mask_obs = _line_sampling(obs_rx, obs_ry, line_to_obs, recv_line, keep_ratio, seed)
    elif mode == "line_shot":
        if shot_line is None:
            raise ValueError("line_shot mode requires shot_line array")
        # 炮线需要用炮点坐标映射
        sx_arr = np.asarray(sx, dtype=np.float64)
        sy_arr = np.asarray(sy, dtype=np.float64)
        _, _, line_to_obs = get_unique_receivers(sx_arr, sy_arr, atol=1e-3)
        mask_obs = _line_sampling(obs_rx, obs_ry, line_to_obs, shot_line, keep_ratio, seed)
    elif mode == "mixed":
        # 先 jitter 再随机 dropout
        mask_obs = _jitter_sampling(obs_rx, obs_ry, trace_to_obs, 0.7, seed)
        n_kept = mask_obs.sum()
        target_keep = int(keep_ratio * n_obs)
        if target_keep < n_kept:
            drop_indices = np.where(mask_obs)[0]
            np.random.seed(seed + 1 if seed else None)
            keep_indices = np.random.choice(drop_indices, target_keep, replace=False)
            mask_obs = np.zeros(n_obs, dtype=bool)
            mask_obs[keep_indices] = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 映射到地震道
    kept_traces = mask_obs[trace_to_obs]
    kept_trace_indices = np.where(kept_traces)[0].astype(np.int64)
    rx_kept = rx[kept_trace_indices]
    ry_kept = ry[kept_trace_indices]
    return kept_trace_indices, mask_obs, rx_kept, ry_kept


def plot_jittered_receiver(rx_all, ry_all, rx_kept, ry_kept, keep_ratio=0.4, save_path=None):
    """
    双子图：图 A 原始观测系统 (rx, ry)，图 B 降采样后 (rx_kept, ry_kept)。
    坐标轴 'Receiver X', 'Receiver Y'。
    """
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(rx_all, ry_all, s=0.1, alpha=0.6, c="C0")
    axes[0].set_xlabel("Receiver X")
    axes[0].set_ylabel("Receiver Y")
    axes[0].set_title("Original receiver layout")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(rx_kept, ry_kept, s=0.1, alpha=0.6, c="C1")
    axes[1].set_xlabel("Receiver X")
    axes[1].set_ylabel("Receiver Y")
    axes[1].set_title(f"Jittered (keep_ratio={keep_ratio})")
    axes[1].set_aspect("equal", adjustable="box")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_one_shot_receivers(rx_shot, ry_shot, rx_shot_kept, ry_shot_kept, sx_val, sy_val, n_total, n_kept, save_path=None):
    """
    单炮检波点：双子图展示该炮集降采样前后检波点分布。
    图 A：该炮全部检波点 (rx, ry)；图 B：该炮保留的检波点。
    """
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(rx_shot, ry_shot, s=2, alpha=0.7, c="C0")
    axes[0].set_xlabel("Receiver X")
    axes[0].set_ylabel("Receiver Y")
    axes[0].set_title(f"One shot (sx={sx_val:.0f}, sy={sy_val:.0f}), N = {n_total} receivers")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(rx_shot_kept, ry_shot_kept, s=2, alpha=0.7, c="C1")
    axes[1].set_xlabel("Receiver X")
    axes[1].set_ylabel("Receiver Y")
    axes[1].set_title(f"Same shot after jitter, N = {n_kept} receivers")
    axes[1].set_aspect("equal", adjustable="box")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def generate_pos(info_f):
    sx = info_f['sx'][:]  # 形状: (N,)
    sy = info_f['sy'][:]  # 形状: (N,)
    rx = info_f['rx'][:]  # 形状: (N,)
    ry = info_f['ry'][:]  # 形状: (N,)
    pos_array = np.column_stack((sx, sy, rx, ry)).astype(np.float32)
    return pos_array


def _parse_float_list(text):
    if text is None or text == "":
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _load_json_arg(json_path=None, json_text=None):
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if json_text:
        return json.loads(json_text)
    return None


def _build_scope_config(args):
    if args.scope == "global":
        return {"type": "global"}

    scope = {
        "type": "local",
        "patch_ratio": args.patch_ratio,
    }
    if args.patch_width is not None:
        scope["width"] = args.patch_width
    if args.patch_height is not None:
        scope["height"] = args.patch_height
    if args.start_imx is not None and args.start_imy is not None:
        scope["start_imx"] = args.start_imx
        scope["start_imy"] = args.start_imy
    if args.center_imx is not None:
        scope["center_imx"] = args.center_imx
    if args.center_imy is not None:
        scope["center_imy"] = args.center_imy
    return scope


def _build_single_mask_config(args):
    config = {"scope": _build_scope_config(args)}
    if args.mode == "random_bin":
        config["missing_ratio"] = args.missing_ratio
    elif args.mode == "azimuth_sector":
        starts = _parse_float_list(args.sector_starts)
        widths = _parse_float_list(args.sector_widths)
        ends = _parse_float_list(args.sector_ends)
        sectors = []
        if widths:
            if len(widths) != len(starts):
                raise ValueError("sector_starts and sector_widths must have the same length")
            for start, width in zip(starts, widths):
                sectors.append({"start": start, "width": width})
        elif ends:
            if len(ends) != len(starts):
                raise ValueError("sector_starts and sector_ends must have the same length")
            for start, end in zip(starts, ends):
                sectors.append({"start": start, "end": end})
        else:
            raise ValueError("azimuth_sector requires sector_starts with sector_widths or sector_ends")
        config["sectors"] = sectors
        config["angle_unit"] = args.angle_unit
        config["reciprocal_pair"] = args.reciprocal_pair
    elif args.mode == "offset_truncation":
        config["truncation_mode"] = args.truncation_mode
        config["near_threshold"] = args.near_threshold
        config["far_threshold"] = args.far_threshold
        config["near_quantile"] = args.near_quantile
        config["far_quantile"] = args.far_quantile
        config["quantile_scope"] = args.truncation_quantile_scope
    elif args.mode == "midpoint_block":
        config["width"] = args.block_width
        config["height"] = args.block_height
        config["patch_ratio"] = args.patch_ratio
        if args.start_imx is not None:
            config["start_imx"] = args.start_imx
        if args.start_imy is not None:
            config["start_imy"] = args.start_imy
        if args.center_imx is not None:
            config["center_imx"] = args.center_imx
        if args.center_imy is not None:
            config["center_imy"] = args.center_imy
    else:
        raise ValueError(f"unsupported OVT mode: {args.mode}")
    return config


def _build_ovt_output_tag(args, result):
    if args.output_tag:
        return args.output_tag
    if args.mask_mode == "train":
        applied = "-".join(result["applied_modes"])
        return f"ovt_{args.mask_mode}_{applied}_seed{args.seed}"
    return f"ovt_{args.mode}_{args.scope}_seed{args.seed}"


def run_ovt_sample_cli(argv=None):
    parser = argparse.ArgumentParser(description="OVT domain masking sampler")
    parser.add_argument("--mode", choices=["random_bin", "azimuth_sector", "offset_truncation", "midpoint_block"],
                        default="random_bin")
    parser.add_argument("--mask_mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--source_type", choices=["h5", "table"], default="h5")
    parser.add_argument("--h5_path", default=info_h5)
    parser.add_argument("--key", default=list(segyPairs.keys())[0] if segyPairs else None)
    parser.add_argument("--table_path", default=None)
    parser.add_argument("--table_fmt", choices=["csv", "parquet"], default=None)
    parser.add_argument("--output_dir", default=target_dir)
    parser.add_argument("--output_tag", default=None)
    parser.add_argument("--save_fmt", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_keep_cells", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--scope", choices=["global", "local"], default="global")
    parser.add_argument("--patch_ratio", type=float, default=0.25)
    parser.add_argument("--patch_width", type=int, default=None)
    parser.add_argument("--patch_height", type=int, default=None)
    parser.add_argument("--start_imx", type=int, default=None)
    parser.add_argument("--start_imy", type=int, default=None)
    parser.add_argument("--center_imx", type=int, default=None)
    parser.add_argument("--center_imy", type=int, default=None)
    parser.add_argument("--missing_ratio", type=float, default=0.4)
    parser.add_argument("--sector_starts", default=None)
    parser.add_argument("--sector_widths", default=None)
    parser.add_argument("--sector_ends", default=None)
    parser.add_argument("--angle_unit", choices=["degree", "radian"], default="degree")
    parser.add_argument("--reciprocal_pair", action="store_true")
    parser.add_argument("--truncation_mode", choices=["remove_near", "remove_far", "keep_mid_only"],
                        default="remove_far")
    parser.add_argument("--truncation_quantile_scope", choices=["global_cell", "per_midpoint_cell"],
                        default="global_cell")
    parser.add_argument("--near_threshold", type=float, default=None)
    parser.add_argument("--far_threshold", type=float, default=None)
    parser.add_argument("--near_quantile", type=float, default=None)
    parser.add_argument("--far_quantile", type=float, default=None)
    parser.add_argument("--block_width", type=int, default=None)
    parser.add_argument("--block_height", type=int, default=None)
    parser.add_argument("--mixture_json", default=None,
                        help="Path to a JSON file describing train-mode mixture configs")
    parser.add_argument("--config_json", default=None,
                        help="Path to a JSON file overriding single-mode config")
    parser.add_argument("--config_text", default=None,
                        help="Raw JSON string overriding single-mode config")
    parser.add_argument("--mx_bin", type=float, default=None)
    parser.add_argument("--my_bin", type=float, default=None)
    parser.add_argument("--hx_bin", type=float, default=None)
    parser.add_argument("--hy_bin", type=float, default=None)

    args = parser.parse_args(argv)

    if args.source_type == "table" and args.table_path is None:
        raise ValueError("table source requires --table_path")

    base_config = _build_single_mask_config(args)
    json_override = _load_json_arg(json_path=args.config_json, json_text=args.config_text)
    if json_override:
        base_config.update(json_override)

    mixture = _load_json_arg(json_path=args.mixture_json)
    source = args.h5_path if args.source_type == "h5" else args.table_path

    result = dispatch_ovt_mask(
        source=source,
        source_type=args.source_type,
        mode=args.mode,
        mask_mode=args.mask_mode,
        config=base_config,
        mixture=mixture,
        group_name=args.key,
        table_fmt=args.table_fmt,
        mx_bin=args.mx_bin,
        my_bin=args.my_bin,
        hx_bin=args.hx_bin,
        hy_bin=args.hy_bin,
        seed=args.seed,
        min_keep_cells=args.min_keep_cells,
    )

    output_tag = _build_ovt_output_tag(args, result)
    output_paths = save_mask_results(
        result,
        output_dir=args.output_dir,
        output_tag=output_tag,
        save_fmt=args.save_fmt,
        save_preview=args.plot,
    )
    print(f"[ovt_sample] mode={args.mode}, mask_mode={args.mask_mode}, seed={args.seed}")
    print(
        f"[ovt_sample] support={result['stats']['total_support']}, "
        f"masked={result['stats']['masked_support']}, "
        f"ratio={result['stats']['actual_missing_ratio']:.4f}"
    )
    print(f"[ovt_sample] kept traces: {len(result['kept_trace_indices'])}, "
          f"masked traces: {len(result['masked_trace_indices'])}")
    for name, path in output_paths.items():
        print(f"[ovt_sample] {name}: {path}")
    return result

segyPairs = dataset_config.segyPairs
info_h5 = dataset_config.info_h5
target_dir = os.path.join(os.path.dirname(info_h5), f"{info_h5.split('/')[-1].split('.')[0]}_info")
os.makedirs(target_dir, exist_ok=True)

# 检波点域采样：python split_core.py jitter [key] [mode]；输出索引供 Segy2H5 的 traces_idx 使用
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        keep_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
        mode = sys.argv[3] if len(sys.argv) > 3 else "jitter"
        key = sys.argv[4] if len(sys.argv) > 4 else list(segyPairs.keys())[0]

        with File(info_h5, "r") as info_f:
            grp = info_f[key]
            pos = generate_pos(grp)
            recv_line = grp['recv_line'][:] if 'recv_line' in grp else None
            shot_line = grp['shot_line'][:] if 'shot_line' in grp else None

        sx, sy = pos[:, 0], pos[:, 1]
        rx, ry = pos[:, 2], pos[:, 3]
        print(f"sx: {sx.shape}, sy: {sy.shape}, rx: {rx.shape}, ry: {ry.shape}")
        print(f"Mode: {mode}, keep_ratio: {keep_ratio}")

        kept_trace_indices, mask_obs, rx_kept, ry_kept = receiver_sampling(
            sx, sy, rx, ry, keep_ratio=keep_ratio, mode=mode,
            recv_line=recv_line, shot_line=shot_line, seed=42
        )
        n_obs = len(mask_obs)
        out_path = os.path.join(target_dir, f"kept_trace_indices_{mode}_{keep_ratio}.npy")
        np.save(out_path, kept_trace_indices)
        print(f"Observation system: {n_obs} unique points; kept {np.sum(mask_obs)} -> {len(kept_trace_indices)} / {len(rx)} traces")
        print(f"Indices saved to {out_path}")

        plot_jittered_receiver(
            rx, ry, rx_kept, ry_kept,
            keep_ratio=keep_ratio,
            save_path=os.path.join(target_dir, f"kept_receiver_{mode}_{keep_ratio}.png"),
        )
        print(f"Figure saved to {os.path.join(target_dir, f'kept_receiver_{mode}_{keep_ratio}.png')}")

        # 随机取一个炮集，检查其检波点尺寸并做降采样前后可视化
        kept_traces = np.zeros(len(rx), dtype=bool)
        kept_traces[kept_trace_indices] = True
        shot_rx, shot_ry, trace_to_shot = get_unique_receivers(sx, sy, atol=1e-3)
        n_shots = len(shot_rx)
        shot_idx = np.random.randint(0, n_shots)
        shot_trace_idx = np.where(trace_to_shot == shot_idx)[0]
        rx_shot = rx[shot_trace_idx]
        ry_shot = ry[shot_trace_idx]
        kept_for_shot = kept_traces[shot_trace_idx]
        rx_shot_kept = rx_shot[kept_for_shot]
        ry_shot_kept = ry_shot[kept_for_shot]
        n_shot_total = len(shot_trace_idx)
        n_shot_kept = int(np.sum(kept_for_shot))
        print(f"Random shot [{shot_idx}] (sx={shot_rx[shot_idx]:.0f}, sy={shot_ry[shot_idx]:.0f}): {n_shot_total} receivers -> {n_shot_kept} after {mode}")
        plot_one_shot_receivers(
            rx_shot, ry_shot, rx_shot_kept, ry_shot_kept,
            sx_val=shot_rx[shot_idx], sy_val=shot_ry[shot_idx],
            n_total=n_shot_total, n_kept=n_shot_kept,
            save_path=os.path.join(target_dir, f"kept_receiver_{mode}_{keep_ratio}_one_shot.png"),
        )
        print(f"One-shot figure saved to {os.path.join(target_dir, f'kept_receiver_{mode}_{keep_ratio}_one_shot.png')}")
    elif len(sys.argv) > 1 and sys.argv[1] == "ovt_sample":
        run_ovt_sample_cli(sys.argv[2:])