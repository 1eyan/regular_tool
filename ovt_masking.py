import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from h5py import File

try:
    from .ovt_domain import build_ovt_bins, compute_midpoint_offset, compute_ovt_fields
except ImportError:
    from ovt_domain import build_ovt_bins, compute_midpoint_offset, compute_ovt_fields

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


CELL_KEY_COLS = ["imx", "imy", "ihx", "ihy"]
MIDPOINT_KEY_COLS = ["imx", "imy"]


def _get_first_data_group(h5_path: str, group_name: Optional[str] = None):
    with File(h5_path, "r") as f:
        if group_name is not None:
            if group_name not in f:
                raise ValueError(f"group '{group_name}' not found in {h5_path}")
            return group_name
        for key in f:
            node = f[key]
            if hasattr(node, "keys") and "data" in node:
                return key
    raise ValueError(f"no group containing 'data' found in {h5_path}")


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ensure_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _resolve_trace_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "trace_index" in df.columns:
        df["trace_index"] = pd.to_numeric(df["trace_index"], errors="coerce").astype(np.int64)
        return df

    if "trace_idx" in df.columns:
        df["trace_index"] = pd.to_numeric(df["trace_idx"], errors="coerce").astype(np.int64)
        return df

    if "trace_id" in df.columns:
        df["trace_index"] = pd.to_numeric(df["trace_id"], errors="coerce").astype(np.int64)
        return df

    if "trace" in df.columns:
        df["trace_index"] = pd.to_numeric(df["trace"], errors="coerce").astype(np.int64)
        return df

    df["trace_index"] = np.arange(len(df), dtype=np.int64)
    return df


def _normalize_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _as_angle_rad(value: float, angle_unit: str) -> float:
    if angle_unit == "degree":
        return float(np.deg2rad(value))
    return float(value)


def _compute_half_offset_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hx" not in df.columns or "hy" not in df.columns:
        if {"shot_x", "shot_y", "rec_x", "rec_y"}.issubset(df.columns):
            df = compute_midpoint_offset(df, return_full_offset=False)
        elif {"sx", "sy", "rx", "ry"}.issubset(df.columns):
            df = compute_midpoint_offset(df, return_full_offset=False)
        else:
            raise ValueError("support table missing hx/hy and cannot infer from coordinates")

    hx = pd.to_numeric(df["hx"], errors="coerce").to_numpy(dtype=np.float64)
    hy = pd.to_numeric(df["hy"], errors="coerce").to_numpy(dtype=np.float64)
    zero_vec = (np.abs(hx) < 1e-12) & (np.abs(hy) < 1e-12)

    if "offset_mag" not in df.columns:
        df["offset_mag"] = np.sqrt(hx ** 2 + hy ** 2)
    if "azimuth" not in df.columns:
        azimuth = np.arctan2(hy, hx)
        azimuth[zero_vec] = 0.0
        df["azimuth"] = azimuth
    df["zero_offset_vector"] = zero_vec
    return df


def _load_trace_table(table_path: str, table_fmt: Optional[str] = None) -> pd.DataFrame:
    table_path = str(table_path)
    if table_fmt is None:
        suffix = Path(table_path).suffix.lower()
        if suffix == ".parquet":
            table_fmt = "parquet"
        else:
            table_fmt = "csv"

    if table_fmt == "parquet":
        df = pd.read_parquet(table_path)
    elif table_fmt == "csv":
        df = pd.read_csv(table_path)
    else:
        raise ValueError(f"unsupported table format: {table_fmt}")
    return df


def _load_h5_support_table(h5_path: str,
                           group_name: Optional[str] = None,
                           mx_bin: float = None,
                           my_bin: float = None,
                           hx_bin: float = None,
                           hy_bin: float = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    resolved_group = _get_first_data_group(h5_path, group_name=group_name)
    with File(h5_path, "r") as f:
        g = f[resolved_group]
        n_traces = g["data"].shape[0]
        df = pd.DataFrame({"trace_index": np.arange(n_traces, dtype=np.int64)})
        if "trace_idx" in g:
            df["trace_idx"] = g["trace_idx"][:].astype(np.int64)

        existing_cols = [
            "sx", "sy", "rx", "ry",
            "mx", "my", "hx", "hy",
            "imx", "imy", "ihx", "ihy",
            "mx_center", "my_center", "hx_center", "hy_center",
            "fold", "offset_mag", "azimuth",
        ]
        for key in existing_cols:
            if key in g:
                df[key] = g[key][:]

        attrs = {key: g.attrs[key] for key in g.attrs.keys()}

        missing_core = any(col not in df.columns for col in ["mx", "my", "hx", "hy", "imx", "imy", "ihx", "ihy"])
        if missing_core:
            if not {"sx", "sy", "rx", "ry"}.issubset(df.columns):
                raise ValueError(f"H5 group {resolved_group} missing OVT fields and sx/sy/rx/ry")
            ovt = compute_ovt_fields(
                df["sx"].to_numpy(dtype=np.float64),
                df["sy"].to_numpy(dtype=np.float64),
                df["rx"].to_numpy(dtype=np.float64),
                df["ry"].to_numpy(dtype=np.float64),
                mx_bin=mx_bin if mx_bin is not None else attrs.get("mx_bin"),
                my_bin=my_bin if my_bin is not None else attrs.get("my_bin"),
                hx_bin=hx_bin if hx_bin is not None else attrs.get("hx_bin"),
                hy_bin=hy_bin if hy_bin is not None else attrs.get("hy_bin"),
                mx_origin=attrs.get("mx_origin"),
                my_origin=attrs.get("my_origin"),
                hx_origin=attrs.get("hx_origin"),
                hy_origin=attrs.get("hy_origin"),
            )
            for key, value in ovt.items():
                if isinstance(value, np.ndarray):
                    df[key] = value
                else:
                    attrs[key] = value

    return df, {
        "source_type": "h5",
        "h5_path": h5_path,
        "group_name": resolved_group,
        "attrs": attrs,
    }


def _prepare_table_support(df: pd.DataFrame,
                           mx_bin: float = None,
                           my_bin: float = None,
                           hx_bin: float = None,
                           hy_bin: float = None) -> pd.DataFrame:
    df = _resolve_trace_index(df)
    df = _compute_half_offset_metrics(df)

    if "mx" not in df.columns or "my" not in df.columns:
        if {"shot_x", "shot_y", "rec_x", "rec_y"}.issubset(df.columns):
            df = compute_midpoint_offset(df, return_full_offset=False)
        elif {"sx", "sy", "rx", "ry"}.issubset(df.columns):
            df = compute_midpoint_offset(df, return_full_offset=False)
        else:
            raise ValueError("support table missing mx/my and cannot infer from coordinates")

    if not set(CELL_KEY_COLS).issubset(df.columns):
        df = build_ovt_bins(
            df,
            mx_bin=mx_bin,
            my_bin=my_bin,
            hx_bin=hx_bin,
            hy_bin=hy_bin,
        )

    return df


def build_support_index(source: Any,
                        source_type: str = "h5",
                        group_name: Optional[str] = None,
                        table_fmt: Optional[str] = None,
                        mx_bin: float = None,
                        my_bin: float = None,
                        hx_bin: float = None,
                        hy_bin: float = None) -> Dict[str, Any]:
    """
    构建 OVT support 索引，只基于已有 support，不扩展理论网格。
    """
    if source_type == "h5":
        trace_df, source_meta = _load_h5_support_table(
            h5_path=str(source),
            group_name=group_name,
            mx_bin=mx_bin,
            my_bin=my_bin,
            hx_bin=hx_bin,
            hy_bin=hy_bin,
        )
    elif source_type == "table":
        trace_df = _load_trace_table(str(source), table_fmt=table_fmt)
        trace_df = _prepare_table_support(
            trace_df,
            mx_bin=mx_bin,
            my_bin=my_bin,
            hx_bin=hx_bin,
            hy_bin=hy_bin,
        )
        source_meta = {
            "source_type": "table",
            "table_path": str(source),
            "table_fmt": table_fmt or Path(str(source)).suffix.lower().lstrip("."),
        }
    elif source_type == "dataframe":
        trace_df = _prepare_table_support(
            source.copy(),
            mx_bin=mx_bin,
            my_bin=my_bin,
            hx_bin=hx_bin,
            hy_bin=hy_bin,
        )
        source_meta = {"source_type": "dataframe"}
    else:
        raise ValueError(f"unsupported source_type: {source_type}")

    trace_df = _resolve_trace_index(trace_df)
    trace_df = _compute_half_offset_metrics(trace_df)
    trace_df = _ensure_numeric_columns(
        trace_df,
        [
            "trace_index", "trace_idx",
            "mx", "my", "hx", "hy",
            "imx", "imy", "ihx", "ihy",
            "mx_center", "my_center", "hx_center", "hy_center",
            "fold", "offset_mag", "azimuth",
        ],
    )

    for key in CELL_KEY_COLS:
        trace_df[key] = trace_df[key].astype(np.int64)

    if "mx_center" not in trace_df.columns:
        trace_df["mx_center"] = trace_df["mx"]
    if "my_center" not in trace_df.columns:
        trace_df["my_center"] = trace_df["my"]
    if "hx_center" not in trace_df.columns:
        trace_df["hx_center"] = trace_df["hx"]
    if "hy_center" not in trace_df.columns:
        trace_df["hy_center"] = trace_df["hy"]

    cell_df = (
        trace_df.groupby(CELL_KEY_COLS, as_index=False)
        .agg(
            mx_center=("mx_center", "first"),
            my_center=("my_center", "first"),
            hx_center=("hx_center", "first"),
            hy_center=("hy_center", "first"),
            offset_mag=("offset_mag", "mean"),
            trace_count=("trace_index", "size"),
            zero_offset_vector=("zero_offset_vector", "all"),
        )
        .sort_values(CELL_KEY_COLS)
        .reset_index(drop=True)
    )
    cell_df["ovt_cell_id"] = np.arange(len(cell_df), dtype=np.int64)
    cell_df["azimuth"] = np.arctan2(
        cell_df["hy_center"].to_numpy(dtype=np.float64),
        cell_df["hx_center"].to_numpy(dtype=np.float64),
    )
    cell_df.loc[cell_df["zero_offset_vector"], "azimuth"] = 0.0

    midpoint_df = (
        cell_df[MIDPOINT_KEY_COLS + ["mx_center", "my_center"]]
        .drop_duplicates()
        .sort_values(MIDPOINT_KEY_COLS)
        .reset_index(drop=True)
    )

    if "ovt_cell_id" in trace_df.columns:
        trace_df = trace_df.drop(columns=["ovt_cell_id"])

    trace_df = trace_df.merge(
        cell_df[CELL_KEY_COLS + ["ovt_cell_id"]],
        on=CELL_KEY_COLS,
        how="left",
    )
    trace_df["ovt_cell_id"] = trace_df["ovt_cell_id"].astype(np.int64)

    return {
        "trace_df": trace_df,
        "cell_df": cell_df,
        "midpoint_df": midpoint_df,
        "source_meta": source_meta,
    }


def select_midpoint_scope(cell_df: pd.DataFrame,
                          midpoint_df: pd.DataFrame,
                          scope_cfg: Optional[Dict[str, Any]],
                          rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
    scope_cfg = dict(scope_cfg or {})
    scope_type = scope_cfg.get("type", "global")

    if scope_type == "global" or midpoint_df.empty:
        return np.ones(len(cell_df), dtype=bool), {
            "scope_type": "global",
            "scope_midpoint_count": int(len(midpoint_df)),
            "eligible_cell_count": int(len(cell_df)),
        }

    if scope_type != "local":
        raise ValueError(f"unsupported midpoint scope type: {scope_type}")

    imx_vals = midpoint_df["imx"].to_numpy(dtype=np.int64)
    imy_vals = midpoint_df["imy"].to_numpy(dtype=np.int64)
    span_imx = int(imx_vals.max() - imx_vals.min() + 1)
    span_imy = int(imy_vals.max() - imy_vals.min() + 1)

    patch_ratio = float(scope_cfg.get("patch_ratio", 0.25))
    width = scope_cfg.get("width")
    height = scope_cfg.get("height")
    width = int(width) if width is not None else max(1, int(math.ceil(span_imx * patch_ratio)))
    height = int(height) if height is not None else max(1, int(math.ceil(span_imy * patch_ratio)))

    if "start_imx" in scope_cfg and "start_imy" in scope_cfg:
        start_imx = int(scope_cfg["start_imx"])
        start_imy = int(scope_cfg["start_imy"])
    else:
        anchor_row = midpoint_df.iloc[int(rng.randint(0, len(midpoint_df)))]
        center_imx = int(scope_cfg.get("center_imx", anchor_row["imx"]))
        center_imy = int(scope_cfg.get("center_imy", anchor_row["imy"]))
        start_imx = center_imx - width // 2
        start_imy = center_imy - height // 2

    end_imx = start_imx + width - 1
    end_imy = start_imy + height - 1

    midpoint_mask = (
        (midpoint_df["imx"] >= start_imx) & (midpoint_df["imx"] <= end_imx) &
        (midpoint_df["imy"] >= start_imy) & (midpoint_df["imy"] <= end_imy)
    )
    selected_midpoints = midpoint_df.loc[midpoint_mask, MIDPOINT_KEY_COLS]
    if selected_midpoints.empty:
        return np.zeros(len(cell_df), dtype=bool), {
            "scope_type": "local",
            "scope_midpoint_count": 0,
            "eligible_cell_count": 0,
            "start_imx": int(start_imx),
            "start_imy": int(start_imy),
            "end_imx": int(end_imx),
            "end_imy": int(end_imy),
        }

    eligible_key_set = {
        (int(row.imx), int(row.imy))
        for row in selected_midpoints.itertuples(index=False)
    }
    eligible_cell_mask = np.array(
        [(int(imx), int(imy)) in eligible_key_set for imx, imy in zip(cell_df["imx"], cell_df["imy"])],
        dtype=bool,
    )
    return eligible_cell_mask, {
        "scope_type": "local",
        "scope_midpoint_count": int(len(selected_midpoints)),
        "eligible_cell_count": int(eligible_cell_mask.sum()),
        "start_imx": int(start_imx),
        "start_imy": int(start_imy),
        "end_imx": int(end_imx),
        "end_imy": int(end_imy),
        "width": int(width),
        "height": int(height),
    }


def _safe_choice(mask: np.ndarray,
                 ratio: float,
                 rng: np.random.RandomState) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return np.zeros_like(mask, dtype=bool)
    n_drop = int(round(len(indices) * float(ratio)))
    n_drop = max(0, min(n_drop, len(indices)))
    if n_drop == 0:
        return np.zeros_like(mask, dtype=bool)
    selected = rng.choice(indices, size=n_drop, replace=False)
    out = np.zeros_like(mask, dtype=bool)
    out[selected] = True
    return out


def _canonicalize_sectors(sectors: Iterable[Dict[str, Any]],
                          angle_unit: str = "degree") -> List[Tuple[float, float]]:
    sector_ranges: List[Tuple[float, float]] = []
    for sector in sectors:
        if "start" in sector and "end" in sector:
            start = _as_angle_rad(float(sector["start"]), angle_unit)
            end = _as_angle_rad(float(sector["end"]), angle_unit)
        elif "center" in sector and "width" in sector:
            center = _as_angle_rad(float(sector["center"]), angle_unit)
            width = _as_angle_rad(float(sector["width"]), angle_unit)
            start = center - 0.5 * width
            end = center + 0.5 * width
        elif "start" in sector and "width" in sector:
            start = _as_angle_rad(float(sector["start"]), angle_unit)
            end = start + _as_angle_rad(float(sector["width"]), angle_unit)
        else:
            raise ValueError("each sector must define (start,end) or (center,width)")
        sector_ranges.append((_normalize_angle_rad(np.array([start]))[0], _normalize_angle_rad(np.array([end]))[0]))
    return sector_ranges


def _angle_in_sector(angle: np.ndarray, start: float, end: float) -> np.ndarray:
    angle = _normalize_angle_rad(angle)
    start = _normalize_angle_rad(np.array([start]))[0]
    end = _normalize_angle_rad(np.array([end]))[0]
    if np.isclose(start, end):
        return np.ones_like(angle, dtype=bool)
    if start <= end:
        return (angle >= start) & (angle <= end)
    return (angle >= start) | (angle <= end)


def generate_random_bin_mask(cell_df: pd.DataFrame,
                             eligible_mask: np.ndarray,
                             missing_ratio: float,
                             rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
    drop_mask = _safe_choice(eligible_mask, missing_ratio, rng)
    return drop_mask, {
        "missing_ratio": float(missing_ratio),
        "eligible_cell_count": int(eligible_mask.sum()),
        "masked_cell_count": int(drop_mask.sum()),
    }


def generate_azimuth_sector_mask(cell_df: pd.DataFrame,
                                 eligible_mask: np.ndarray,
                                 sectors: Sequence[Dict[str, Any]],
                                 angle_unit: str = "degree",
                                 reciprocal_pair: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(sectors) == 0:
        raise ValueError("azimuth_sector requires at least one sector")

    azimuth = cell_df["azimuth"].to_numpy(dtype=np.float64)
    zero_vec = cell_df["zero_offset_vector"].to_numpy(dtype=bool)
    sector_ranges = _canonicalize_sectors(sectors, angle_unit=angle_unit)

    hit_mask = np.zeros(len(cell_df), dtype=bool)
    for start, end in sector_ranges:
        hit_mask |= _angle_in_sector(azimuth, start, end)
        if reciprocal_pair:
            hit_mask |= _angle_in_sector(azimuth, start + np.pi, end + np.pi)

    hit_mask &= ~zero_vec
    drop_mask = eligible_mask & hit_mask
    return drop_mask, {
        "eligible_cell_count": int(eligible_mask.sum()),
        "masked_cell_count": int(drop_mask.sum()),
        "sector_count": int(len(sector_ranges)),
        "reciprocal_pair": bool(reciprocal_pair),
    }


def generate_offset_truncation_mask(cell_df: pd.DataFrame,
                                    eligible_mask: np.ndarray,
                                    truncation_mode: str,
                                    near_threshold: float = None,
                                    far_threshold: float = None,
                                    near_quantile: float = None,
                                    far_quantile: float = None,
                                    quantile_scope: str = "global_cell") -> Tuple[np.ndarray, Dict[str, Any]]:
    h = cell_df["offset_mag"].to_numpy(dtype=np.float64)
    eligible_h = h[eligible_mask]

    if eligible_h.size == 0:
        return np.zeros(len(cell_df), dtype=bool), {
            "eligible_cell_count": 0,
            "masked_cell_count": 0,
            "truncation_mode": truncation_mode,
            "quantile_scope": quantile_scope,
        }

    def _build_condition(values: np.ndarray,
                         near_val: float,
                         far_val: float) -> np.ndarray:
        if truncation_mode == "remove_near":
            if near_val is None:
                raise ValueError("remove_near requires near_threshold or near_quantile")
            return values <= float(near_val)
        if truncation_mode == "remove_far":
            if far_val is None:
                raise ValueError("remove_far requires far_threshold or far_quantile")
            return values >= float(far_val)
        if truncation_mode == "keep_mid_only":
            if near_val is None or far_val is None:
                raise ValueError("keep_mid_only requires both near_threshold and far_threshold")
            return (values < float(near_val)) | (values > float(far_val))
        raise ValueError(f"unsupported truncation_mode: {truncation_mode}")

    cond = np.zeros(len(cell_df), dtype=bool)
    midpoint_group_count = 0

    if quantile_scope == "global_cell":
        if near_threshold is None and near_quantile is not None:
            near_threshold = float(np.quantile(eligible_h, near_quantile))
        if far_threshold is None and far_quantile is not None:
            far_threshold = float(np.quantile(eligible_h, far_quantile))
        cond = _build_condition(h, near_threshold, far_threshold)
    elif quantile_scope == "per_midpoint_cell":
        midpoint_keys = cell_df[["imx", "imy"]].to_numpy(dtype=np.int64)
        eligible_indices = np.flatnonzero(eligible_mask)
        midpoint_to_indices: Dict[Tuple[int, int], List[int]] = {}
        for idx in eligible_indices:
            key = (int(midpoint_keys[idx, 0]), int(midpoint_keys[idx, 1]))
            midpoint_to_indices.setdefault(key, []).append(int(idx))

        midpoint_group_count = len(midpoint_to_indices)
        for indices in midpoint_to_indices.values():
            group_idx = np.asarray(indices, dtype=np.int64)
            group_h = h[group_idx]

            near_val = near_threshold
            far_val = far_threshold
            if near_val is None and near_quantile is not None:
                near_val = float(np.quantile(group_h, near_quantile))
            if far_val is None and far_quantile is not None:
                far_val = float(np.quantile(group_h, far_quantile))

            cond[group_idx] = _build_condition(group_h, near_val, far_val)
    else:
        raise ValueError(f"unsupported quantile_scope: {quantile_scope}")

    drop_mask = eligible_mask & cond
    return drop_mask, {
        "eligible_cell_count": int(eligible_mask.sum()),
        "masked_cell_count": int(drop_mask.sum()),
        "truncation_mode": truncation_mode,
        "quantile_scope": quantile_scope,
        "midpoint_group_count": int(midpoint_group_count),
        "near_threshold": None if near_threshold is None else float(near_threshold),
        "far_threshold": None if far_threshold is None else float(far_threshold),
    }


def _apply_subcondition(cell_df: pd.DataFrame,
                        base_mask: np.ndarray,
                        subcondition: Optional[Dict[str, Any]]) -> np.ndarray:
    if not subcondition:
        return base_mask

    sub_type = subcondition.get("type")
    if sub_type == "azimuth_sector":
        extra_mask, _ = generate_azimuth_sector_mask(
            cell_df,
            eligible_mask=base_mask,
            sectors=subcondition.get("sectors", []),
            angle_unit=subcondition.get("angle_unit", "degree"),
            reciprocal_pair=subcondition.get("reciprocal_pair", False),
        )
        return extra_mask

    if sub_type == "offset_truncation":
        extra_mask, _ = generate_offset_truncation_mask(
            cell_df,
            eligible_mask=base_mask,
            truncation_mode=subcondition["truncation_mode"],
            near_threshold=subcondition.get("near_threshold"),
            far_threshold=subcondition.get("far_threshold"),
            near_quantile=subcondition.get("near_quantile"),
            far_quantile=subcondition.get("far_quantile"),
            quantile_scope=subcondition.get("quantile_scope", "global_cell"),
        )
        return extra_mask

    raise ValueError(f"unsupported midpoint_block subcondition type: {sub_type}")


def generate_midpoint_block_mask(cell_df: pd.DataFrame,
                                 midpoint_df: pd.DataFrame,
                                 eligible_mask: np.ndarray,
                                 rng: np.random.RandomState,
                                 width: int = None,
                                 height: int = None,
                                 patch_ratio: float = 0.25,
                                 start_imx: int = None,
                                 start_imy: int = None,
                                 center_imx: int = None,
                                 center_imy: int = None,
                                 subcondition: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    scope_cfg: Dict[str, Any] = {
        "type": "local",
        "width": width,
        "height": height,
        "patch_ratio": patch_ratio,
    }
    if start_imx is not None and start_imy is not None:
        scope_cfg["start_imx"] = start_imx
        scope_cfg["start_imy"] = start_imy
    if center_imx is not None:
        scope_cfg["center_imx"] = center_imx
    if center_imy is not None:
        scope_cfg["center_imy"] = center_imy

    block_mask, block_info = select_midpoint_scope(cell_df, midpoint_df, scope_cfg, rng)
    drop_mask = eligible_mask & block_mask
    drop_mask = _apply_subcondition(cell_df, drop_mask, subcondition=subcondition)
    block_info["masked_cell_count"] = int(drop_mask.sum())
    return drop_mask, block_info


def combine_masks(mask_items: Sequence[np.ndarray],
                  min_keep_cells: int = 1,
                  rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    if len(mask_items) == 0:
        raise ValueError("combine_masks requires at least one mask")
    combined = np.zeros_like(mask_items[0], dtype=bool)
    for item in mask_items:
        combined |= np.asarray(item, dtype=bool)

    total = len(combined)
    keep_count = total - int(combined.sum())
    if keep_count >= min_keep_cells or total == 0:
        return combined

    if rng is None:
        rng = np.random.RandomState(0)
    masked_indices = np.flatnonzero(combined)
    n_restore = min(len(masked_indices), min_keep_cells - keep_count)
    if n_restore > 0:
        restore_idx = rng.choice(masked_indices, size=n_restore, replace=False)
        combined[restore_idx] = False
    return combined


def apply_mask_to_support(trace_df: pd.DataFrame,
                          cell_df: pd.DataFrame,
                          drop_mask: np.ndarray) -> Dict[str, Any]:
    cell_mask_table = cell_df.copy()
    cell_mask_table["drop"] = np.asarray(drop_mask, dtype=bool)
    cell_mask_table["keep"] = ~cell_mask_table["drop"]

    trace_mask_table = trace_df.merge(
        cell_mask_table[CELL_KEY_COLS + ["ovt_cell_id", "drop", "keep"]],
        on=CELL_KEY_COLS,
        how="left",
    )
    trace_mask_table["drop"] = trace_mask_table["drop"].fillna(False).astype(bool)
    trace_mask_table["keep"] = trace_mask_table["keep"].fillna(True).astype(bool)

    kept_trace_indices = trace_mask_table.loc[trace_mask_table["keep"], "trace_index"].to_numpy(dtype=np.int64)
    masked_trace_indices = trace_mask_table.loc[trace_mask_table["drop"], "trace_index"].to_numpy(dtype=np.int64)

    return {
        "trace_mask_table": trace_mask_table,
        "cell_mask_table": cell_mask_table,
        "kept_trace_indices": kept_trace_indices,
        "masked_trace_indices": masked_trace_indices,
    }


def summarize_mask_statistics(trace_mask_table: pd.DataFrame,
                              cell_mask_table: pd.DataFrame,
                              scope_info: Dict[str, Any],
                              mask_info: Dict[str, Any],
                              source_meta: Dict[str, Any],
                              applied_modes: Sequence[str]) -> Dict[str, Any]:
    eligible_mask = np.zeros(len(cell_mask_table), dtype=bool)
    if scope_info.get("scope_type") == "global":
        eligible_mask[:] = True
    elif scope_info.get("eligible_cell_count", 0) > 0:
        start_imx = scope_info.get("start_imx")
        end_imx = scope_info.get("end_imx")
        start_imy = scope_info.get("start_imy")
        end_imy = scope_info.get("end_imy")
        eligible_mask = (
            (cell_mask_table["imx"] >= start_imx) & (cell_mask_table["imx"] <= end_imx) &
            (cell_mask_table["imy"] >= start_imy) & (cell_mask_table["imy"] <= end_imy)
        ).to_numpy(dtype=bool)

    masked_support = int(cell_mask_table["drop"].sum())
    total_support = int(len(cell_mask_table))
    masked_trace = int(trace_mask_table["drop"].sum())
    total_trace = int(len(trace_mask_table))
    local_support = int(eligible_mask.sum())
    local_masked = int((cell_mask_table["drop"].to_numpy(dtype=bool) & eligible_mask).sum())

    stats = {
        "applied_modes": list(applied_modes),
        "source_meta": source_meta,
        "scope_info": scope_info,
        "mask_info": mask_info,
        "total_support": total_support,
        "masked_support": masked_support,
        "kept_support": total_support - masked_support,
        "actual_missing_ratio": float(masked_support / total_support) if total_support > 0 else 0.0,
        "total_traces": total_trace,
        "masked_traces": masked_trace,
        "kept_traces": total_trace - masked_trace,
        "trace_missing_ratio": float(masked_trace / total_trace) if total_trace > 0 else 0.0,
        "local_midpoint_support": local_support,
        "local_midpoint_masked": local_masked,
        "local_midpoint_missing_ratio": float(local_masked / local_support) if local_support > 0 else 0.0,
    }

    if "sector_count" in mask_info:
        stats["azimuth_sector_count"] = int(mask_info["sector_count"])
    if "near_threshold" in mask_info:
        stats["near_threshold"] = mask_info["near_threshold"]
    if "far_threshold" in mask_info:
        stats["far_threshold"] = mask_info["far_threshold"]
    return stats


def _save_table(df: pd.DataFrame, path: str, fmt: str = "csv") -> str:
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported save format: {fmt}")
    return path


def _plot_mask_preview(cell_mask_table: pd.DataFrame, output_path: str):
    if plt is None or cell_mask_table.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = np.where(cell_mask_table["drop"].to_numpy(dtype=bool), "tab:red", "tab:blue")

    axes[0].scatter(
        cell_mask_table["mx_center"],
        cell_mask_table["my_center"],
        c=colors,
        s=18,
        alpha=0.8,
        edgecolors="none",
    )
    axes[0].set_title("Midpoint support mask")
    axes[0].set_xlabel("mx_center")
    axes[0].set_ylabel("my_center")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].scatter(
        cell_mask_table["hx_center"],
        cell_mask_table["hy_center"],
        c=colors,
        s=18,
        alpha=0.8,
        edgecolors="none",
    )
    axes[1].set_title("Offset-vector support mask")
    axes[1].set_xlabel("hx_center")
    axes[1].set_ylabel("hy_center")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_mask_results(result: Dict[str, Any],
                      output_dir: str,
                      output_tag: str,
                      save_fmt: str = "csv",
                      save_preview: bool = False) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    safe_tag = output_tag.replace("/", "_")

    kept_path = os.path.join(output_dir, f"kept_trace_indices_{safe_tag}.npy")
    masked_path = os.path.join(output_dir, f"masked_trace_indices_{safe_tag}.npy")
    stats_path = os.path.join(output_dir, f"ovt_mask_stats_{safe_tag}.json")
    trace_table_path = os.path.join(output_dir, f"ovt_mask_table_{safe_tag}.{save_fmt}")
    cell_table_path = os.path.join(output_dir, f"ovt_mask_cells_{safe_tag}.{save_fmt}")
    preview_path = os.path.join(output_dir, f"ovt_mask_preview_{safe_tag}.png")

    np.save(kept_path, result["kept_trace_indices"])
    np.save(masked_path, result["masked_trace_indices"])
    _save_table(result["trace_mask_table"], trace_table_path, fmt=save_fmt)
    _save_table(result["cell_mask_table"], cell_table_path, fmt=save_fmt)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result["stats"]), f, indent=2, ensure_ascii=False)

    output_paths = {
        "kept_trace_indices": kept_path,
        "masked_trace_indices": masked_path,
        "stats": stats_path,
        "trace_mask_table": trace_table_path,
        "cell_mask_table": cell_table_path,
    }
    if save_preview:
        maybe_path = _plot_mask_preview(result["cell_mask_table"], preview_path)
        if maybe_path is not None:
            output_paths["preview"] = maybe_path
    return output_paths


def _dispatch_single_mask(cell_df: pd.DataFrame,
                          midpoint_df: pd.DataFrame,
                          eligible_mask: np.ndarray,
                          mode: str,
                          config: Dict[str, Any],
                          rng: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
    if mode == "random_bin":
        return generate_random_bin_mask(
            cell_df,
            eligible_mask=eligible_mask,
            missing_ratio=config.get("missing_ratio", 0.4),
            rng=rng,
        )

    if mode == "azimuth_sector":
        return generate_azimuth_sector_mask(
            cell_df,
            eligible_mask=eligible_mask,
            sectors=config.get("sectors", []),
            angle_unit=config.get("angle_unit", "degree"),
            reciprocal_pair=config.get("reciprocal_pair", False),
        )

    if mode == "offset_truncation":
        return generate_offset_truncation_mask(
            cell_df,
            eligible_mask=eligible_mask,
            truncation_mode=config["truncation_mode"],
            near_threshold=config.get("near_threshold"),
            far_threshold=config.get("far_threshold"),
            near_quantile=config.get("near_quantile"),
            far_quantile=config.get("far_quantile"),
            quantile_scope=config.get("quantile_scope", "global_cell"),
        )

    if mode == "midpoint_block":
        return generate_midpoint_block_mask(
            cell_df,
            midpoint_df=midpoint_df,
            eligible_mask=eligible_mask,
            rng=rng,
            width=config.get("width"),
            height=config.get("height"),
            patch_ratio=config.get("patch_ratio", 0.25),
            start_imx=config.get("start_imx"),
            start_imy=config.get("start_imy"),
            center_imx=config.get("center_imx"),
            center_imy=config.get("center_imy"),
            subcondition=config.get("subcondition"),
        )

    raise ValueError(f"unsupported OVT mask mode: {mode}")


def dispatch_ovt_mask(source: Any,
                      source_type: str = "h5",
                      mode: str = "random_bin",
                      mask_mode: str = "eval",
                      config: Optional[Dict[str, Any]] = None,
                      mixture: Optional[Sequence[Dict[str, Any]]] = None,
                      group_name: Optional[str] = None,
                      table_fmt: Optional[str] = None,
                      mx_bin: float = None,
                      my_bin: float = None,
                      hx_bin: float = None,
                      hy_bin: float = None,
                      seed: int = 42,
                      min_keep_cells: int = 1) -> Dict[str, Any]:
    config = dict(config or {})
    rng = np.random.RandomState(seed)
    support = build_support_index(
        source=source,
        source_type=source_type,
        group_name=group_name,
        table_fmt=table_fmt,
        mx_bin=mx_bin,
        my_bin=my_bin,
        hx_bin=hx_bin,
        hy_bin=hy_bin,
    )
    trace_df = support["trace_df"]
    cell_df = support["cell_df"]
    midpoint_df = support["midpoint_df"]

    scope_cfg = config.get("scope", {"type": "global"})
    eligible_mask, scope_info = select_midpoint_scope(cell_df, midpoint_df, scope_cfg, rng)

    if mask_mode == "eval":
        cell_drop_mask, mask_info = _dispatch_single_mask(
            cell_df,
            midpoint_df,
            eligible_mask=eligible_mask,
            mode=mode,
            config=config,
            rng=rng,
        )
        applied_modes = [mode]
    elif mask_mode == "train":
        mixture = list(mixture or [])
        if not mixture:
            mixture = [{
                "type": mode,
                "prob": 1.0,
                "params": config,
            }]
        probs = np.array([float(item.get("prob", 1.0)) for item in mixture], dtype=np.float64)
        if np.any(probs < 0):
            raise ValueError("mixture probabilities must be non-negative")
        if probs.sum() <= 0:
            raise ValueError("mixture probabilities must sum to a positive value")
        probs = probs / probs.sum()

        sample_strategy = config.get("sample_strategy", "one")
        mask_items: List[np.ndarray] = []
        applied_modes = []
        mask_info = {"sample_strategy": sample_strategy}
        if sample_strategy == "one":
            choice = int(rng.choice(len(mixture), p=probs))
            item = mixture[choice]
            current_mode = item["type"]
            item_scope = item.get("params", {}).get("scope", scope_cfg)
            item_eligible_mask, scope_info = select_midpoint_scope(cell_df, midpoint_df, item_scope, rng)
            current_mask, current_info = _dispatch_single_mask(
                cell_df,
                midpoint_df,
                eligible_mask=item_eligible_mask,
                mode=current_mode,
                config=item.get("params", {}),
                rng=rng,
            )
            mask_items.append(current_mask)
            mask_info["selected_mode"] = current_mode
            mask_info.update(current_info)
            applied_modes.append(current_mode)
        elif sample_strategy == "all":
            for item, prob in zip(mixture, probs):
                if rng.rand() > prob:
                    continue
                current_mode = item["type"]
                item_scope = item.get("params", {}).get("scope", scope_cfg)
                item_eligible_mask, _ = select_midpoint_scope(cell_df, midpoint_df, item_scope, rng)
                current_mask, _ = _dispatch_single_mask(
                    cell_df,
                    midpoint_df,
                    eligible_mask=item_eligible_mask,
                    mode=current_mode,
                    config=item.get("params", {}),
                    rng=rng,
                )
                mask_items.append(current_mask)
                applied_modes.append(current_mode)
            if not mask_items:
                mask_items = [np.zeros(len(cell_df), dtype=bool)]
                applied_modes = ["none"]
        else:
            raise ValueError(f"unsupported train sample_strategy: {sample_strategy}")

        cell_drop_mask = combine_masks(mask_items, min_keep_cells=min_keep_cells, rng=rng)
    else:
        raise ValueError(f"unsupported mask_mode: {mask_mode}")

    cell_drop_mask = combine_masks([cell_drop_mask], min_keep_cells=min_keep_cells, rng=rng)
    applied = apply_mask_to_support(trace_df, cell_df, cell_drop_mask)
    stats = summarize_mask_statistics(
        applied["trace_mask_table"],
        applied["cell_mask_table"],
        scope_info=scope_info,
        mask_info=mask_info,
        source_meta=support["source_meta"],
        applied_modes=applied_modes,
    )

    result = {
        **applied,
        "stats": stats,
        "scope_info": scope_info,
        "mask_info": mask_info,
        "support": support,
        "applied_modes": applied_modes,
    }
    return result
