import json
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path

def _normalize_json_id(v: Any) -> Any:
    """将 JSON 读出的 numpy/float/int 统一为稳定的 Python 标量。"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        fv = float(v)
        return int(fv) if fv.is_integer() else fv
    return v


def _load_json_ids(path: Path) -> set:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Split file must contain a JSON list, got: {path}")
    return set(_normalize_json_id(v) for v in data)


def _candidate_split_dirs(split_dir: Path, mode_hint: Optional[str]) -> List[Path]:
    """按优先级返回可能的 split 目录。"""
    dirs: List[Path] = []
    split_dir = Path(split_dir)

    if mode_hint:
        hinted = split_dir / mode_hint
        if hinted.is_dir():
            dirs.append(hinted)

    dirs.append(split_dir)

    if split_dir.is_dir():
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and d not in dirs:
                dirs.append(d)
    return dirs


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _resolve_split_artifacts(
    split_dir: Path,
    file_name: str,
    split: str,
    missing_ratio: Optional[float] = None,
    missing_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    解析 split 文件与配置文件，兼容多种命名：
    1) <file>_<split>_ids_<missing_ratio>_<missing_mode>.json
    2) <file>_<split>_ids_<holdout_ratio>_<mode>.json
    3) <file>_<split>_ids.json
    """
    dirs = _candidate_split_dirs(Path(split_dir), missing_mode)

    # 先找可用的 split_config（可能用于推断 holdout_ratio/mode）
    cfg_candidates = [d / f"{file_name}_split_config.json" for d in dirs]
    cfg_path = _find_first_existing(cfg_candidates)
    cfg = _read_json_if_exists(cfg_path) if cfg_path is not None else None

    exact_names: List[str] = []
    if missing_ratio is not None and missing_mode is not None:
        exact_names.append(f"{file_name}_{split}_ids_{missing_ratio}_{missing_mode}.json")

    if cfg is not None:
        hr = cfg.get("holdout_ratio", None)
        md = cfg.get("mode", None)
        if hr is not None and md is not None:
            exact_names.append(f"{file_name}_{split}_ids_{hr}_{md}.json")

    exact_names.append(f"{file_name}_{split}_ids.json")

    # 先按精确文件名查找
    exact_paths = [d / n for d in dirs for n in exact_names]
    split_ids_path = _find_first_existing(exact_paths)

    # 再用模式匹配兜底
    if split_ids_path is None:
        patterns: List[str] = []
        if cfg is not None and cfg.get("mode", None) is not None:
            patterns.append(f"{file_name}_{split}_ids_*_{cfg['mode']}.json")
        if missing_mode is not None:
            patterns.append(f"{file_name}_{split}_ids_*_{missing_mode}.json")
        patterns.append(f"{file_name}_{split}_ids_*.json")

        for d in dirs:
            for pat in patterns:
                matches = sorted(d.glob(pat))
                if matches:
                    split_ids_path = matches[0]
                    break
            if split_ids_path is not None:
                break

    if split_ids_path is None:
        searched = ", ".join(str(d) for d in dirs)
        raise FileNotFoundError(
            f"Cannot find split file for split='{split}', file='{file_name}' under: {searched}"
        )

    # 优先使用 split_ids 所在目录下的 config
    cfg_near_ids = split_ids_path.parent / f"{file_name}_split_config.json"
    cfg = _read_json_if_exists(cfg_near_ids) or cfg
    cfg_path = cfg_near_ids if cfg_near_ids.exists() else cfg_path

    return {
        "split_ids_path": split_ids_path,
        "split_ids": _load_json_ids(split_ids_path),
        "split_config": cfg if cfg is not None else {},
        "split_config_path": cfg_path,
        "split_base_dir": split_ids_path.parent,
    }