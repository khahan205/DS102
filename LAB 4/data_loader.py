"""
data_loader.py
==============
Hàm tiện ích để tìm và load dữ liệu Wine Quality (Red + White).

"""

import yaml
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path


def load_config(config_path: Path | str = "config.yaml") -> dict:
    """
    Đọc file config YAML chứa thông tin dataset.

    Parameters
    ----------
    config_path : Path | str – đường dẫn đến file config

    Returns
    -------

    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_csv_in_paths(filename: str, search_paths: list[str]) -> Path | None:


    for raw in search_paths:
        base = Path(raw).expanduser()
        if not base.exists():
            continue
        for p in base.rglob(filename):
            if p.is_file():
                return p
    return None


def download_csv(url: str, dest: Path) -> Path:

    
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading from {url} ...")
        request = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0'},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            dest.write_bytes(response.read())
        print(f"  Saved to {dest}")
        return dest
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def resolve_csv(filename: str, dataset_cfg: dict) -> Path:
    
    found = find_csv_in_paths(filename, dataset_cfg.get('search_paths', []))
    if found is not None:
        print(f"  Found local copy at {found}")
        return found

    download_cfg = dataset_cfg.get('download', {})
    if download_cfg.get('enabled', False):
        url     = download_cfg['base_url'].rstrip('/') + '/' + filename
        save_to = Path(download_cfg.get('save_to', '.')).expanduser()
        dest    = save_to / filename
        return download_csv(url, dest)

    raise FileNotFoundError(
        f"Cannot find '{filename}' in any search path "
        f"and download is disabled in config."
    )


def load_wine_data(
    config_path: Path | str = "config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    cfg         = load_config(config_path)
    dataset_cfg = cfg['dataset']

    red_path   = resolve_csv(dataset_cfg['files']['red'],   dataset_cfg)
    white_path = resolve_csv(dataset_cfg['files']['white'], dataset_cfg)

    red_df   = pd.read_csv(red_path,   sep=';')
    white_df = pd.read_csv(white_path, sep=';')
    df       = pd.concat([red_df, white_df], ignore_index=True)

    print(f"  Red wine   : {len(red_df)} samples")
    print(f"  White wine : {len(white_df)} samples")
    print(f"  Combined   : {len(df)} samples")

    return red_df, white_df, df


def extract_features(
    df:         pd.DataFrame,
    target_col: str = 'quality',
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Tách DataFrame thành ma trận đặc trưng X, vector nhãn y và tên các đặc trưng.

    Parameters
    ----------
    df         : pd.DataFrame – dữ liệu đầu vào
    target_col : str          – tên cột nhãn (mặc định 'quality')

    Returns
    -------
    X             : np.ndarray, shape (N, 11), dtype float64
    y             : np.ndarray, shape (N,),    dtype int64
    feature_names : list[str] – tên 11 đặc trưng hoá học
    """
    feature_names = list(df.drop(columns=[target_col]).columns)
    X = df.drop(columns=[target_col]).values.astype(np.float64)
    y = df[target_col].values.astype(np.int64)
    return X, y, feature_names
