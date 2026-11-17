"""Utilidades de carga de datos de mercado y fijación de semillas."""

import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MARKET_COLUMNS = ["ask1", "bid1", "vask1", "vbid1", "last", "cumvol"]


def set_seeds(seed: int = 42) -> None:
    """Fija las semillas aleatorias de Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_market_data(
    asset: str = "rep",
    split: str = "train",
    data_dir: str | Path = "data",
) -> dict[datetime.date, pd.DataFrame]:
    """Carga los datos del libro de órdenes agrupados por día bursátil."""
    split_norm = "val" if split == "validation" else split
    base_dir = Path(data_dir)
    if not base_dir.exists():
        # Resolución relativa a la raíz del repositorio
        repo_root = Path(__file__).resolve().parents[2]
        base_dir = repo_root / data_dir

    file_path = base_dir / f"{asset}_{split_norm}.csv.gz"
    if not file_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de datos: {file_path}")

    df = pd.read_csv(file_path, index_col="time", parse_dates=["time"])
    cols = [c for c in MARKET_COLUMNS if c in df.columns]
    return {
        datetime.date.fromisoformat(str(day)): group[cols].copy()
        for day, group in df.groupby("date", sort=True)
    }
