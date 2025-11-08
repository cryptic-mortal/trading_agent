from pathlib import Path
from typing import List
import pandas as pd


def load_tickers_from_master(root_path: str = None, which: str = "tickers") -> List[str]:
    """
    Load tickers from top-level tickers CSV files in the SmartFolio repo.

    Args:
        root_path: path to repository root. If None, assumes current working directory.
        which: 'tickers' -> tickers.csv, 'tickers1' -> tickers1.csv

    Returns:
        List of ticker symbols (strings)
    """
    if root_path is None:
        # Try to find the workspace root (parent of integration folder)
        current = Path(__file__).resolve().parent
        if current.name == 'integration':
            root_path = current.parent
        else:
            root_path = Path.cwd()
    else:
        root_path = Path(root_path)

    filename = "tickers.csv" if which == "tickers" else "tickers1.csv"
    csv_path = root_path / "SmartFolio" / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'ticker' not in df.columns:
        # try first column
        tickers = df.iloc[:, 0].astype(str).tolist()
    else:
        tickers = df['ticker'].astype(str).tolist()

    # Ensure no nan or empty strings
    tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
    return tickers


def write_ticker_list(tickers: List[str], out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        for t in tickers:
            f.write(t + '\n')

