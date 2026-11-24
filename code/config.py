"""Shared paths and constants for the SDNET2018 crack-detection project.
CE 235 Course Project - Shreyansh Tripathi (24B0726)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RAW_REL = Path("DATA_Maguire_20180517_ALL") / "DATA_Maguire_20180517_ALL" / "SDNET2018"
# raw dataset inside the project folder; fall back to a copy next to it
RAW_DIR = ROOT / _RAW_REL if (ROOT / _RAW_REL).exists() else ROOT.parent / _RAW_REL
PROC_DIR = ROOT / "data_processed"
CACHE_DIR = ROOT / "cache"          # large intermediate arrays (not submitted)
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
for d in (PROC_DIR, CACHE_DIR, FIG_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 128                       # CNN input / feature-extraction resolution
SURFACES = {"D": "Bridge deck", "W": "Wall", "P": "Pavement"}
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}
