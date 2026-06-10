"""
Load the full iRacing car catalog scraped from Trading Paints.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from cars_config import IRacingCar

CATALOG_PATH = Path(__file__).parent / "data" / "iracing_car_catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Car catalog missing at {CATALOG_PATH}. "
            "Run: python scripts/scrape_trading_paints_catalog.py"
        )
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def all_iracing_cars() -> tuple[IRacingCar, ...]:
    data = load_catalog()
    cars: list[IRacingCar] = []
    for entry in data["cars"]:
        cars.append(
            IRacingCar(
                display_name=entry["display_name"],
                folder_path=entry["folder_path"],
                category=entry.get("category", "Road"),
                template_hint=entry.get("template_hint", "gt"),
            )
        )
    return tuple(cars)


@lru_cache(maxsize=1)
def template_zip_by_folder() -> dict[str, str]:
    return dict(load_catalog()["zip_by_folder"])