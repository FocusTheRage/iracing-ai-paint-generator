"""
Scrape Trading Paints cartemplates page and build iRacing car catalog JSON.

Output: data/iracing_car_catalog.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "iracing_car_catalog.json"
URL = "https://www.tradingpaints.com/cartemplates"

SECTION_HINT: dict[str, tuple[str, str]] = {
    "oval racing cars": ("Oval", "nascar"),
    "road racing cars": ("Road", "gt"),
    "other templates": ("Other", "gt"),
}

CAR_ANCHOR_RE = re.compile(
    r'<a[^>]+href="https://ir-core-sites\.iracing\.com/members/member_images/cars/car_templates/([^"]+\.zip)"[^>]*>'
    r"(.*?)</a>",
    re.I | re.S,
)


def tp_folder_to_iracing(folder_raw: str) -> str:
    """Convert 'stockcars toyotacamry2022' -> r'stockcars\\toyotacamry2022'."""
    folder_raw = folder_raw.strip().replace("/", "\\")
    if "\\" in folder_raw:
        return folder_raw
    parts = folder_raw.split(" ", 1)
    if len(parts) == 1:
        return parts[0]
    return parts[0] + "\\" + parts[1]


def infer_hint(category: str, name: str, folder: str) -> str:
    low = (name + " " + folder).lower()
    if any(x in low for x in ("dirt", "sprint car", "midget", "late model", "ump", "legends dirt")):
        return "dirt"
    if any(
        x in low
        for x in (
            "nascar",
            "arca",
            "truck",
            "stockcar",
            "street stock",
            "silver crown",
            "srx",
            "super late",
            "mini stock",
            "modified",
            "sprint",
        )
    ):
        if "dirt" in low:
            return "dirt"
        return "nascar"
    if any(
        x in low
        for x in (
            "indycar",
            "formula",
            "dallara",
            "indy pro",
            "super formula",
            "usf",
            "ir-",
            "ir01",
            "ir18",
        )
    ):
        return "openwheel"
    if category == "Oval":
        return "nascar"
    return "gt"


def section_at(pos: int, section_starts: list[tuple[int, str, str, str]]) -> tuple[str, str]:
    category, hint = "Road", "gt"
    for start, _title, cat, h in section_starts:
        if start <= pos:
            category, hint = cat, h
        else:
            break
    return category, hint


def parse_html(html: str) -> list[dict]:
    section_starts: list[tuple[int, str, str, str]] = []
    for m in re.finditer(r'<h2[^>]*>([^<]+)</h2>', html, re.I):
        title = m.group(1).strip()
        key = title.lower()
        if key in SECTION_HINT:
            cat, hint = SECTION_HINT[key]
            section_starts.append((m.start(), title, cat, hint))

    cars: list[dict] = []
    seen_names: set[str] = set()

    for m in CAR_ANCHOR_RE.finditer(html):
        zip_name = m.group(1)
        inner = m.group(2)
        name_m = re.search(r"<h3[^>]*>([^<]+)</h3>", inner, re.I)
        folder_m = re.search(
            r"Documents/iRacing/paint/([^<]+)",
            inner,
            re.I,
        )
        if not name_m or not folder_m:
            continue

        display_name = name_m.group(1).strip()
        if display_name.lower() in seen_names:
            continue
        seen_names.add(display_name.lower())

        folder_path = tp_folder_to_iracing(folder_m.group(1).strip())
        category, base_hint = section_at(m.start(), section_starts)
        hint = infer_hint(category, display_name, folder_path) or base_hint

        cars.append(
            {
                "display_name": display_name,
                "folder_path": folder_path,
                "category": category,
                "template_hint": hint,
                "zip_name": zip_name,
            }
        )

    return cars


def main() -> int:
    print(f"Fetching {URL} ...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    cars = parse_html(resp.text)
    if len(cars) < 100:
        print(f"WARNING: only parsed {len(cars)} cars — HTML structure may have changed")
        return 1

    by_folder: dict[str, str] = {}
    for c in cars:
        fp = c["folder_path"]
        if fp not in by_folder:
            by_folder[fp] = c["zip_name"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": URL,
        "car_count": len(cars),
        "unique_folders": len(by_folder),
        "cars": sorted(cars, key=lambda x: (x["category"], x["display_name"])),
        "zip_by_folder": by_folder,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(cars)} cars ({len(by_folder)} unique folders) -> {OUT_PATH}")

    # Sanity: spot-check known cars
    checks = {
        "ARCA Chevrolet SS": "198_template_ACSS25.zip",
        "NASCAR Cup Series Next Gen Toyota Camry": "141_template_NGT.zip",
        "BMW M4 GT3 EVO": "132_template_M4GT3.zip",
    }
    by_name = {c["display_name"]: c for c in cars}
    for name, expected_zip in checks.items():
        got = by_name.get(name, {}).get("zip_name", "MISSING")
        status = "OK" if got == expected_zip else f"FAIL (got {got})"
        print(f"  {status}: {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())