"""
iRacing car definitions with exact paint-directory folder paths.
Source: https://support.iracing.com/support/solutions/articles/31000172625
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class IRacingCar:
    """Metadata for a single iRacing vehicle paint slot."""

    display_name: str
    folder_path: str  # Relative to Documents/iRacing/paint/
    category: str
    resolution: int = 2048
    supports_spec_map: bool = True
    template_hint: str = "gt"  # gt | nascar | openwheel | dirt


def _load_cars() -> list[IRacingCar]:
    from car_catalog import all_iracing_cars

    return list(all_iracing_cars())


IRACING_CARS: list[IRacingCar] = _load_cars()
CAR_CHOICES: list[str] = [car.display_name for car in IRACING_CARS]
CAR_BY_NAME: dict[str, IRacingCar] = {car.display_name: car for car in IRACING_CARS}

PAINT_ROOT = Path.home() / "Documents" / "iRacing" / "paint"


def get_nested_paint_install_path(car: IRacingCar) -> Path:
    """Official nested path, e.g. paint\\stockcars2\\arcaford25."""
    dest = PAINT_ROOT
    for part in re.split(r"[/\\]+", car.folder_path.strip()):
        if part:
            dest = dest / part
    return dest


def get_legacy_flat_paint_install_path(car: IRacingCar) -> Path:
    """
    Legacy Windows flat folder many installs use, e.g. paint\\stockcars2 arcaford25.
    Backslashes in the catalog path become spaces in a single folder name.
    """
    flat_name = re.sub(r"[/\\]+", " ", car.folder_path.strip())
    return PAINT_ROOT / flat_name


def _paint_folder_in_use(path: Path) -> bool:
    """True if iRacing or the user has placed paints in this folder before."""
    if not path.exists():
        return False
    return any(path.glob("*.mip")) or any(path.glob("car_*.tga"))


def get_paint_install_paths(car: IRacingCar, customer_id: str = "") -> list[Path]:
    """
    Return paint folder(s) to write TGAs into.

    Many Windows iRacing installs use legacy flat folders (backslash -> space) even
    though the official docs list nested paths. We copy to every applicable folder
    so paints show up regardless of which layout this install uses.
    """
    nested = get_nested_paint_install_path(car)
    flat = get_legacy_flat_paint_install_path(car)
    if nested == flat:
        return [nested]

    nested_used = _paint_folder_in_use(nested)
    flat_used = _paint_folder_in_use(flat)

    if flat_used and not nested_used:
        ordered = [flat, nested]
    elif nested_used and not flat_used:
        ordered = [nested, flat]
    elif flat_used and nested_used:
        flat_mips = any(flat.glob("*.mip"))
        nested_mips = any(nested.glob("*.mip"))
        if flat_mips and not nested_mips:
            ordered = [flat, nested]
        elif nested_mips and not flat_mips:
            ordered = [nested, flat]
        else:
            ordered = [flat, nested]
    else:
        # New car — write both layouts so the sim finds the files either way.
        ordered = [flat, nested]

    seen: set[str] = set()
    paths: list[Path] = []
    for candidate in ordered:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            paths.append(candidate)
    return paths


def get_paint_install_path(car: IRacingCar, customer_id: str = "") -> Path:
    """Primary paint folder (first choice from get_paint_install_paths)."""
    return get_paint_install_paths(car, customer_id)[0]


def get_output_filenames(customer_id: str) -> tuple[str, str]:
    """Return standard iRacing paint filenames for a customer ID."""
    cid = str(customer_id).strip()
    return f"car_{cid}.tga", f"car_spec_{cid}.tga"


def build_install_instructions(car: IRacingCar, customer_id: str) -> str:
    """Human-readable copy instructions for local iRacing install."""
    install_dirs = get_paint_install_paths(car, customer_id)
    paint_name, spec_name = get_output_filenames(customer_id)
    folder_lines = "\n".join(f"   - `{d}`" for d in install_dirs)
    return (
        f"### Install in iRacing (local)\n"
        f"1. Copy `{paint_name}` and `{spec_name}` to:\n"
        f"{folder_lines}\n"
        f"2. Launch iRacing and select your custom paint in the garage.\n"
        f"3. If the paint does not appear, delete any older `car_{customer_id}.mip` "
        f"files in that folder and restart the sim.\n"
    )


def build_trading_paints_instructions(car: IRacingCar, customer_id: str) -> str:
    """Steps for uploading to Trading Paints."""
    paint_name, spec_name = get_output_filenames(customer_id)
    return (
        f"### Upload to Trading Paints\n"
        f"1. Go to [tradingpaints.com](https://www.tradingpaints.com) and log in.\n"
        f"2. Open **My Paints** → **Upload Paint**.\n"
        f"3. Select car: **{car.display_name}**.\n"
        f"4. Upload `{paint_name}` as the paint file.\n"
        f"5. Upload `{spec_name}` as the spec map (Trading Paints accepts .tga; "
        f"it will convert to .mip on download).\n"
        f"6. Set visibility (Private / Friends / Public) and save.\n"
        f"7. In iRacing, enable Trading Paints and search by your Customer ID "
        f"**{customer_id}** or paint name.\n"
        f"\n"
        f"**Tip:** For chrome or metallic finishes, keep albedo colors lighter "
        f"than real life — the spec map controls reflectivity.\n"
    )