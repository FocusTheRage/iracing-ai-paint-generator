"""
iRacing car definitions with exact paint-directory folder paths.
Source: https://support.iracing.com/support/solutions/articles/31000172625
"""

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


def get_paint_install_path(car: IRacingCar, customer_id: str) -> Path:
    """Return the full Windows path where paint files should be dropped."""
    docs = Path.home() / "Documents" / "iRacing" / "paint" / car.folder_path
    return docs


def get_output_filenames(customer_id: str) -> tuple[str, str]:
    """Return standard iRacing paint filenames for a customer ID."""
    cid = str(customer_id).strip()
    return f"car_{cid}.tga", f"car_spec_{cid}.tga"


def build_install_instructions(car: IRacingCar, customer_id: str) -> str:
    """Human-readable copy instructions for local iRacing install."""
    install_dir = get_paint_install_path(car, customer_id)
    paint_name, spec_name = get_output_filenames(customer_id)
    return (
        f"### Install in iRacing (local)\n"
        f"1. Copy `{paint_name}` to:\n"
        f"   `{install_dir}`\n"
        f"2. Copy `{spec_name}` to the same folder.\n"
        f"3. Launch iRacing and select your custom paint in the garage.\n"
        f"4. If the paint does not appear, delete any older `car_{customer_id}.mip` "
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