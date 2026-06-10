"""
Download/cache every car in cars_config and export raw wireframe PNGs for mapping.

Wireframes land in templates/wireframes/<slug>/wireframe.png (cyan on black).
Full template cache (mask, ai_generation_guide, etc.) stays in templates/cache/<slug>/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cars_config import IRACING_CARS  # noqa: E402
from template_manager import (  # noqa: E402
    CACHE_DIR,
    _mask_outline_overlay,
    _safe_slug,
    get_car_template,
)
from uv_atlas import load_atlas_for_car  # noqa: E402

WIREFRAMES_DIR = ROOT / "templates" / "wireframes"


def export_wireframe_png(mask: Image.Image, size: int) -> Image.Image:
    """Black background + cyan UV wire — for user zone mapping."""
    mask_l = mask.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    guide = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    wire = _mask_outline_overlay(mask_l, (size, size))
    return Image.alpha_composite(guide, wire)


def bootstrap_car(force_refresh: bool = False) -> dict:
    results: list[dict] = []
    WIREFRAMES_DIR.mkdir(parents=True, exist_ok=True)

    for car in IRACING_CARS:
        slug = _safe_slug(car.folder_path)
        entry: dict = {
            "display_name": car.display_name,
            "folder_path": car.folder_path,
            "slug": slug,
            "category": car.category,
            "status": "pending",
        }
        try:
            template = get_car_template(car, force_refresh=force_refresh)
            mask = template.paintable_mask
            size = car.resolution
            mask_pct = round((np.array(mask) > 128).mean() * 100, 1)
            atlas = load_atlas_for_car(car.folder_path)
            has_atlas = atlas is not None
            region_count = len(atlas.regions) if atlas else 0

            car_wire_dir = WIREFRAMES_DIR / slug
            car_wire_dir.mkdir(parents=True, exist_ok=True)
            wire_path = car_wire_dir / "wireframe.png"
            export_wireframe_png(mask, size).save(wire_path)

            # Flat alias for quick browsing: wireframes/<slug>.png
            flat_path = WIREFRAMES_DIR / f"{slug}.png"
            if not flat_path.exists() or flat_path.stat().st_mtime < wire_path.stat().st_mtime:
                Image.open(wire_path).save(flat_path)

            entry.update(
                {
                    "status": "ok",
                    "mask_coverage_pct": mask_pct,
                    "has_atlas": has_atlas,
                    "atlas_regions": region_count,
                    "wireframe": str(wire_path.relative_to(ROOT)).replace("\\", "/"),
                    "cache_dir": str((CACHE_DIR / slug).relative_to(ROOT)).replace("\\", "/"),
                    "source_zip": template.source_zip,
                }
            )
            print(
                f"OK  {car.display_name}: mask {mask_pct}%, "
                f"atlas={'yes' if has_atlas else 'no'} ({region_count} regions)"
            )
        except Exception as exc:
            entry["status"] = f"error: {exc}"
            print(f"ERR {car.display_name}: {exc}")

        results.append(entry)

    manifest = {
        "cars": results,
        "wireframes_dir": str(WIREFRAMES_DIR.relative_to(ROOT)).replace("\\", "/"),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] != "ok"),
    }
    manifest_path = WIREFRAMES_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}")
    print(f"Loaded {manifest['ok']}/{len(results)} cars")
    return manifest


def main() -> None:
    force = "--force" in sys.argv
    if force:
        print("Force refresh: re-downloading all templates from iRacing CDN...")
    bootstrap_car(force_refresh=force)


if __name__ == "__main__":
    main()