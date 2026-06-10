"""
Build UV atlas + labeled guides for NASCAR Cup Next Gen Toyota Camry.

Cup NG uses a different layout than ARCA: top-down plan view in the center and
side-profile panels in the lower half of the 2048 sheet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from template_manager import CACHE_VERSION  # noqa: E402
from uv_atlas import (  # noqa: E402
    build_ai_generation_guide,
    build_ai_layout_guide,
    build_labeled_guide,
    load_atlas_for_car,
    sync_atlas_panel_regions,
)

FOLDER = r"stockcars\toyotacamry2022"
SLUG = "stockcars_toyotacamry2022"
CACHE_DIR = ROOT / "templates" / "cache" / SLUG
ATLAS_JSON = ROOT / "templates" / "atlas" / f"{SLUG}.json"

# Zones from templates/ai_layout_guideEdited.png (delete markers removed, tags relocated).
REGION_ZONES: dict[str, tuple[float, float, float, float]] = {
    "front_corner_upper_left": (0.0, 0.0, 0.24, 0.25),
    "front_bumper": (0.15, 0.03, 0.55, 0.20),
    "front_corner_upper_bumper_right": (0.25, 0.0, 0.60, 0.25),
    "back": (0.55, 0.0, 0.88, 0.20),
    "rear_diffuser": (0.86, 0.05, 1.0, 0.35),
    "front_corner_upper_right": (0.34, 0.13, 0.68, 0.31),
    "left_side_door": (0.35, 0.22, 0.65, 0.40),
    "trunk": (0.0, 0.24, 0.35, 0.48),
    "roof": (0.30, 0.24, 0.62, 0.48),
    "hood": (0.55, 0.24, 1.0, 0.48),
    "rear_corner_lower": (0.0, 0.48, 0.22, 0.72),
    "right_side_door": (0.35, 0.40, 0.65, 0.58),
    "front_corner_lower_right": (0.52, 0.48, 0.72, 0.72),
    "pit_sign": (0.0, 0.72, 0.30, 0.88),
    "right_lower_door": (0.30, 0.72, 0.65, 0.88),
    "front_splitter_right": (0.30, 0.88, 0.75, 1.0),
}

LABEL_ANCHORS: dict[str, tuple[float, float]] = {
    "front_corner_upper_left": (0.067, 0.118),
    "front_bumper": (0.253, 0.065),
    "front_corner_upper_bumper_right": (0.426, 0.114),
    "back": (0.682, 0.087),
    "rear_diffuser": (0.953, 0.099),
    "front_corner_upper_right": (0.511, 0.176),
    "left_side_door": (0.500, 0.259),
    "trunk": (0.220, 0.308),
    "roof": (0.481, 0.353),
    "hood": (0.745, 0.378),
    "rear_corner_lower": (0.131, 0.580),
    "right_side_door": (0.500, 0.590),
    "front_corner_lower_right": (0.601, 0.628),
    "pit_sign": (0.158, 0.812),
    "right_lower_door": (0.479, 0.839),
    "front_splitter_right": (0.472, 0.934),
}

REGION_META: dict[str, dict] = {
    "front_corner_upper_left": {
        "label": "B CORNER LEFT",
        "display_name": "Bumper corner (left)",
        "aliases": ["bumper corner left", "b corner left", "front bumper corner left"],
        "in_game_hint": "Top-left bumper corner cap.",
    },
    "front_corner_upper_bumper_right": {
        "label": "B CORNER RIGHT",
        "display_name": "Bumper corner (right)",
        "aliases": ["bumper corner right", "b corner right", "front bumper corner right"],
        "in_game_hint": "Top-center/right bumper corner cap.",
    },
    "front_bumper": {
        "label": "FRONT BUMPER",
        "display_name": "Front bumper",
        "aliases": ["front bumper", "front", "nose", "grille"],
        "in_game_hint": "Top-row front bumper fascia.",
    },
    "back": {
        "label": "REAR",
        "display_name": "Rear body",
        "aliases": ["back", "rear", "rear body", "rear clip"],
        "in_game_hint": "Top-right rear body panel.",
    },
    "rear_diffuser": {
        "label": "REAR DIFFUSER",
        "display_name": "Rear diffuser",
        "aliases": ["rear diffuser", "diffuser"],
        "in_game_hint": "Top-right rear diffuser stack.",
    },
    "front_corner_upper_right": {
        "label": "FRONT L QUARTER",
        "display_name": "Front left quarter",
        "aliases": [
            "front left corner",
            "front left quarter",
            "front left quarter panel",
            "left front quarter",
        ],
        "in_game_hint": "Upper accent — front left quarter panel.",
    },
    "left_side_door": {
        "label": "LEFT DOOR",
        "display_name": "Left door",
        "aliases": ["left door", "left side door", "driver door", "driver side door"],
        "in_game_hint": "Center plan view — driver door.",
    },
    "trunk": {
        "label": "TRUNK",
        "display_name": "Trunk / rear deck",
        "aliases": ["trunk", "rear deck", "deck lid", "tail"],
        "in_game_hint": "Plan view left — trunk lid.",
    },
    "roof": {
        "label": "ROOF",
        "display_name": "Roof",
        "aliases": ["roof", "top", "roof top"],
        "in_game_hint": "Plan view center — roof and greenhouse.",
    },
    "hood": {
        "label": "HOOD",
        "display_name": "Hood",
        "aliases": ["hood", "bonnet"],
        "in_game_hint": "Plan view right — hood.",
    },
    "rear_corner_lower": {
        "label": "REAR QUARTER",
        "display_name": "Rear quarter",
        "aliases": [
            "rear quarter",
            "rear corner",
            "right rear corner",
            "right rear quarter",
            "rear corner lower",
        ],
        "in_game_hint": "Lower-left plan silhouette — rear quarter panel.",
    },
    "right_side_door": {
        "label": "RIGHT DOOR",
        "display_name": "Right door",
        "aliases": [
            "right door",
            "right side door",
            "passenger door",
            "passenger side door",
        ],
        "in_game_hint": "Center-lower plan view — passenger door.",
    },
    "front_corner_lower_right": {
        "label": "FRONT R QUARTER",
        "display_name": "Front right quarter",
        "aliases": [
            "front right corner",
            "front right quarter",
            "right front corner",
            "right front quarter",
        ],
        "in_game_hint": "Lower-right plan silhouette — front right quarter.",
    },
    "pit_sign": {
        "label": "PIT SIGN",
        "display_name": "Pit sign area",
        "aliases": ["pit sign", "pit box sign", "windshield banner"],
        "in_game_hint": "Lower-left pit sign / banner panel.",
    },
    "right_lower_door": {
        "label": "RIGHT LOWER DOOR",
        "display_name": "Right lower door",
        "aliases": ["right lower door", "lower right door", "passenger lower door"],
        "in_game_hint": "Bottom-center lower door / rocker area.",
    },
    "front_splitter_right": {
        "label": "FRONT SPLITTER",
        "display_name": "Front splitter",
        "aliases": ["front splitter", "splitter", "front lip"],
        "in_game_hint": "Bottom strip — front splitter / lip.",
    },
}

PREFERRED_ORDER = [
    "front_corner_upper_left",
    "front_bumper",
    "front_corner_upper_bumper_right",
    "back",
    "rear_diffuser",
    "front_corner_upper_right",
    "left_side_door",
    "trunk",
    "roof",
    "hood",
    "rear_corner_lower",
    "right_side_door",
    "front_corner_lower_right",
    "pit_sign",
    "right_lower_door",
    "front_splitter_right",
]


def bbox_from_mask_zone(
    mask: np.ndarray,
    zone: tuple[float, float, float, float],
    min_pixels: int = 80,
) -> list[float]:
    paintable = mask > 128
    h, w = mask.shape
    zx0, zy0, zx1, zy1 = zone
    px0, py0 = max(0, int(zx0 * w)), max(0, int(zy0 * h))
    px1, py1 = min(w, int(zx1 * w)), min(h, int(zy1 * h))
    sub = paintable[py0:py1, px0:px1]
    if sub.sum() < min_pixels:
        return []
    ys, xs = np.where(sub)
    return [
        round((px0 + int(xs.min())) / w, 4),
        round((py0 + int(ys.min())) / h, 4),
        round((px0 + int(xs.max()) + 1) / w, 4),
        round((py0 + int(ys.max()) + 1) / h, 4),
    ]


def main() -> None:
    mask_path = CACHE_DIR / "mask.png"
    if not mask_path.exists():
        raise SystemExit(f"Missing mask — run get_car_template first: {mask_path}")

    mask = np.array(Image.open(mask_path).convert("L"))
    regions: dict[str, dict] = {}

    print("Cup Camry refined bboxes:")
    for region_id in PREFERRED_ORDER:
        zone = REGION_ZONES[region_id]
        bbox = bbox_from_mask_zone(mask, zone)
        if not bbox:
            print(f"  {region_id}: WARNING empty zone {zone}")
            continue
        meta = REGION_META[region_id]
        entry = {
            "id": region_id,
            "label": meta["label"],
            "display_name": meta["display_name"],
            "aliases": meta.get("aliases", []),
            "in_game_hint": meta.get("in_game_hint", ""),
            "bbox": bbox,
            "label_anchor": list(LABEL_ANCHORS.get(region_id, (
                round((bbox[0] + bbox[2]) / 2, 4),
                round((bbox[1] + bbox[3]) / 2, 4),
            ))),
        }
        regions[region_id] = entry
        print(f"  {region_id}: {bbox}")

    atlas = {
        "car_name": "NASCAR Cup Next Gen Toyota Camry",
        "folder_path": FOLDER,
        "resolution": 2048,
        "version": 4,
        "reference_png": "",
        "layout_lines": [
            "TOP ROW = B CORNER LEFT + FRONT BUMPER + B CORNER RIGHT + REAR + REAR DIFFUSER",
            "UPPER BODY = FRONT L QUARTER above plan view",
            "PLAN VIEW = LEFT DOOR (center-top), TRUNK (left), ROOF (center), HOOD (right)",
            "MID BODY = REAR QUARTER (left), RIGHT DOOR (center), FRONT R QUARTER (right)",
            "LOWER BODY = PIT SIGN (left), RIGHT LOWER DOOR (center)",
            "BOTTOM = FRONT SPLITTER strip",
        ],
        "alias_groups": {
            "rear": ["back", "rear_diffuser", "trunk", "rear_corner_lower"],
            "back": ["back"],
            "front": [
                "front_bumper",
                "front_corner_upper_left",
                "front_corner_upper_bumper_right",
                "front_corner_upper_right",
                "front_corner_lower_right",
                "front_splitter_right",
            ],
            "hood": ["hood"],
            "roof": ["roof"],
            "trunk": ["trunk"],
            "driver_side": ["left_side_door", "pit_sign"],
            "passenger_side": ["right_side_door", "right_lower_door"],
            "left_side": ["left_side_door", "rear_corner_lower", "trunk", "pit_sign"],
            "right_side": [
                "right_side_door",
                "right_lower_door",
                "hood",
                "front_corner_upper_bumper_right",
                "front_corner_upper_right",
                "front_corner_lower_right",
                "front_splitter_right",
            ],
            "door": ["left_side_door", "right_side_door", "right_lower_door"],
            "bumper": [
                "front_bumper",
                "front_corner_upper_left",
                "front_corner_upper_bumper_right",
            ],
            "corner": ["rear_corner_lower", "front_corner_upper_right", "front_corner_lower_right"],
            "fender": ["rear_corner_lower", "front_corner_upper_right", "front_corner_lower_right"],
            "diffuser": ["rear_diffuser"],
            "rear_quarter": ["rear_corner_lower"],
            "quarter": ["rear_corner_lower", "front_corner_upper_right", "front_corner_lower_right"],
            "front_bumper_corner": [
                "front_corner_upper_left",
                "front_corner_upper_bumper_right",
            ],
            "pit_sign": ["pit_sign"],
        },
        "regions": [regions[rid] for rid in PREFERRED_ORDER if rid in regions],
    }

    ATLAS_JSON.parent.mkdir(parents=True, exist_ok=True)
    ATLAS_JSON.write_text(json.dumps(atlas, indent=2), encoding="utf-8")
    print(f"\nWrote {ATLAS_JSON} ({len(atlas['regions'])} regions)")

    uv_atlas = load_atlas_for_car(FOLDER)
    if uv_atlas is None:
        print("Atlas load failed")
        return

    mask_img = Image.open(mask_path)
    pa_path = CACHE_DIR / "paintable_reference.png"
    paintable_reference = Image.open(pa_path) if pa_path.exists() else None
    labeled = build_labeled_guide(mask_img, uv_atlas, paintable_reference, 2048)
    ai_layout = build_ai_layout_guide(mask_img, uv_atlas, 2048)
    ai_gen = build_ai_generation_guide(mask_img, uv_atlas, 2048)
    labeled.save(CACHE_DIR / "labeled_guide.png")
    ai_layout.save(CACHE_DIR / "ai_layout_guide.png")
    ai_gen.save(CACHE_DIR / "ai_generation_guide.png")
    labeled.save(ROOT / "templates" / "atlas" / f"{SLUG}_labeled_guide.png")
    panels = sync_atlas_panel_regions(uv_atlas)
    (CACHE_DIR / "panels.json").write_text(json.dumps(panels, indent=2), encoding="utf-8")
    (CACHE_DIR / "version.txt").write_text(str(CACHE_VERSION), encoding="utf-8")
    print("Regenerated guides and panels.json")


if __name__ == "__main__":
    main()