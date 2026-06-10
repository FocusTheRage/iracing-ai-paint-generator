"""
Refine atlas region bboxes from PSD mask zones (full panel coverage).

Each region maps to a UV zone; bbox = paintable mask pixels in that zone.
Also adds rear_quarter_behind_door for the unlabeled panel behind the doors.
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

MASK = ROOT / "templates" / "cache" / "stockcars2_arcaford25" / "mask.png"
ATLAS_JSON = ROOT / "templates" / "atlas" / "stockcars2_arcaford25.json"
CACHE_DIR = ROOT / "templates" / "cache" / "stockcars2_arcaford25"
FOLDER = r"stockcars2\arcaford25"

# (x0, y0, x1, y1) normalized zones tuned to ARCA Ford Mustang UV layout.
REGION_ZONES: dict[str, tuple[float, float, float, float]] = {
    # Top row — bumper wings + center grille; shared bottom below left cap tail.
    "front_corner_upper_left": (0.0, 0.03, 0.058, 0.11),
    "front_bumper": (0.058, 0.03, 0.476, 0.11),
    "front_corner_upper_bumper_right": (0.476, 0.03, 0.529, 0.11),
    "back": (0.56, 0.0, 0.87, 0.14),
    "rear_bumper": (0.56, 0.08, 0.87, 0.22),
    "rear_diffuser": (0.86, 0.0, 1.0, 0.32),
    "left_side_door": (0.38, 0.17, 0.63, 0.41),
    "rear_quarter_behind_door": (0.22, 0.34, 0.55, 0.46),
    # Side quarter panels — pulled back from nose/tail clutter.
    "front_corner_upper_right": (0.62, 0.22, 0.88, 0.40),
    "front_corner_lower_right": (0.62, 0.76, 0.90, 0.92),
    # Rear quarters — shifted down to match side-profile alignment.
    "rear_corner_upper": (0.0, 0.28, 0.26, 0.47),
    "rear_corner_lower": (0.0, 0.76, 0.28, 0.94),
    "trunk": (0.0, 0.47, 0.27, 0.72),
    "roof": (0.24, 0.44, 0.54, 0.72),
    "hood": (0.66, 0.41, 1.0, 0.74),
    "right_side_door": (0.38, 0.74, 0.66, 1.0),
}

# Fixed label anchors (aligned bumper row + preserved anchors elsewhere).
LABEL_ANCHORS: dict[str, tuple[float, float]] = {
    "front_corner_upper_left": (0.029, 0.07),
    "front_bumper": (0.27, 0.07),
    "front_corner_upper_bumper_right": (0.502, 0.07),
    "back": (0.6987, 0.0957),
    "rear_bumper": (0.7026, 0.1519),
    "rear_diffuser": (0.9365, 0.1689),
    "left_side_door": (0.5117, 0.2754),
    "rear_quarter_behind_door": (0.38, 0.40),
    "front_corner_upper_right": (0.7319, 0.31),
    "rear_corner_upper": (0.1274, 0.37),
    "trunk": (0.0542, 0.5923),
    "roof": (0.4067, 0.5835),
    "hood": (0.8657, 0.5996),
    "right_side_door": (0.5479, 0.8896),
    "front_corner_lower_right": (0.7227, 0.84),
    "rear_corner_lower": (0.1221, 0.85),
}

REGION_METADATA: dict[str, dict] = {
    "front_corner_upper_left": {
        "label": "FRONT BUMPER CORNER",
        "display_name": "Front bumper corner (left)",
        "aliases": [
            "front bumper corner left",
            "left bumper corner",
            "front bumper left corner",
        ],
        "in_game_hint": "Left end cap of the front bumper row on the UV sheet.",
    },
    "front_corner_upper_bumper_right": {
        "label": "FRONT BUMPER CORNER",
        "display_name": "Front bumper corner (right)",
        "aliases": [
            "front bumper corner right",
            "right bumper corner",
            "front bumper right corner",
        ],
        "in_game_hint": "Right end cap of the front bumper row on the UV sheet.",
    },
    "front_corner_upper_right": {
        "label": "FRONT CORNER",
        "display_name": "Front corner (upper right quarter)",
        "aliases": ["front corner", "front quarter", "right front corner"],
        "in_game_hint": "Right-side upper quarter panel — not part of the bumper row.",
    },
    "front_corner_lower_right": {
        "label": "FRONT CORNER",
        "display_name": "Front corner (lower right quarter)",
        "aliases": ["front corner lower", "right front lower"],
        "in_game_hint": "Right-side lower quarter panel on the UV sheet.",
    },
}

NEW_REGION_DEFAULTS: dict[str, dict] = {
    "rear_quarter_behind_door": {
        "label": "REAR QUARTER",
        "display_name": "Rear quarter (behind door)",
        "aliases": [
            "rear quarter",
            "rear quarter panel",
            "quarter panel",
            "behind door",
            "behind the door",
            "sail panel",
            "c pillar",
            "c-pillar",
            "side panel behind door",
        ],
        "in_game_hint": "Horizontal panel between the side door and rear corner on the UV sheet.",
    },
}

NEW_REGION = {
    "id": "rear_quarter_behind_door",
    "label": "REAR QUARTER",
    "display_name": "Rear quarter (behind door)",
    "aliases": [
        "rear quarter",
        "rear quarter panel",
        "quarter panel",
        "behind door",
        "behind the door",
        "sail panel",
        "c pillar",
        "c-pillar",
        "side panel behind door",
    ],
    "in_game_hint": "Horizontal panel between the side door and rear corner on the UV sheet.",
}


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
    mask = np.array(Image.open(MASK).convert("L"))
    atlas = json.loads(ATLAS_JSON.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in atlas["regions"]}

    print("Refined bboxes from mask zones:")
    for region_id, zone in REGION_ZONES.items():
        bbox = bbox_from_mask_zone(mask, zone)
        if not bbox:
            print(f"  {region_id}: WARNING — empty zone {zone}")
            continue
        print(f"  {region_id}: {bbox}")

        if region_id not in by_id:
            defaults = NEW_REGION_DEFAULTS.get(region_id, NEW_REGION)
            entry = {**defaults, "id": region_id}
            cx = round((bbox[0] + bbox[2]) / 2, 4)
            cy = round((bbox[1] + bbox[3]) / 2, 4)
            entry["label_anchor"] = [cx, cy]
            by_id[region_id] = entry
        by_id[region_id]["bbox"] = bbox
        if region_id in LABEL_ANCHORS:
            by_id[region_id]["label_anchor"] = list(LABEL_ANCHORS[region_id])
        if region_id in REGION_METADATA:
            by_id[region_id].update(REGION_METADATA[region_id])

    preferred_order = [
        "front_bumper",
        "front_corner_upper_left",
        "front_corner_upper_bumper_right",
        "back",
        "rear_bumper",
        "rear_diffuser",
        "left_side_door",
        "rear_quarter_behind_door",
        "front_corner_upper_right",
        "rear_corner_upper",
        "trunk",
        "roof",
        "hood",
        "right_side_door",
        "front_corner_lower_right",
        "rear_corner_lower",
    ]
    atlas["regions"] = [by_id[rid] for rid in preferred_order if rid in by_id]
    atlas["version"] = atlas.get("version", 1) + 1

    groups = atlas.setdefault("alias_groups", {})
    groups["rear"] = [
        "back",
        "rear_bumper",
        "rear_diffuser",
        "trunk",
        "rear_corner_upper",
        "rear_corner_lower",
        "rear_quarter_behind_door",
    ]
    groups["rear_quarter"] = ["rear_quarter_behind_door", "rear_corner_upper", "rear_corner_lower"]
    groups["quarter"] = ["rear_quarter_behind_door", "rear_corner_upper", "rear_corner_lower"]
    groups["front"] = [
        "front_bumper",
        "front_corner_upper_left",
        "front_corner_upper_bumper_right",
        "front_corner_upper_right",
        "front_corner_lower_right",
    ]
    groups["left_side"] = [
        "left_side_door",
        "rear_corner_upper",
        "rear_corner_lower",
        "rear_quarter_behind_door",
        "trunk",
    ]
    groups["right_side"] = [
        "right_side_door",
        "hood",
        "front_corner_upper_bumper_right",
        "front_corner_upper_right",
        "front_corner_lower_right",
        "rear_quarter_behind_door",
        "back",
        "rear_bumper",
        "rear_diffuser",
    ]
    groups["bumper"] = [
        "front_bumper",
        "front_corner_upper_left",
        "front_corner_upper_bumper_right",
    ]
    groups["front_bumper_corner"] = [
        "front_corner_upper_left",
        "front_corner_upper_bumper_right",
    ]
    groups["corner"] = [
        "rear_corner_upper",
        "rear_corner_lower",
        "front_corner_upper_right",
        "front_corner_lower_right",
    ]
    groups["fender"] = [
        "rear_corner_upper",
        "rear_corner_lower",
        "front_corner_upper_right",
        "front_corner_lower_right",
    ]

    ATLAS_JSON.write_text(json.dumps(atlas, indent=2), encoding="utf-8")
    print(f"\nUpdated {ATLAS_JSON} v{atlas['version']} ({len(atlas['regions'])} regions)")

    uv_atlas = load_atlas_for_car(FOLDER)
    if uv_atlas is None:
        print("Atlas load failed — guides not regenerated")
        return

    mask_img = Image.open(MASK)
    pa_path = CACHE_DIR / "paintable_reference.png"
    paintable_reference = Image.open(pa_path) if pa_path.exists() else None
    labeled = build_labeled_guide(mask_img, uv_atlas, paintable_reference, 2048)
    ai_layout = build_ai_layout_guide(mask_img, uv_atlas, 2048)
    ai_gen = build_ai_generation_guide(mask_img, uv_atlas, 2048)
    labeled.save(CACHE_DIR / "labeled_guide.png")
    ai_layout.save(CACHE_DIR / "ai_layout_guide.png")
    ai_gen.save(CACHE_DIR / "ai_generation_guide.png")
    panels = sync_atlas_panel_regions(uv_atlas)
    (CACHE_DIR / "panels.json").write_text(json.dumps(panels, indent=2), encoding="utf-8")
    labeled.save(ROOT / "templates" / "atlas" / "stockcars2_arcaford25_labeled_guide.png")
    (CACHE_DIR / "version.txt").write_text(str(CACHE_VERSION), encoding="utf-8")
    print("Regenerated labeled_guide.png, ai_layout_guide.png, and ai_generation_guide.png")


if __name__ == "__main__":
    main()