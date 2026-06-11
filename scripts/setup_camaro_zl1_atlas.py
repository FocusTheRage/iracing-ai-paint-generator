"""
Build UV atlas + labeled guides for NASCAR Cup Series Chevrolet Camaro ZL1 [Gen 6].

Uses the user-labeled wireframe at:
  templates/wireframes/stockcars_camarozl12018/wireframe.png
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from template_manager import (  # noqa: E402
    CACHE_VERSION,
    bundled_wireframe_path,
    cyan_wire_overlay_from_image,
    get_car_template,
)
from cars_config import CAR_BY_NAME  # noqa: E402
from uv_atlas import (  # noqa: E402
    build_ai_generation_guide,
    build_ai_layout_guide,
    build_labeled_guide,
    load_atlas_for_car,
    sync_atlas_panel_regions,
)

FOLDER = r"stockcars\camarozl12018"
SLUG = "stockcars_camarozl12018"
CAR_NAME = "NASCAR Cup Series Chevrolet Camaro ZL1 [Gen 6]"
CACHE_DIR = ROOT / "templates" / "cache" / SLUG
ATLAS_JSON = ROOT / "templates" / "atlas" / f"{SLUG}.json"
WIREFRAME_SRC = ROOT / "templates" / "wireframes" / SLUG / "wireframe.png"
WIREFRAME_REF = ROOT / "templates" / "atlas" / f"{SLUG}_wireframe.png"

# Label anchors from templates/wireframes/stockcars_camarozl12018/wireframe.png.
# Camaro ZL1 Gen 6 uses its own UV layout (not the Camry grid).
LABEL_ANCHORS: dict[str, tuple[float, float]] = {
    "front_corner_upper_left": (0.067, 0.118),
    "front_bumper": (0.224, 0.092),
    "front_corner_upper_bumper_right": (0.426, 0.114),
    "back": (0.649, 0.068),
    "rear_diffuser": (0.642, 0.155),
    "front_corner_upper_right": (0.423, 0.276),
    "left_side_door": (0.728, 0.331),
    "trunk": (0.220, 0.308),
    "roof": (0.233, 0.577),
    "hood": (0.803, 0.593),
    "rear_corner_lower": (0.077, 0.255),
    "right_side_door": (0.209, 0.884),
    "front_corner_lower_right": (0.736, 0.855),
    "pit_sign": (0.226, 0.945),
    "right_lower_door": (0.479, 0.839),
    "front_splitter_right": (0.472, 0.934),
}

# Search half-extents around each label on the user wireframe.
SEARCH_WINDOWS: dict[str, tuple[float, float]] = {
    "front_corner_upper_left": (0.12, 0.14),
    "front_bumper": (0.22, 0.12),
    "front_corner_upper_bumper_right": (0.12, 0.14),
    "back": (0.18, 0.12),
    "rear_diffuser": (0.14, 0.14),
    "rear_corner_lower": (0.14, 0.16),
    "trunk": (0.20, 0.16),
    "front_corner_upper_right": (0.16, 0.14),
    "left_side_door": (0.18, 0.14),
    "roof": (0.18, 0.16),
    "hood": (0.20, 0.16),
    "right_side_door": (0.20, 0.10),
    "front_corner_lower_right": (0.16, 0.12),
    "right_lower_door": (0.22, 0.10),
    "front_splitter_right": (0.28, 0.08),
    "pit_sign": (0.18, 0.08),
}

REGION_META: dict[str, dict] = {
    "front_corner_upper_left": {
        "label": "B CORNER LEFT",
        "display_name": "Bumper corner (left)",
        "aliases": ["bumper corner left", "b corner left", "front bumper corner left"],
        "in_game_hint": "Top-left bumper corner cap.",
    },
    "front_bumper": {
        "label": "FRONT OF CAR",
        "display_name": "Front bumper",
        "aliases": ["front of car", "front bumper", "front", "nose", "grille"],
        "in_game_hint": "Top-left — front bumper / nose.",
    },
    "front_corner_upper_bumper_right": {
        "label": "B CORNER RIGHT",
        "display_name": "Bumper corner (right)",
        "aliases": ["bumper corner right", "b corner right", "front bumper corner right"],
        "in_game_hint": "Top-center bumper corner cap.",
    },
    "back": {
        "label": "REAR/BACK",
        "display_name": "Rear body",
        "aliases": ["rear/back", "back", "rear", "rear body", "rear clip"],
        "in_game_hint": "Top-right rear body panel.",
    },
    "rear_diffuser": {
        "label": "REAR BUMPER",
        "display_name": "Rear bumper",
        "aliases": ["rear bumper", "bumper", "diffuser", "rear diffuser"],
        "in_game_hint": "Below rear/back — rear bumper cover.",
    },
    "front_corner_upper_right": {
        "label": "LEFT FRONT CORNER",
        "display_name": "Left front quarter",
        "aliases": [
            "left front corner",
            "left front quarter",
            "front left corner",
            "front left quarter",
        ],
        "in_game_hint": "Upper accent — left front quarter panel.",
    },
    "left_side_door": {
        "label": "LEFT DOOR",
        "display_name": "Left door",
        "aliases": ["left door", "left side door", "driver door", "driver side door"],
        "in_game_hint": "Plan view — driver door.",
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
        "label": "LEFT REAR CORNER",
        "display_name": "Left rear quarter",
        "aliases": [
            "left rear corner",
            "left rear quarter",
            "rear quarter",
            "rear corner",
        ],
        "in_game_hint": "Left rear quarter panel.",
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
        "in_game_hint": "Lower plan view — passenger door.",
    },
    "front_corner_lower_right": {
        "label": "RIGHT FRONT CORNER",
        "display_name": "Right front quarter",
        "aliases": [
            "right front corner",
            "right front quarter",
            "front right corner",
            "front right quarter",
        ],
        "in_game_hint": "Lower-right — right front quarter.",
    },
    "pit_sign": {
        "label": "PIT STALL SIGN",
        "display_name": "Pit stall sign",
        "aliases": ["pit stall sign", "pit sign", "pit box sign"],
        "in_game_hint": "Bottom-left pit stall sign panel.",
    },
    "right_lower_door": {
        "label": "RIGHT LOWER DOOR",
        "display_name": "Right lower door",
        "aliases": ["right lower door", "lower right door"],
        "in_game_hint": "Bottom-center lower door / rocker area.",
    },
    "front_splitter_right": {
        "label": "FRONT SPLITTER",
        "display_name": "Front splitter",
        "aliases": ["front splitter", "splitter", "front lip"],
        "in_game_hint": "Bottom strip — front splitter.",
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


def bbox_from_anchor_window(
    mask: np.ndarray,
    anchor: tuple[float, float],
    half: tuple[float, float],
    min_pixels: int = 80,
) -> list[float]:
    h, w = mask.shape
    ax, ay = anchor
    hx, hy = half
    px0 = max(0, int((ax - hx) * w))
    py0 = max(0, int((ay - hy) * h))
    px1 = min(w, int((ax + hx) * w))
    py1 = min(h, int((ay + hy) * h))
    sub = mask[py0:py1, px0:px1] > 128
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
    if not WIREFRAME_SRC.exists():
        raise SystemExit(f"Missing labeled wireframe: {WIREFRAME_SRC}")

    car = CAR_BY_NAME[CAR_NAME]
    mask_path = CACHE_DIR / "mask.png"
    if not mask_path.exists():
        get_car_template(car, force_refresh=True)
    else:
        get_car_template(car)
    if not mask_path.exists():
        raise SystemExit(f"Template cache missing mask: {mask_path}")

    shutil.copy2(WIREFRAME_SRC, WIREFRAME_REF)
    print(f"Copied wireframe reference -> {WIREFRAME_REF}")

    mask = np.array(Image.open(mask_path).convert("L"))
    regions: dict[str, dict] = {}

    print("Camaro ZL1 Gen 6 refined bboxes:")
    for region_id in PREFERRED_ORDER:
        anchor = LABEL_ANCHORS[region_id]
        half = SEARCH_WINDOWS[region_id]
        bbox = bbox_from_anchor_window(mask, anchor, half)
        if not bbox:
            print(f"  {region_id}: WARNING empty window anchor={anchor} half={half}")
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
        "car_name": CAR_NAME,
        "folder_path": FOLDER,
        "resolution": 2048,
        "version": 1,
        "reference_png": WIREFRAME_REF.name,
        "layout_lines": [
            "TOP = FRONT OF CAR + B CORNERS + REAR/BACK + REAR BUMPER",
            "UPPER LEFT = LEFT REAR CORNER + TRUNK",
            "UPPER CENTER = LEFT FRONT CORNER",
            "UPPER RIGHT = LEFT DOOR",
            "MID = ROOF (left-center) + HOOD (right)",
            "LOWER = RIGHT DOOR + RIGHT LOWER DOOR + RIGHT FRONT CORNER",
            "BOTTOM = PIT STALL SIGN + FRONT SPLITTER",
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
                "rear_diffuser",
            ],
            "corner": ["rear_corner_lower", "front_corner_upper_right", "front_corner_lower_right"],
            "fender": ["rear_corner_lower", "front_corner_upper_right", "front_corner_lower_right"],
            "diffuser": ["rear_diffuser"],
            "rear_quarter": ["rear_corner_lower"],
            "quarter": [
                "rear_corner_lower",
                "front_corner_upper_right",
                "front_corner_lower_right",
            ],
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
        raise SystemExit("Atlas load failed after write")

    mask_img = Image.open(mask_path)
    pa_path = CACHE_DIR / "paintable_reference.png"
    paintable_reference = Image.open(pa_path) if pa_path.exists() else None

    bundled = bundled_wireframe_path(car)
    if bundled is not None:
        wire = cyan_wire_overlay_from_image(bundled, (2048, 2048))
        wire.save(CACHE_DIR / "wire.png")
        print(f"Updated wire.png from {bundled}")

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
    print("Regenerated guides, wire, and panels.json")


if __name__ == "__main__":
    main()