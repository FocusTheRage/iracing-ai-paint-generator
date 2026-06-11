"""
Derive Camaro ZL1 Gen 6 atlas zones from user-labeled wireframe anchors.

Each label position defines a search window; bbox = tight mask bounds in that window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / "templates" / "wireframes" / "stockcars_camarozl12018" / "wireframe.png"
MASK = ROOT / "templates" / "cache" / "stockcars_camarozl12018" / "mask.png"
OUT = ROOT / "templates" / "atlas" / "camaro_zone_debug.png"

# Anchors from templates/wireframes/stockcars_camarozl12018/wireframe.png
ANCHORS: dict[str, tuple[float, float]] = {
    "front_corner_upper_left": (0.067, 0.118),
    "front_bumper": (0.224, 0.092),
    "front_corner_upper_bumper_right": (0.426, 0.114),
    "back": (0.649, 0.068),
    "rear_diffuser": (0.642, 0.155),
    "rear_corner_lower": (0.077, 0.255),
    "trunk": (0.220, 0.308),
    "front_corner_upper_right": (0.423, 0.276),
    "left_side_door": (0.728, 0.331),
    "roof": (0.233, 0.577),
    "hood": (0.803, 0.593),
    "right_side_door": (0.209, 0.884),
    "front_corner_lower_right": (0.736, 0.855),
    "right_lower_door": (0.479, 0.839),
    "front_splitter_right": (0.472, 0.934),
    "pit_sign": (0.226, 0.945),
}

# Search half-extents (fraction of image) tuned per region from labeled wireframe.
WINDOWS: dict[str, tuple[float, float]] = {
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


def bbox_in_window(
    mask: np.ndarray,
    anchor: tuple[float, float],
    half: tuple[float, float],
    *,
    min_pixels: int = 80,
) -> list[float] | None:
    h, w = mask.shape
    ax, ay = anchor
    hx, hy = half
    px0 = max(0, int((ax - hx) * w))
    py0 = max(0, int((ay - hy) * h))
    px1 = min(w, int((ax + hx) * w))
    py1 = min(h, int((ay + hy) * h))
    sub = mask[py0:py1, px0:px1] > 128
    if sub.sum() < min_pixels:
        return None
    ys, xs = np.where(sub)
    return [
        round((px0 + int(xs.min())) / w, 4),
        round((py0 + int(ys.min())) / h, 4),
        round((px0 + int(xs.max()) + 1) / w, 4),
        round((py0 + int(ys.max()) + 1) / h, 4),
    ]


def main() -> None:
    mask = np.array(Image.open(MASK).convert("L"))
    h, w = mask.shape
    debug = Image.open(MASK).convert("RGB")
    draw = ImageDraw.Draw(debug)
    colors = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 255, 80),
        (255, 80, 255),
        (80, 255, 255),
    ]

    print("Camaro ZL1 zones from wireframe anchors:")
    for i, (rid, anchor) in enumerate(ANCHORS.items()):
        half = WINDOWS[rid]
        bbox = bbox_in_window(mask, anchor, half)
        if not bbox:
            print(f"  {rid}: EMPTY window {half}")
            continue
        x0, y0, x1, y1 = bbox
        px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        sub = mask[py0:py1, px0:px1]
        cov = (sub > 128).mean()
        print(f"  {rid:30} {bbox}  cov={cov:.1%}  window={half}")
        c = colors[i % len(colors)]
        draw.rectangle((px0, py0, px1, py1), outline=c, width=3)
        sx, sy = int(anchor[0] * w), int(anchor[1] * h)
        draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), fill=c)

    debug.save(OUT)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()