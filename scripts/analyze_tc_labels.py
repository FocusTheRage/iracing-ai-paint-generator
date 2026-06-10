"""Analyze red labels on TC.png and propose zone layout."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
TC = ROOT / "templates" / "TC.png"
MASK = ROOT / "templates" / "cache" / "stockcars_toyotacamry2022" / "mask.png"
ATLAS = ROOT / "templates" / "atlas" / "stockcars_toyotacamry2022.json"
OUT = ROOT / "templates" / "atlas"


def red_blobs(img: np.ndarray) -> list[dict]:
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    red = (r > 200) & (g < 80) & (b < 80)
    h, w = red.shape
    visited = np.zeros(red.shape, bool)
    blobs: list[dict] = []
    for sy in range(h):
        for sx in range(w):
            if not red[sy, sx] or visited[sy, sx]:
                continue
            q = deque([(sx, sy)])
            visited[sy, sx] = True
            xs, ys = [], []
            while q:
                x, y = q.popleft()
                xs.append(x)
                ys.append(y)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and red[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if len(xs) < 30:
                continue
            blobs.append(
                {
                    "area": len(xs),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "cent": (float(np.mean(xs)), float(np.mean(ys))),
                }
            )
    blobs.sort(key=lambda b: (b["cent"][1], b["cent"][0]))
    return blobs


def bbox_from_zone(mask: np.ndarray, zone: tuple[float, float, float, float]) -> list[float] | None:
    paintable = mask > 128
    h, w = mask.shape
    zx0, zy0, zx1, zy1 = zone
    px0, py0 = max(0, int(zx0 * w)), max(0, int(zy0 * h))
    px1, py1 = min(w, int(zx1 * w)), min(h, int(zy1 * h))
    sub = paintable[py0:py1, px0:px1]
    if sub.sum() < 80:
        return None
    ys, xs = np.where(sub)
    return [
        round((px0 + int(xs.min())) / w, 4),
        round((py0 + int(ys.min())) / h, 4),
        round((px0 + int(xs.max()) + 1) / w, 4),
        round((py0 + int(ys.max()) + 1) / h, 4),
    ]


# TC.png user labels → zone ids (from visual reference)
TC_LAYOUT_ZONES: dict[str, tuple[float, float, float, float]] = {
    # Top row fascia (user: front bumper, rear)
    "front_corner_upper_left": (0.0, 0.0, 0.50, 0.14),
    "front_corner_upper_bumper_right": (0.50, 0.0, 0.87, 0.20),
    "front_bumper": (0.20, 0.13, 0.62, 0.25),
    "back": (0.86, 0.0, 0.995, 0.20),
    "rear_bumper": (0.86, 0.07, 0.995, 0.28),
    "rear_diffuser": (0.88, 0.14, 1.0, 0.34),
    # Accent panels above plan view
    "rear_corner_upper": (0.0, 0.13, 0.27, 0.27),
    "front_corner_upper_right": (0.40, 0.13, 0.53, 0.20),
    "rear_quarter_behind_door": (0.14, 0.26, 0.29, 0.36),
    # Plan view — user: left side (top), trunk|roof|hood (middle), right side (bottom)
    "left_side_door": (0.0, 0.24, 1.0, 0.28),
    "trunk": (0.0, 0.28, 0.34, 0.43),
    "roof": (0.34, 0.28, 0.58, 0.43),
    "hood": (0.58, 0.28, 1.0, 0.43),
    "right_side_door": (0.0, 0.43, 1.0, 0.52),
    # Plan view lower corners (small islands on silhouette)
    "rear_corner_lower": (0.09, 0.50, 0.17, 0.67),
    "front_corner_lower_right": (0.54, 0.50, 0.66, 0.76),
    # Lower half side profiles (unlabeled in TC but separate columns)
    "left_side_lower": (0.0, 0.52, 0.28, 0.89),
    "center_side_lower": (0.28, 0.52, 0.60, 0.89),
    "right_side_lower": (0.60, 0.52, 1.0, 0.93),
    # Bottom strips
    "side_skirt_left": (0.14, 0.72, 0.58, 0.79),
    "rear_bumper_lower": (0.15, 0.88, 0.29, 1.0),
    "front_splitter_right": (0.77, 0.93, 1.0, 1.0),
    "rear_corner_lower_left": (0.0, 0.71, 0.15, 1.0),
}

LABEL_ANCHORS = {
    "front_corner_upper_left": (0.24, 0.07),
    "front_corner_upper_bumper_right": (0.68, 0.09),
    "front_bumper": (0.35, 0.16),
    "back": (0.92, 0.10),
    "rear_bumper": (0.92, 0.18),
    "rear_diffuser": (0.95, 0.24),
    "rear_corner_upper": (0.13, 0.20),
    "front_corner_upper_right": (0.51, 0.18),
    "rear_quarter_behind_door": (0.22, 0.31),
    "left_side_door": (0.50, 0.26),
    "trunk": (0.17, 0.36),
    "roof": (0.46, 0.36),
    "hood": (0.79, 0.36),
    "right_side_door": (0.50, 0.47),
    "front_corner_lower_right": (0.60, 0.63),
    "rear_corner_lower": (0.13, 0.58),
    "left_side_lower": (0.14, 0.71),
    "center_side_lower": (0.44, 0.71),
    "right_side_lower": (0.80, 0.73),
    "side_skirt_left": (0.36, 0.76),
    "rear_bumper_lower": (0.22, 0.94),
    "front_splitter_right": (0.89, 0.97),
    "rear_corner_lower_left": (0.07, 0.86),
}


def main() -> None:
    img = np.array(Image.open(TC).convert("RGB"))
    h, w = img.shape[:2]
    blobs = red_blobs(img)
    print("Red label blobs:")
    for i, b in enumerate(blobs):
        cx, cy = b["cent"]
        x0, y0, x1, y1 = b["bbox"]
        print(
            f"  #{i:2d} area={b['area']:5d} cent=({cx/w:.3f},{cy/h:.3f}) "
            f"bbox=({x0/w:.3f},{y0/h:.3f},{(x1+1)/w:.3f},{(y1+1)/h:.3f})"
        )

    mask = np.array(Image.open(MASK).convert("L"))
    print("\nTC layout zone refinement:")
    for rid, zone in TC_LAYOUT_ZONES.items():
        bbox = bbox_from_zone(mask, zone)
        print(f"  {rid}: {bbox}")

    # Overlay proposed zones on TC.png
    tc = Image.open(TC).convert("RGBA")
    draw = ImageDraw.Draw(tc)
    for rid, zone in TC_LAYOUT_ZONES.items():
        x0, y0, x1, y1 = zone
        px = [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)]
        draw.rectangle(px, outline=(255, 0, 255, 255), width=2)
        ax, ay = LABEL_ANCHORS[rid]
        draw.text((int(ax * w), int(ay * h)), rid.replace("_", " ")[:12], fill=(255, 255, 0, 255))
    tc.save(OUT / "TC_zone_proposed.png")
    print(f"\nWrote {OUT / 'TC_zone_proposed.png'}")


if __name__ == "__main__":
    main()