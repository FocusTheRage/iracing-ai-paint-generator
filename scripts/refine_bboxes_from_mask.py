"""Refine atlas region bboxes using PSD mask islands + label anchors."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASK = ROOT / "templates" / "cache" / "stockcars2_arcaford25" / "mask.png"
ATLAS_JSON = ROOT / "templates" / "atlas" / "stockcars2_arcaford25.json"
MEGA_AREA = 400_000


def mask_islands(mask: np.ndarray, min_area: int = 2500) -> list[dict]:
    h, w = mask.shape
    paintable = mask > 128
    visited = np.zeros_like(paintable, dtype=bool)
    islands: list[dict] = []
    for sy in range(h):
        for sx in range(w):
            if paintable[sy, sx] and not visited[sy, sx]:
                q: deque[tuple[int, int]] = deque([(sx, sy)])
                visited[sy, sx] = True
                xs, ys = [sx], [sy]
                while q:
                    cx, cy = q.popleft()
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and paintable[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                            xs.append(nx)
                            ys.append(ny)
                if len(xs) >= min_area:
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
                    islands.append({
                        "bbox": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
                        "center": (round((x0 + x1) / 2 / w, 4), round((y0 + y1) / 2 / h, 4)),
                        "area": len(xs),
                    })
    return [i for i in islands if i["area"] < MEGA_AREA]


def overlap_area(a: list[float], b: list[float]) -> float:
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ox * oy


def point_in_bbox(px: float, py: float, bbox: list[float]) -> bool:
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]


def main() -> None:
    atlas = json.loads(ATLAS_JSON.read_text(encoding="utf-8"))
    mask = np.array(Image.open(MASK).convert("L"))
    islands = sorted(mask_islands(mask), key=lambda x: -x["area"])

    print(f"Found {len(islands)} non-mega mask islands\n")

    # Region -> islands containing anchor
    print("ANCHOR-BASED island assignment:")
    used: set[int] = set()
    for region in atlas["regions"]:
        ax, ay = region.get("label_anchor", [0, 0])
        if ax <= 0 and ay <= 0:
            continue
        containing = []
        for idx, isl in enumerate(islands):
            if point_in_bbox(ax, ay, isl["bbox"]):
                containing.append((idx, isl))
        if containing:
            best = max(containing, key=lambda x: x[1]["area"])
            print(f"  {region['id']}: anchor=({ax},{ay}) -> island {best[0]} bbox={best[1]['bbox']}")
        else:
            best_idx = min(
                range(len(islands)),
                key=lambda i: ((ax - islands[i]["center"][0]) ** 2 + (ay - islands[i]["center"][1]) ** 2) ** 0.5,
            )
            print(f"  {region['id']}: anchor=({ax},{ay}) -> nearest island {best_idx} bbox={islands[best_idx]['bbox']}")

    print("\nREGION current bbox vs best overlapping islands:")
    for region in atlas["regions"]:
        rb = region["bbox"]
        hits = [(i, isl) for i, isl in enumerate(islands) if overlap_area(rb, isl["bbox"]) > 0.0001]
        hits.sort(key=lambda x: -x[1]["area"])
        print(f"  {region['id']}: {rb}")
        for i, isl in hits[:4]:
            print(f"    island {i}: area={isl['area']} bbox={isl['bbox']}")

    assigned: set[int] = set()
    for region in atlas["regions"]:
        rb = region["bbox"]
        for idx, isl in enumerate(islands):
            if overlap_area(rb, isl["bbox"]) > 0.0008:
                assigned.add(idx)

    print("\nUNASSIGNED islands (area >= 4000):")
    for idx, isl in enumerate(islands):
        if idx not in assigned and isl["area"] >= 4000:
            print(f"  {idx}: area={isl['area']} center={isl['center']} bbox={isl['bbox']}")


if __name__ == "__main__":
    main()