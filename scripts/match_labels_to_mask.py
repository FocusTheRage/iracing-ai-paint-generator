"""Match user PNG red labels to PSD mask islands (exclude mega-blobs)."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PNG = ROOT / "templates" / "atlas" / "ARCA Ford Mustang.png"
MASK = ROOT / "templates" / "cache" / "stockcars2_arcaford25" / "mask.png"
JSON = ROOT / "templates" / "atlas" / "stockcars2_arcaford25.json"

MEGA_AREA = 400_000


def find_red_blobs(arr: np.ndarray, min_area: int = 40) -> list[dict]:
    h, w = arr.shape[:2]
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    visited = np.zeros_like(red, dtype=bool)
    blobs = []
    for sy in range(h):
        for sx in range(w):
            if red[sy, sx] and not visited[sy, sx]:
                q = deque([(sx, sy)])
                visited[sy, sx] = True
                xs, ys = [sx], [sy]
                while q:
                    cx, cy = q.popleft()
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and red[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                            xs.append(nx)
                            ys.append(ny)
                if len(xs) >= min_area:
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
                    blobs.append({
                        "bbox": (x0, y0, x1, y1),
                        "center": ((x0 + x1) // 2, (y0 + y1) // 2),
                    })
    return blobs


def cluster_blobs(blobs: list[dict], y_thresh: int = 30, x_gap: int = 80) -> list[dict]:
    lines: list[list[dict]] = []
    for blob in sorted(blobs, key=lambda b: b["center"][1]):
        cy = blob["center"][1]
        placed = False
        for line in lines:
            if abs(cy - line[0]["center"][1]) <= y_thresh:
                line.append(blob)
                placed = True
                break
        if not placed:
            lines.append([blob])
    clusters = []
    for line in lines:
        line.sort(key=lambda b: b["center"][0])
        group = [line[0]]
        for blob in line[1:]:
            prev = group[-1]
            if blob["bbox"][0] - prev["bbox"][2] <= x_gap:
                group.append(blob)
            else:
                xs = [b["bbox"][0] for b in group] + [b["bbox"][2] for b in group]
                ys = [b["bbox"][1] for b in group] + [b["bbox"][3] for b in group]
                clusters.append({
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "center": ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2),
                })
                group = [blob]
        xs = [b["bbox"][0] for b in group] + [b["bbox"][2] for b in group]
        ys = [b["bbox"][1] for b in group] + [b["bbox"][3] for b in group]
        clusters.append({
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "center": ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2),
        })
    return sorted(clusters, key=lambda c: (c["center"][1], c["center"][0]))


def mask_islands(mask: np.ndarray, min_area: int = 3000) -> list[dict]:
    h, w = mask.shape
    paintable = mask > 128
    visited = np.zeros_like(paintable, dtype=bool)
    islands = []
    for sy in range(h):
        for sx in range(w):
            if paintable[sy, sx] and not visited[sy, sx]:
                q = deque([(sx, sy)])
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
    return islands


# Label clusters sorted by Y — assign semantic names in layout order
LABEL_ORDER = [
    "front_bumper",      # top
    "left_side_door",    # upper-mid
    "rear_corner_upper", # left-mid
    "trunk",             # left-mid lower (or roof area left)
    "roof",              # center
    "hood",              # right-mid
    "rear_corner_lower", # bottom-left
    "right_side_door",   # bottom-center
]


def bbox_from_seed(white: np.ndarray, fx: float, fy: float, limit: int) -> list[float]:
    h, w = white.shape
    x0, y0 = int(fx * w), int(fy * h)
    seed = None
    for radius in range(150):
        for dy in range(-radius, radius + 1, max(1, radius // 4 or 1)):
            for dx in range(-radius, radius + 1, max(1, radius // 4 or 1)):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < w and 0 <= y < h and white[y, x]:
                    seed = (x, y)
                    break
            if seed:
                break
        if seed:
            break
    if seed is None:
        return []

    visited = {seed}
    queue: deque[tuple[int, int]] = deque([seed])
    xs = [seed[0]]
    ys = [seed[1]]
    while queue and len(visited) < limit:
        cx, cy = queue.popleft()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < w and 0 <= ny < h and white[ny, nx] and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
                xs.append(nx)
                ys.append(ny)

    return [
        round(min(xs) / w, 4),
        round(min(ys) / h, 4),
        round((max(xs) + 1) / w, 4),
        round((max(ys) + 1) / h, 4),
    ]


def main() -> None:
    arr = np.array(Image.open(PNG).convert("RGB"))
    mask = np.array(Image.open(MASK).convert("L"))
    h, w = arr.shape[:2]
    white = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)

    blobs = find_red_blobs(arr)
    clusters = cluster_blobs(blobs)
    islands = [i for i in mask_islands(mask) if i["area"] < MEGA_AREA]

    print(f"Non-mega mask islands ({len(islands)}):")
    for i, isl in enumerate(sorted(islands, key=lambda x: -x["area"])):
        print(f"  {i}: area={isl['area']} center={isl['center']} bbox={isl['bbox']}")

    print(f"\nLabel clusters ({len(clusters)}):")
    for i, c in enumerate(clusters):
        cx, cy = c["center"]
        print(f"  {i}: center=({cx/w:.4f},{cy/h:.4f})")

    # Merge front bumper clusters (first two on same line)
    merged: list[tuple[str, tuple[float, float]]] = []
    ci = 0
    if len(clusters) >= 2 and abs(clusters[0]["center"][1] - clusters[1]["center"][1]) < 30:
        cx = (clusters[0]["center"][0] + clusters[1]["center"][0]) / 2
        cy = (clusters[0]["center"][1] + clusters[1]["center"][1]) / 2
        merged.append(("front_bumper", (cx / w, cy / h)))
        ci = 2
    for j, name in enumerate(LABEL_ORDER[1:], start=1):
        if ci < len(clusters):
            c = clusters[ci]
            merged.append((name, (c["center"][0] / w, c["center"][1] / h)))
            ci += 1

    print("\nMerged label -> seed positions:")
    for name, (fx, fy) in merged:
        print(f"  {name}: ({fx:.4f}, {fy:.4f})")

    # Flood fill limits per region (tuned)
    limits = {
        "front_bumper": 80_000,
        "left_side_door": 90_000,
        "rear_corner_upper": 70_000,
        "trunk": 80_000,
        "roof": 120_000,
        "hood": 120_000,
        "rear_corner_lower": 70_000,
        "right_side_door": 90_000,
    }

    print("\nFlood-fill bboxes from label seeds on user PNG:")
    for name, (fx, fy) in merged:
        bb = bbox_from_seed(white, fx, fy + 0.02, limits[name])  # probe below label
        print(f"  {name}: {bb}")

    # Also try mask island nearest (non-mega) below each label
    print("\nNearest non-mega mask island below each label:")
    used: set[int] = set()
    for name, (fx, fy) in merged:
        lx, ly = fx, fy + 0.03
        best = None
        best_score = 1e9
        for idx, isl in enumerate(islands):
            if idx in used:
                continue
            ib = isl["bbox"]
            icx, icy = isl["center"]
            inside = ib[0] <= lx <= ib[2] and ib[1] <= ly <= ib[3]
            dist = ((lx - icx) ** 2 + (ly - icy) ** 2) ** 0.5
            score = dist - (0.5 if inside else 0)
            if score < best_score:
                best_score = score
                best = (idx, isl)
        if best:
            used.add(best[0])
            print(f"  {name}: island {best[0]} {best[1]['bbox']}")


if __name__ == "__main__":
    main()