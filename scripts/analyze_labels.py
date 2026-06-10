"""Analyze red + black label clusters in user PNG."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

PNG = Path(r"D:\iracing-ai-paint-generator\templates\atlas\ARCA Ford Mustang.png")
MASK = Path(r"D:\iracing-ai-paint-generator\templates\cache\stockcars2_arcaford25\mask.png")


def brown_background(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1], arr[:, :, 2]
    return (r < 120) & (g < 100) & (b < 100) & (r > 40)


def label_pixels(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red = (r > 200) & (g < 80) & (b < 80)
    brown = brown_background(arr)
    black = (r < 45) & (g < 45) & (b < 45) & brown
    return red, black


def find_blobs(bitmap: np.ndarray, min_area: int = 30) -> list[dict]:
    h, w = bitmap.shape
    visited = np.zeros_like(bitmap, dtype=bool)
    blobs = []
    for sy in range(h):
        for sx in range(w):
            if bitmap[sy, sx] and not visited[sy, sx]:
                q = deque([(sx, sy)])
                visited[sy, sx] = True
                xs, ys = [sx], [sy]
                while q:
                    cx, cy = q.popleft()
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and bitmap[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                            xs.append(nx)
                            ys.append(ny)
                if len(xs) >= min_area:
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
                    blobs.append({
                        "bbox": (x0, y0, x1, y1),
                        "center": ((x0 + x1) // 2, (y0 + y1) // 2),
                        "area": len(xs),
                    })
    return blobs


def cluster_blobs(blobs: list[dict], y_thresh: int = 26, x_gap: int = 85) -> list[dict]:
    lines: list[list[dict]] = []
    for blob in sorted(blobs, key=lambda b: b["center"][1]):
        cy = blob["center"][1]
        for line in lines:
            if abs(cy - line[0]["center"][1]) <= y_thresh:
                line.append(blob)
                break
        else:
            lines.append([blob])
    clusters = []
    for line in lines:
        line.sort(key=lambda b: b["center"][0])
        group = [line[0]]
        for blob in line[1:]:
            if blob["bbox"][0] - group[-1]["bbox"][2] <= x_gap:
                group.append(blob)
            else:
                xs = [b["bbox"][0] for b in group] + [b["bbox"][2] for b in group]
                ys = [b["bbox"][1] for b in group] + [b["bbox"][3] for b in group]
                clusters.append(_cluster_from(group))
                group = [blob]
        clusters.append(_cluster_from(group))
    return sorted(clusters, key=lambda c: (c["center"][1], c["center"][0]))


def _cluster_from(group: list[dict]) -> dict:
    xs = [b["bbox"][0] for b in group] + [b["bbox"][2] for b in group]
    ys = [b["bbox"][1] for b in group] + [b["bbox"][3] for b in group]
    return {
        "bbox": (min(xs), min(ys), max(xs), max(ys)),
        "center": ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2),
        "w": max(xs) - min(xs),
        "h": max(ys) - min(ys),
    }


def classify_black(cx: int, cy: int, w: int, h: int) -> str:
    nx, ny = cx / w, cy / h
    if ny < 0.14 and nx > 0.78 and nx < 0.92:
        return "back"
    if ny < 0.20 and nx > 0.88:
        return "rear_bumper"
    if 0.14 < ny < 0.22 and nx > 0.90:
        return "rear_diffuser"
    if 0.28 < ny < 0.40 and nx > 0.60:
        return "front_corner_upper"
    if ny > 0.80 and nx > 0.60:
        return "front_corner_lower"
    if ny < 0.18 and 0.60 < nx < 0.78:
        return "back"
    return "unknown_black"


def nearest_island(lx: float, ly: float, islands: list[dict], used: set[int]) -> dict | None:
    best_idx, best_score = None, 1e9
    for idx, isl in enumerate(islands):
        if idx in used:
            continue
        ib, icx, icy = isl["bbox"], isl["center"][0], isl["center"][1]
        inside = ib[0] <= lx <= ib[2] and ib[1] <= ly <= ib[3]
        dist = ((lx - icx) ** 2 + (ly - icy) ** 2) ** 0.5
        score = dist - (0.4 if inside else 0)
        if score < best_score:
            best_score, best_idx = score, idx
    if best_idx is None:
        return None
    used.add(best_idx)
    return islands[best_idx]


def mask_islands(mask: np.ndarray, min_area: int = 2500, mega: int = 400_000) -> list[dict]:
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
    return [i for i in islands if i["area"] < mega]


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from import_atlas_from_png import classify_dark, find_dark_clusters
    arr = np.array(Image.open(PNG).convert("RGB"))
    h, w = arr.shape[:2]
    for c in find_dark_clusters(arr):
        cx, cy = c["center"]
        rid = classify_dark(cx, cy, w, h, c)
        print(f"({cx/w:.4f},{cy/h:.4f}) w={c['w']} h={c['h']} -> {rid}")
    return

    # legacy below
    arr = np.array(Image.open(PNG).convert("RGB"))
    mask = np.array(Image.open(MASK).convert("L"))
    h, w = arr.shape[:2]
    red_m, black_m = label_pixels(arr)
    red_clusters = cluster_blobs(find_blobs(red_m))
    black_raw = find_blobs(black_m, min_area=20)
    # Text-like black blobs only
    black_text_blobs = [
        b for b in black_raw
        if b["bbox"][3] - b["bbox"][1] <= 40 and b["bbox"][2] - b["bbox"][0] >= 30
        and b["area"] < 5000
    ]
    black_clusters = cluster_blobs(black_text_blobs, y_thresh=22, x_gap=60)
    black_clusters = [c for c in black_clusters if c["h"] <= 45 and c["w"] >= 40]

    islands = mask_islands(mask)
    used: set[int] = set()

    print(f"Red clusters ({len(red_clusters)}):")
    for i, c in enumerate(red_clusters):
        cx, cy = c["center"]
        print(f"  {i}: ({cx/w:.4f}, {cy/h:.4f})")

    print(f"\nBlack text clusters ({len(black_clusters)}):")
    for i, c in enumerate(black_clusters):
        cx, cy = c["center"]
        rid = classify_black(cx, cy, w, h)
        lx, ly = cx / w, (c["bbox"][3] + 8) / h
        isl = nearest_island(lx, min(ly, 0.99), islands, set())  # don't consume yet
        print(f"  {i}: ({cx/w:.4f}, {cy/h:.4f}) -> {rid} island={isl['bbox'] if isl else None}")


if __name__ == "__main__":
    main()