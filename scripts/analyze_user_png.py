"""Analyze red labels and white islands in user's ARCA Ford Mustang.png."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PNG = ROOT / "templates" / "atlas" / "ARCA Ford Mustang.png"
JSON = ROOT / "templates" / "atlas" / "stockcars2_arcaford25.json"


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
                        "area": len(xs),
                    })
    return blobs


def cluster_blobs(blobs: list[dict], y_thresh: int = 25, x_thresh: int = 120) -> list[dict]:
    """Merge letter blobs into label phrases."""
    if not blobs:
        return []
    sorted_b = sorted(blobs, key=lambda b: (b["center"][1], b["center"][0]))
    clusters: list[dict] = []
    for blob in sorted_b:
        cx, cy = blob["center"]
        matched = None
        for cluster in clusters:
            ccx, ccy = cluster["center"]
            if abs(cy - ccy) <= y_thresh and abs(cx - ccx) <= x_thresh:
                matched = cluster
                break
        if matched is None:
            clusters.append({
                "blobs": [blob],
                "center": blob["center"],
                "bbox": blob["bbox"],
            })
        else:
            matched["blobs"].append(blob)
            xs = [b["bbox"][0] for b in matched["blobs"]] + [b["bbox"][2] for b in matched["blobs"]]
            ys = [b["bbox"][1] for b in matched["blobs"]] + [b["bbox"][3] for b in matched["blobs"]]
            matched["bbox"] = (min(xs), min(ys), max(xs), max(ys))
            matched["center"] = ((matched["bbox"][0] + matched["bbox"][2]) // 2,
                                 (matched["bbox"][1] + matched["bbox"][3]) // 2)
    return sorted(clusters, key=lambda c: (c["center"][1], c["center"][0]))


def find_white_islands(white: np.ndarray, min_area: int = 5000) -> list[dict]:
    h, w = white.shape
    visited = np.zeros_like(white, dtype=bool)
    islands = []
    for sy in range(h):
        for sx in range(w):
            if white[sy, sx] and not visited[sy, sx]:
                q = deque([(sx, sy)])
                visited[sy, sx] = True
                xs, ys = [sx], [sy]
                while q:
                    cx, cy = q.popleft()
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and white[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                            xs.append(nx)
                            ys.append(ny)
                if len(xs) >= min_area:
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
                    islands.append({
                        "bbox": [x0 / w, y0 / h, x1 / w, y1 / h],
                        "center": ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h),
                        "area": len(xs),
                        "pixels": len(xs),
                    })
    islands.sort(key=lambda i: -i["area"])
    return islands


def island_for_label(cluster: dict, islands: list[dict], used: set[int]) -> tuple[int, dict] | tuple[None, None]:
    cx, cy = cluster["center"]
    x0, y0, x1, y1 = cluster["bbox"]
    # Prefer island whose bbox contains point below label, or nearest center
    probe_y = min(y1 + 8, 2047)
    best_idx = None
    best_score = float("inf")
    for idx, island in enumerate(islands):
        if idx in used:
            continue
        ib = island["bbox"]
        icx, icy = island["center"]
        # Check if probe point (center x, below label) is inside island bbox
        px, py = cx / 2048, probe_y / 2048
        inside = ib[0] <= px <= ib[2] and ib[1] <= py <= ib[3]
        dist = ((px - icx) ** 2 + (py - icy) ** 2) ** 0.5
        score = dist - (0.5 if inside else 0)
        if score < best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return None, None
    return best_idx, islands[best_idx]


def classify_cluster(cluster: dict) -> str:
    """Guess label from position (y,x ordering matches user's layout)."""
    cx, cy = cluster["center"]
    nx, ny = cx / 2048, cy / 2048
    # Position-based classification tuned to user's PNG
    if ny < 0.18 and 0.35 < nx < 0.65:
        return "front_bumper"
    if ny < 0.32 and 0.40 < nx < 0.60:
        return "left_side_door"
    if ny < 0.42 and nx < 0.22:
        return "rear_corner_upper"
    if 0.50 < ny < 0.65 and 0.30 < nx < 0.50:
        return "roof"
    if 0.50 < ny < 0.65 and nx > 0.75:
        return "hood"
    if 0.78 < ny < 0.87 and nx < 0.22:
        return "trunk" if nx < 0.12 else "rear_corner_lower"
    if ny > 0.85 and 0.40 < nx < 0.65:
        return "right_side_door"
    return "unknown"


def main() -> None:
    arr = np.array(Image.open(PNG).convert("RGB"))
    h, w = arr.shape[:2]
    print(f"Size: {w}x{h}")

    blobs = find_red_blobs(arr)
    clusters = cluster_blobs(blobs)
    print(f"\nLabel clusters ({len(clusters)}):")
    for i, c in enumerate(clusters):
        cx, cy = c["center"]
        guess = classify_cluster(c)
        print(f"  {i}: center ({cx/w:.4f},{cy/h:.4f}) guess={guess} bbox={[round(x/w,4) for x in c['bbox']]}")

    white = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)
    islands = find_white_islands(white)
    print(f"\nWhite islands ({len(islands)}):")
    for i, isl in enumerate(islands):
        print(f"  {i}: area={isl['area']} center=({isl['center'][0]:.4f},{isl['center'][1]:.4f}) bbox={isl['bbox']}")

    # Match clusters to islands
    used_islands: set[int] = set()
    region_bboxes: dict[str, list[float]] = {}
    print("\nCluster -> island mapping:")
    for c in clusters:
        guess = classify_cluster(c)
        idx, isl = island_for_label(c, islands, used_islands)
        if isl is None:
            print(f"  {guess}: NO ISLAND")
            continue
        used_islands.add(idx)
        print(f"  {guess}: island {idx} bbox={isl['bbox']}")
        if guess != "unknown" and guess not in region_bboxes:
            region_bboxes[guess] = [round(x, 4) for x in isl["bbox"]]

    print("\nDerived region bboxes:")
    for k, v in sorted(region_bboxes.items()):
        print(f"  {k}: {v}")

    atlas = json.loads(JSON.read_text(encoding="utf-8"))
    print("\nCurrent JSON bboxes:")
    for r in atlas["regions"]:
        print(f"  {r['id']}: {r['bbox']}")


if __name__ == "__main__":
    main()