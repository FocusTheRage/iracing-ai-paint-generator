"""
Re-extract UV region bboxes from the user's labeled reference PNG.

Detects red and black text labels on panel surfaces, classifies by position,
flood-fills the nearest white panel below each label, and refines with PSD mask
islands when fill bleeds.

Usage:
    python scripts/import_atlas_from_png.py "templates/atlas/ARCA Ford Mustang.png"
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ATLAS_JSON = ROOT / "templates" / "atlas" / "stockcars2_arcaford25.json"
MASK_PATH = ROOT / "templates" / "cache" / "stockcars2_arcaford25" / "mask.png"

FLOOD_LIMITS: dict[str, int] = {
    "front_bumper": 80_000,
    "left_side_door": 90_000,
    "rear_corner_upper": 70_000,
    "trunk": 75_000,
    "roof": 110_000,
    "hood": 110_000,
    "rear_corner_lower": 70_000,
    "right_side_door": 90_000,
    "back": 70_000,
    "rear_bumper": 55_000,
    "rear_diffuser": 45_000,
    "front_corner_upper_right": 65_000,
    "front_corner_lower_right": 65_000,
    "front_corner_upper_left": 65_000,
}

MEGA_ISLAND_AREA = 400_000
MAX_BBOX_AREA = 0.18

# Default region metadata for newly discovered labels.
NEW_REGION_DEFAULTS: dict[str, dict] = {
    "back": {
        "label": "BACK",
        "display_name": "Back / rear body",
        "aliases": ["back", "rear body", "rear clip"],
        "in_game_hint": "Top-right panel — rear body section on the UV sheet.",
    },
    "rear_bumper": {
        "label": "REAR BUMPER",
        "display_name": "Rear bumper",
        "aliases": ["rear bumper", "back bumper"],
        "in_game_hint": "Top-far-right panel — rear bumper cover.",
    },
    "rear_diffuser": {
        "label": "REAR DIFFUSER",
        "display_name": "Rear diffuser",
        "aliases": ["rear diffuser", "diffuser"],
        "in_game_hint": "Top-far-right lower panel — rear diffuser.",
    },
    "front_corner_upper_right": {
        "label": "FRONT CORNER",
        "display_name": "Front corner (upper right)",
        "aliases": ["front corner", "front quarter", "right front corner"],
        "in_game_hint": "Upper-right panel ahead of the hood — right front quarter.",
    },
    "front_corner_lower_right": {
        "label": "FRONT CORNER",
        "display_name": "Front corner (lower right)",
        "aliases": ["front corner lower", "right front lower"],
        "in_game_hint": "Lower-right panel — right front quarter extension.",
    },
    "front_corner_upper_left": {
        "label": "FRONT CORNER",
        "display_name": "Front corner (upper left)",
        "aliases": ["left front corner", "front corner left"],
        "in_game_hint": "Upper-left panel near the front — left front quarter.",
    },
}


def find_color_blobs(bitmap: np.ndarray, min_area: int = 30) -> list[dict]:
    h, w = bitmap.shape
    visited = np.zeros_like(bitmap, dtype=bool)
    blobs: list[dict] = []
    for sy in range(h):
        for sx in range(w):
            if bitmap[sy, sx] and not visited[sy, sx]:
                q: deque[tuple[int, int]] = deque([(sx, sy)])
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


def cluster_blobs(blobs: list[dict], y_thresh: int = 28, x_gap: int = 90) -> list[dict]:
    lines: list[list[dict]] = []
    for blob in sorted(blobs, key=lambda b: b["center"][1]):
        cy = blob["center"][1]
        for line in lines:
            if abs(cy - line[0]["center"][1]) <= y_thresh:
                line.append(blob)
                break
        else:
            lines.append([blob])

    clusters: list[dict] = []
    for line in lines:
        line.sort(key=lambda b: b["center"][0])
        group = [line[0]]
        for blob in line[1:]:
            if blob["bbox"][0] - group[-1]["bbox"][2] <= x_gap:
                group.append(blob)
            else:
                clusters.append(_merge_group(group))
                group = [blob]
        clusters.append(_merge_group(group))
    return clusters


def _merge_group(group: list[dict]) -> dict:
    xs = [b["bbox"][0] for b in group] + [b["bbox"][2] for b in group]
    ys = [b["bbox"][1] for b in group] + [b["bbox"][3] for b in group]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return {
        "bbox": (x0, y0, x1, y1),
        "center": ((x0 + x1) // 2, (y0 + y1) // 2),
        "w": x1 - x0,
        "h": y1 - y0,
    }


def find_red_clusters(arr: np.ndarray) -> list[dict]:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red = (r > 200) & (g < 80) & (b < 80)
    return cluster_blobs(find_color_blobs(red))


def find_dark_clusters(arr: np.ndarray) -> list[dict]:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    white = (r > 240) & (g > 240) & (b > 240)
    red = (r > 200) & (g < 80) & (b < 80)
    dilated_white = (
        np.array(Image.fromarray((white.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(7)))
        > 128
    )
    min_ch = np.minimum(np.minimum(r, g), b)
    max_ch = np.maximum(np.maximum(r, g), b)
    # Black/dark-gray/dark-blue labels on white panels (user's new labels).
    textish = (
        dilated_white
        & ~red
        & ~white
        & (
            (max_ch < 230)
            | (min_ch < 100)
        )
    )
    blobs = find_color_blobs(textish, min_area=35)
    text_blobs = [
        b
        for b in blobs
        if (b["bbox"][3] - b["bbox"][1]) <= 58
        and (b["bbox"][2] - b["bbox"][0]) >= 18
        and b["area"] < 8000
    ]
    clusters = cluster_blobs(text_blobs, y_thresh=26, x_gap=80)
    return [
        c
        for c in clusters
        if c["h"] <= 58 and c["w"] >= 30 and c["w"] <= 360 and c["h"] >= 10
    ]


def classify_red(cx: int, cy: int, w: int, h: int) -> str | None:
    nx, ny = cx / w, cy / h
    if ny < 0.20:
        return "front_bumper"
    if ny < 0.34 and nx > 0.35:
        return "left_side_door"
    if ny < 0.40 and nx < 0.25:
        return "rear_corner_upper"
    if 0.52 < ny < 0.66:
        if nx < 0.20:
            return "trunk"
        if nx > 0.70:
            return "hood"
        return "roof"
    if ny > 0.78 and nx < 0.25:
        return "rear_corner_lower"
    if ny > 0.84 and nx > 0.35:
        return "right_side_door"
    return None


# Approximate centers of user's black labels — used to match detected clusters.
BLACK_LABEL_SEEDS: dict[str, tuple[float, float]] = {
    "back": (0.699, 0.096),
    "rear_bumper": (0.703, 0.152),
    "rear_diffuser": (0.937, 0.169),
    "front_corner_upper_left": (0.100, 0.075),
    "front_corner_upper_right": (0.732, 0.345),
    "front_corner_lower_right": (0.723, 0.854),
}


def classify_dark(cx: int, cy: int, w: int, h: int, cluster: dict) -> str | None:
    nx, ny = cx / w, cy / h
    if ny > 0.97 or cluster["w"] > 360:
        return None
    best_id = None
    best_dist = 0.12
    for region_id, (sx, sy) in BLACK_LABEL_SEEDS.items():
        dist = ((nx - sx) ** 2 + (ny - sy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_id = region_id
    return best_id


def merge_front_bumper_red(clusters: list[dict], w: int, h: int) -> list[tuple[str, dict]]:
    assigned: list[tuple[str, dict]] = []
    used: set[str] = set()
    i = 0
    while i < len(clusters):
        c = clusters[i]
        cx, cy = c["center"]
        if (
            i + 1 < len(clusters)
            and abs(c["center"][1] - clusters[i + 1]["center"][1]) < 30
            and clusters[i + 1]["bbox"][0] - c["bbox"][2] < 120
            and classify_red(cx, cy, w, h) == "front_bumper"
        ):
            c2 = clusters[i + 1]
            merged = {
                "center": ((c["center"][0] + c2["center"][0]) // 2, (c["center"][1] + c2["center"][1]) // 2),
                "bbox": (
                    min(c["bbox"][0], c2["bbox"][0]),
                    min(c["bbox"][1], c2["bbox"][1]),
                    max(c["bbox"][2], c2["bbox"][2]),
                    max(c["bbox"][3], c2["bbox"][3]),
                ),
            }
            assigned.append(("front_bumper", merged))
            used.add("front_bumper")
            i += 2
            continue
        rid = classify_red(cx, cy, w, h)
        if rid and rid not in used:
            assigned.append((rid, c))
            used.add(rid)
        i += 1
    return assigned


def assign_dark_clusters(clusters: list[dict], w: int, h: int, used: set[str]) -> list[tuple[str, dict]]:
    assigned: list[tuple[str, dict]] = []
    for c in clusters:
        rid = classify_dark(c["center"][0], c["center"][1], w, h, c)
        if rid and rid not in used:
            assigned.append((rid, c))
            used.add(rid)

    # Seed fallback for black labels that did not form a clean text cluster.
    for region_id, (sx, sy) in BLACK_LABEL_SEEDS.items():
        if region_id in used:
            continue
        assigned.append((
            region_id,
            {
                "center": (int(sx * w), int(sy * h)),
                "bbox": (int(sx * w) - 40, int(sy * h) - 10, int(sx * w) + 40, int(sy * h) + 10),
            },
        ))
        used.add(region_id)
    return assigned


def bbox_from_seed(white: np.ndarray, fx: float, fy: float, limit: int) -> list[float]:
    h, w = white.shape
    x0, y0 = int(fx * w), int(fy * h)
    seed = None
    for radius in range(150):
        step = max(1, radius // 4)
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
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


def bbox_area(bbox: list[float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


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
    return [i for i in islands if i["area"] < MEGA_ISLAND_AREA]


def nearest_mask_island(
    lx: float,
    ly: float,
    islands: list[dict],
    used: set[int],
) -> list[float] | None:
    best_idx = None
    best_score = float("inf")
    for idx, isl in enumerate(islands):
        if idx in used:
            continue
        ib = isl["bbox"]
        icx, icy = isl["center"]
        inside = ib[0] <= lx <= ib[2] and ib[1] <= ly <= ib[3]
        dist = ((lx - icx) ** 2 + (ly - icy) ** 2) ** 0.5
        score = dist - (0.4 if inside else 0)
        if score < best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return None
    used.add(best_idx)
    return islands[best_idx]["bbox"]


def update_alias_groups(atlas: dict) -> None:
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
    groups["rear_quarter"] = [
        "rear_quarter_behind_door",
        "rear_corner_upper",
        "rear_corner_lower",
    ]
    groups["quarter"] = groups["rear_quarter"]
    groups["back"] = ["back"]
    groups["rear_bumper"] = ["rear_bumper"]
    groups["diffuser"] = ["rear_diffuser"]
    groups["front"] = [
        "front_bumper",
        "front_corner_upper_left",
        "front_corner_upper_bumper_right",
        "front_corner_upper_right",
        "front_corner_lower_right",
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
    groups["right_side"] = [
        "right_side_door",
        "hood",
        "front_corner_upper_right",
        "front_corner_lower_right",
        "back",
        "rear_bumper",
        "rear_diffuser",
    ]


def main(png_path: Path) -> None:
    arr = np.array(Image.open(png_path).convert("RGB"))
    h, w = arr.shape[:2]
    white = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)

    red_clusters = sorted(find_red_clusters(arr), key=lambda c: (c["center"][1], c["center"][0]))
    dark_clusters = sorted(find_dark_clusters(arr), key=lambda c: (c["center"][1], c["center"][0]))

    used_ids: set[str] = set()
    label_map = merge_front_bumper_red(red_clusters, w, h)
    for rid, _ in label_map:
        used_ids.add(rid)
    label_map.extend(assign_dark_clusters(dark_clusters, w, h, used_ids))

    mask_islands_list: list[dict] = []
    if MASK_PATH.exists():
        mask_islands_list = mask_islands(np.array(Image.open(MASK_PATH).convert("L")))
    used_islands: set[int] = set()

    atlas = json.loads(ATLAS_JSON.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in atlas["regions"]}

    print(f"Detected {len(label_map)} labeled regions in {png_path.name}:")
    for region_id, cluster in sorted(label_map, key=lambda x: x[0]):
        cx, cy = cluster["center"]
        anchor = [round(cx / w, 4), round(cy / h, 4)]
        probe_y = min(cluster["bbox"][3] + 12, h - 1) / h
        probe_x = cx / w

        limit = FLOOD_LIMITS.get(region_id, 80_000)
        bbox = bbox_from_seed(white, probe_x, probe_y, limit)

        if not bbox or bbox_area(bbox) > MAX_BBOX_AREA:
            fallback = nearest_mask_island(probe_x, probe_y, mask_islands_list, used_islands)
            if fallback:
                bbox = fallback
                print(f"  {region_id}: mask island {bbox} anchor={anchor}")
            elif not bbox:
                print(f"  {region_id}: WARNING — no bbox found")
                continue
        else:
            print(f"  {region_id}: {bbox} anchor={anchor}")

        defaults = NEW_REGION_DEFAULTS.get(region_id, {})
        if region_id not in by_id:
            by_id[region_id] = {
                "id": region_id,
                "label": defaults.get("label", region_id.upper().replace("_", " ")),
                "display_name": defaults.get("display_name", region_id.replace("_", " ")),
                "aliases": defaults.get("aliases", []),
                "in_game_hint": defaults.get("in_game_hint", ""),
            }
        by_id[region_id]["bbox"] = bbox
        by_id[region_id]["label_anchor"] = anchor

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
    atlas["reference_png"] = png_path.name
    atlas["version"] = atlas.get("version", 1) + 1
    update_alias_groups(atlas)
    ATLAS_JSON.write_text(json.dumps(atlas, indent=2), encoding="utf-8")
    print(f"Updated {ATLAS_JSON} ({len(atlas['regions'])} regions)")


if __name__ == "__main__":
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "templates" / "atlas" / "ARCA Ford Mustang.png"
    )
    main(path)