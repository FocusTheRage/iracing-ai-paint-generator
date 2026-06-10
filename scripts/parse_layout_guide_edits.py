"""Parse ai_layout_guideEdited.png for delete markers and relocation tags."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
EDITED = ROOT / "templates" / "ai_layout_guideEdited.png"
MASK = ROOT / "templates" / "cache" / "stockcars_toyotacamry2022" / "mask.png"
OUT = ROOT / "templates" / "atlas"


def connected(mask: np.ndarray, min_area: int = 40) -> list[dict]:
    h, w = mask.shape
    visited = np.zeros(mask.shape, bool)
    blobs: list[dict] = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or visited[sy, sx]:
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
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if len(xs) < min_area:
                continue
            blobs.append(
                {
                    "area": len(xs),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "cent": (float(np.mean(xs)), float(np.mean(ys))),
                }
            )
    return blobs


def norm_bbox(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 / w, y0 / h, (x1 + 1) / w, (y1 + 1) / h)


def bbox_from_mask_zone(mask: np.ndarray, zone: tuple[float, float, float, float]) -> list[float] | None:
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


def main() -> None:
    img = np.array(Image.open(EDITED).convert("RGB"))
    h, w = img.shape[:2]
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # White label chips (user relocation tags)
    white = (r > 200) & (g > 200) & (b > 200)
    white_blobs = connected(white, min_area=80)
    white_blobs.sort(key=lambda b: (b["cent"][1], b["cent"][0]))

    print("White label chips (relocation targets):")
    for i, blob in enumerate(white_blobs):
        nb = norm_bbox(blob["bbox"], w, h)
        cx, cy = blob["cent"]
        print(
            f"  #{i:2d} area={blob['area']:5d} cent=({cx/w:.3f},{cy/h:.3f}) "
            f"bbox=({nb[0]:.3f},{nb[1]:.3f},{nb[2]:.3f},{nb[3]:.3f})"
        )

    # Red "delete" text clusters
    red_text = (r > 180) & (g < 100) & (b < 100)
    red_blobs = connected(red_text, min_area=60)
    red_blobs.sort(key=lambda b: -b["area"])
    print("\nRed text blobs (top 30):")
    for i, blob in enumerate(red_blobs[:30]):
        nb = norm_bbox(blob["bbox"], w, h)
        cx, cy = blob["cent"]
        print(
            f"  #{i:2d} area={blob['area']:5d} cent=({cx/w:.3f},{cy/h:.3f}) "
            f"bbox=({nb[0]:.3f},{nb[1]:.3f},{nb[2]:.3f},{nb[3]:.3f})"
        )

    # Visualize chips on wire
    wire = Image.open(ROOT / "templates/cache/stockcars_toyotacamry2022/wire.png").convert("RGBA")
    overlay = wire.copy()
    draw = ImageDraw.Draw(overlay)
    for i, blob in enumerate(white_blobs):
        x0, y0, x1, y1 = blob["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0, 255), width=2)
        draw.text((x0, y0 - 14), str(i), fill=(0, 255, 0, 255))
    overlay.save(OUT / "edited_guide_white_chips.png")
    print(f"\nWrote {OUT / 'edited_guide_white_chips.png'}")


if __name__ == "__main__":
    main()