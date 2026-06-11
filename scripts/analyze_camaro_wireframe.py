"""Analyze labeled Camaro ZL1 Gen 6 wireframe text positions."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / "templates" / "wireframes" / "stockcars_camarozl12018" / "wireframe.png"


def main() -> None:
    wf = np.array(Image.open(WF).convert("RGB"))
    h, w = wf.shape[:2]
    cyan = (wf[:, :, 2] > 150) & (wf[:, :, 1] > 100) & (wf[:, :, 0] < 120)
    bright = (wf.max(axis=2) > 140) & ~cyan

    visited = np.zeros(bright.shape, bool)
    clusters: list[dict] = []
    for sy in range(h):
        for sx in range(w):
            if bright[sy, sx] and not visited[sy, sx]:
                q: deque[tuple[int, int]] = deque([(sx, sy)])
                visited[sy, sx] = True
                xs = [sx]
                ys = [sy]
                while q:
                    cx, cy = q.popleft()
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and bright[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                            xs.append(nx)
                            ys.append(ny)
                if len(xs) >= 20:
                    clusters.append(
                        {
                            "area": len(xs),
                            "cx": sum(xs) / len(xs),
                            "cy": sum(ys) / len(ys),
                        }
                    )

    clusters.sort(key=lambda c: (c["cy"], c["cx"]))
    lines: list[dict] = []
    for c in clusters:
        placed = False
        for line in lines:
            if abs(c["cy"] - line["cy"]) < 18:
                line["chars"].append(c)
                line["cy"] = sum(x["cy"] for x in line["chars"]) / len(line["chars"])
                placed = True
                break
        if not placed:
            lines.append({"cy": c["cy"], "chars": [c]})

    for line in sorted(lines, key=lambda item: item["cy"]):
        chars = sorted(line["chars"], key=lambda c: c["cx"])
        cx = sum(c["cx"] for c in chars) / len(chars)
        cy = line["cy"]
        print(
            f"y={cy / h:.3f} x={cx / w:.3f} "
            f"chars={len(chars)} area={sum(c['area'] for c in chars)}"
        )


if __name__ == "__main__":
    main()