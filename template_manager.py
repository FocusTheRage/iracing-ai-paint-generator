"""
Download and cache official iRacing UV templates (PSD) from iRacing CDN.
Extracts wireframe, paintable mask, and background for constrained paint generation.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageFilter
from psd_tools import PSDImage

from cars_config import IRacingCar
from uv_atlas import (
    CarUVAtlas,
    build_ai_generation_guide,
    build_ai_layout_guide,
    build_ai_paint_reference,
    build_labeled_guide,
    load_atlas_for_car,
    sync_atlas_panel_regions,
)

logger = logging.getLogger(__name__)

TEMPLATE_BASE = (
    "https://ir-core-sites.iracing.com/members/member_images/cars/car_templates/"
)
CACHE_DIR = Path(__file__).parent / "templates" / "cache"
# Bump when mask/wire extraction logic changes (invalidates stale cache).
CACHE_VERSION = 27

def _load_template_zip_map() -> dict[str, str]:
    from car_catalog import template_zip_by_folder

    return template_zip_by_folder()


TEMPLATE_ZIP_BY_FOLDER: dict[str, str] = _load_template_zip_map()

WIRE_LAYER_NAMES = ("Wire", "wire", "WIRE")
# "Mask" is the true UV island mask. "Paintable Area" is a colored decal reference — NOT a mask.
PRIMARY_MASK_LAYER_NAMES = ("Mask", "mask")
REFERENCE_LAYER_NAMES = ("Paintable Area", "paintable area")


@dataclass
class CarTemplate:
    """Extracted iRacing UV template assets for one car."""

    car: IRacingCar
    wire: Image.Image
    paintable_mask: Image.Image
    background: Image.Image
    guide_image: Image.Image
    source_zip: str
    paintable_reference: Optional[Image.Image] = None
    panel_regions: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    labeled_guide: Optional[Image.Image] = None
    ai_layout_guide: Optional[Image.Image] = None
    ai_generation_guide: Optional[Image.Image] = None
    ai_reference: Optional[Image.Image] = None
    uv_atlas: Optional[CarUVAtlas] = None

    @property
    def resolution(self) -> int:
        return self.background.size[0]

    @property
    def ai_guide_image(self) -> Image.Image:
        """Unlabeled UV layout with wireframe — sent to AI for panel placement."""
        if self.ai_generation_guide is not None:
            return self.ai_generation_guide
        if self.ai_reference is not None:
            return self.ai_reference
        if self.ai_layout_guide is not None:
            return self.ai_layout_guide
        if self.labeled_guide is not None:
            return self.labeled_guide
        if self.ai_reference is not None:
            return self.ai_reference
        return _build_clean_ai_reference(
            self.paintable_mask, self.background.size, self.uv_atlas
        )


def _safe_slug(folder_path: str) -> str:
    return folder_path.replace("\\", "_").replace("/", "_")


def _find_layer(psd: PSDImage, names: tuple[str, ...]) -> Image.Image | None:
    for layer in psd.descendants():
        if layer.name in names and layer.width > 0 and layer.height > 0:
            return layer.composite()
    return None


def _align_layer(canvas_size: tuple[int, int], layer: Image.Image) -> Image.Image:
    if layer.size == canvas_size:
        return layer.convert("RGBA")
    return layer.resize(canvas_size, Image.Resampling.LANCZOS).convert("RGBA")


def _mask_from_layer(image: Image.Image) -> Image.Image:
    """
    Build a binary paint mask from the PSD 'Mask' layer.

    iRacing templates use two paintable tones:
      255 = trim, bumpers, and secondary UV islands
      74  = large primary body panels (doors, quarters, roof sections)

    Gray 74 pixels touching a 255 island are window/cutout holes and stay masked out.
    """
    gray = np.array(image.convert("L"), dtype=np.uint8)
    is_white = gray > 200
    is_gray_body = (gray >= 64) & (gray <= 200)

    if is_gray_body.any():
        dilated_white = np.array(
            Image.fromarray((is_white.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(7))
        ) > 0
        isolated_gray = is_gray_body & ~dilated_white
        paintable = is_white | isolated_gray
    else:
        paintable = is_white

    binary = np.where(paintable, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def _mask_coverage(mask: Image.Image) -> float:
    """Fraction of canvas marked paintable (0.0–1.0)."""
    arr = np.array(mask.convert("L"), dtype=np.uint8)
    return float((arr > 128).mean())


def _resolve_paintable_mask(
    canvas_size: tuple[int, int],
    mask_raw: Image.Image,
    paintable_ref_raw: Image.Image | None,
) -> Image.Image:
    """
    Build the best paintable mask from PSD layers.

    Some older templates (e.g. Gen-6 Camaro ZL1) ship a Mask layer that only
    contains gray body panels (~28% coverage) while the Paintable Area layer
    includes the full UV islands (~93%). Prefer the richer source when the
    primary mask is clearly incomplete.
    """
    primary = _mask_from_layer(_align_layer(canvas_size, mask_raw))
    primary_cov = _mask_coverage(primary)

    if paintable_ref_raw is None or primary_cov >= 0.50:
        return primary

    reference = _mask_from_layer(_align_layer(canvas_size, paintable_ref_raw))
    reference_cov = _mask_coverage(reference)

    if reference_cov > primary_cov + 0.15:
        logger.warning(
            "Mask layer only %.1f%% paintable — using Paintable Area (%.1f%%)",
            primary_cov * 100,
            reference_cov * 100,
        )
        return reference

    return primary


def _mask_outline_overlay(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Derive cyan wireframe lines from UV mask edges.
    PSD 'Wire' layers are often filled shapes (not thin lines), so mask edges are more reliable.
    """
    aligned = _align_layer(size, mask.convert("L"))
    edges = aligned.filter(ImageFilter.FIND_EDGES).convert("L")
    edge_mask = edges.point(lambda p: 255 if p > 24 else 0)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    color = Image.new("RGBA", size, (60, 220, 255, 200))
    overlay.paste(color, mask=edge_mask)
    return overlay


def _connected_blobs(
    bitmap: np.ndarray,
    min_area: int = 2000,
) -> list[tuple[int, int, int, int, int]]:
    """Return connected blobs as (area, x0, y0, x1, y1)."""
    h, w = bitmap.shape
    visited = np.zeros_like(bitmap, dtype=bool)
    blobs: list[tuple[int, int, int, int, int]] = []

    for y in range(h):
        for x in range(w):
            if not bitmap[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            xs = [x]
            ys = [y]
            while queue:
                cx, cy = queue.popleft()
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and bitmap[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
                        xs.append(nx)
                        ys.append(ny)
            if len(xs) >= min_area:
                blobs.append((len(xs), min(xs), min(ys), max(xs), max(ys)))

    blobs.sort(reverse=True)
    return blobs


def _norm_bbox(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int
) -> tuple[float, float, float, float]:
    return (
        float(x0) / w,
        float(y0) / h,
        float(x1 + 1) / w,
        float(y1 + 1) / h,
    )


def _bbox_from_mask_zone(
    mask: np.ndarray,
    x0f: float,
    y0f: float,
    x1f: float,
    y1f: float,
) -> Optional[tuple[float, float, float, float]]:
    h, w = mask.shape
    zone = mask.copy()
    zone[: int(y0f * h), :] = False
    zone[int(y1f * h) :, :] = False
    zone[:, : int(x0f * w)] = False
    zone[:, int(x1f * w) :] = False
    if zone.sum() < 500:
        return None
    ys, xs = np.where(zone)
    return _norm_bbox(xs.min(), ys.min(), xs.max(), ys.max(), w, h)


def _derive_panel_regions(
    paintable_reference: Optional[Image.Image],
    paintable_mask: Image.Image,
    template_hint: str,
) -> dict[str, tuple[float, float, float, float]]:
    """
    Derive normalized UV panel rectangles for this car's official template.
    ARCA templates place the door on the upper-right strip — not the NASCAR Cup layout.
    """
    mask = np.array(paintable_mask.convert("L")) > 128
    h, w = mask.shape
    regions: dict[str, tuple[float, float, float, float]] = {}

    if paintable_reference is not None:
        pa = np.array(_align_layer((w, h), paintable_reference).convert("RGB"))
        yellow = (
            (pa[:, :, 0] > 220)
            & (pa[:, :, 1] > 220)
            & (pa[:, :, 2] < 80)
            & mask
        )
        yellow_blobs = _connected_blobs(yellow, min_area=2500)
        if yellow_blobs:
            _, x0, y0, x1, y1 = yellow_blobs[0]
            regions["driver_side"] = _norm_bbox(x0, y0, x1, y1, w, h)
            if len(yellow_blobs) > 1 and yellow_blobs[1][0] > 4000:
                _, x0, y0, x1, y1 = yellow_blobs[1]
                regions["passenger_side"] = _norm_bbox(x0, y0, x1, y1, w, h)

        red = (
            (pa[:, :, 0] > 150)
            & (pa[:, :, 1] < 60)
            & (pa[:, :, 2] < 60)
            & mask
        )
        red_blobs = _connected_blobs(red, min_area=2500)
        if red_blobs:
            _, x0, y0, x1, y1 = red_blobs[0]
            regions["accent_side"] = _norm_bbox(x0, y0, x1, y1, w, h)

    ys, xs = np.where(mask)
    if len(xs):
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        y_span = max(y_max - y_min, 1)
        x_span = max(x_max - x_min, 1)

        row_idx = np.arange(h)

        hood = mask.copy()
        hood[row_idx > y_min + int(y_span * 0.28), :] = False
        if hood.sum() > 800:
            hy, hx = np.where(hood)
            regions.setdefault(
                "hood",
                _norm_bbox(hx.min(), hy.min(), hx.max(), hy.max(), w, h),
            )

        roof = mask.copy()
        roof[row_idx < y_min + int(y_span * 0.05), :] = False
        roof[row_idx > y_min + int(y_span * 0.22), :] = False
        if roof.sum() > 800:
            ry, rx = np.where(roof)
            regions.setdefault(
                "roof",
                _norm_bbox(rx.min(), ry.min(), rx.max(), ry.max(), w, h),
            )

        rear = mask.copy()
        rear[row_idx < y_min + int(y_span * 0.48), :] = False
        rear[row_idx > y_min + int(y_span * 0.72), :] = False
        if rear.sum() > 800:
            ry, rx = np.where(rear)
            regions.setdefault(
                "rear",
                _norm_bbox(rx.min(), ry.min(), rx.max(), ry.max(), w, h),
            )

        front_bumper = mask.copy()
        front_bumper[row_idx < y_min + int(y_span * 0.68), :] = False
        if front_bumper.sum() > 800:
            fy, fx = np.where(front_bumper)
            regions.setdefault(
                "front_bumper",
                _norm_bbox(fx.min(), fy.min(), fx.max(), fy.max(), w, h),
            )

        rear_bumper = mask.copy()
        rear_bumper[row_idx < y_min + int(y_span * 0.82), :] = False
        if rear_bumper.sum() > 800:
            by, bx = np.where(rear_bumper)
            regions.setdefault(
                "rear_bumper",
                _norm_bbox(bx.min(), by.min(), bx.max(), by.max(), w, h),
            )

    if "driver_side" not in regions:
        fallback_sets: dict[str, dict[str, tuple[float, float, float, float]]] = {
            "nascar": {
                "hood": (0.04, 0.03, 0.46, 0.24),
                "roof": (0.52, 0.03, 0.96, 0.20),
                "driver_side": (0.04, 0.22, 0.46, 0.50),
                "passenger_side": (0.52, 0.18, 0.96, 0.50),
                "rear": (0.04, 0.48, 0.96, 0.70),
                "front_bumper": (0.04, 0.70, 0.96, 0.86),
                "rear_bumper": (0.04, 0.86, 0.96, 0.98),
            },
            "gt": {
                "hood": (0.05, 0.04, 0.42, 0.28),
                "roof": (0.44, 0.04, 0.88, 0.22),
                "driver_side": (0.05, 0.26, 0.42, 0.58),
                "passenger_side": (0.44, 0.20, 0.88, 0.58),
                "rear": (0.05, 0.56, 0.88, 0.78),
                "front_splitter": (0.05, 0.76, 0.88, 0.92),
            },
        }
        hint = template_hint if template_hint in fallback_sets else "gt"
        fallback = fallback_sets[hint]
        for name in (
            "hood",
            "roof",
            "driver_side",
            "passenger_side",
            "rear",
            "front_bumper",
            "rear_bumper",
        ):
            if name in fallback:
                regions.setdefault(name, fallback[name])

    return regions


def _build_guide(
    mask: Image.Image,
    size: tuple[int, int],
    paintable_reference: Optional[Image.Image] = None,
) -> Image.Image:
    """AI reference: dark background, gray UV islands, cyan outlines, optional PA overlay."""
    guide = Image.new("RGBA", size, (24, 26, 30, 255))
    mask_l = _align_layer(size, mask.convert("L"))
    panel_hint = Image.new("RGBA", size, (0, 0, 0, 0))
    panels = Image.new("RGBA", size, (42, 44, 48, 255))
    panel_hint.paste(panels, mask=mask_l)
    guide = Image.alpha_composite(guide, panel_hint)

    if paintable_reference is not None:
        pa = _align_layer(size, paintable_reference).convert("RGBA")
        pa_arr = np.array(pa)
        # Fade the official Paintable Area decal layout into the guide.
        pa_arr[:, :, 3] = np.clip(pa_arr[:, :, 3] * 0.45, 0, 110).astype(np.uint8)
        pa_faded = Image.fromarray(pa_arr, mode="RGBA")
        guide = Image.alpha_composite(guide, pa_faded)

    guide = Image.alpha_composite(guide, _mask_outline_overlay(mask, size))
    return guide


def _build_clean_ai_reference(
    mask: Image.Image,
    size: tuple[int, int],
    uv_atlas: Optional[CarUVAtlas] = None,
) -> Image.Image:
    """AI layout reference with panel shapes + wireframe."""
    if uv_atlas is not None:
        return build_ai_paint_reference(mask, uv_atlas, size[0])
    return build_ai_generation_guide(mask, None, size[0])


def _download_zip(zip_name: str) -> bytes:
    url = TEMPLATE_BASE + zip_name
    logger.info("Downloading template %s", zip_name)
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return resp.content


def _load_psd_from_zip(zip_bytes: bytes) -> PSDImage:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        psd_names = [n for n in zf.namelist() if n.lower().endswith(".psd")]
        if not psd_names:
            raise RuntimeError("No PSD found in template ZIP")
        return PSDImage.open(io.BytesIO(zf.read(psd_names[0])))


def _extract_template(car: IRacingCar, zip_name: str) -> CarTemplate:
    zip_bytes = _download_zip(zip_name)
    psd = _load_psd_from_zip(zip_bytes)

    mask_raw = _find_layer(psd, PRIMARY_MASK_LAYER_NAMES)
    if mask_raw is None:
        raise RuntimeError(f"Mask layer not found in template for {car.display_name}")

    paintable_ref_raw = _find_layer(psd, REFERENCE_LAYER_NAMES)

    canvas_size = (car.resolution, car.resolution)
    paintable_mask = _resolve_paintable_mask(
        canvas_size, mask_raw, paintable_ref_raw
    )
    wire = _mask_outline_overlay(paintable_mask, canvas_size)

    paintable_reference = None
    if paintable_ref_raw is not None:
        paintable_reference = _align_layer(canvas_size, paintable_ref_raw)

    uv_atlas = load_atlas_for_car(car.folder_path)
    if uv_atlas is not None:
        panel_regions = sync_atlas_panel_regions(uv_atlas)
    else:
        panel_regions = _derive_panel_regions(
            paintable_reference,
            paintable_mask,
            car.template_hint,
        )

    # iRacing TGAs use black outside UV islands.
    background = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
    guide_image = _build_guide(paintable_mask, canvas_size, paintable_reference)
    ai_reference = _build_clean_ai_reference(paintable_mask, canvas_size, uv_atlas)
    labeled_guide = None
    ai_layout_guide = None
    ai_generation_guide = build_ai_generation_guide(
        paintable_mask,
        uv_atlas,
        canvas_size[0],
    )
    if uv_atlas is not None:
        labeled_guide = build_labeled_guide(
            paintable_mask,
            uv_atlas,
            paintable_reference,
            canvas_size[0],
        )
        ai_layout_guide = build_ai_layout_guide(
            paintable_mask,
            uv_atlas,
            canvas_size[0],
        )

    slug = _safe_slug(car.folder_path)
    car_cache = CACHE_DIR / slug
    car_cache.mkdir(parents=True, exist_ok=True)
    wire.save(car_cache / "wire.png")
    paintable_mask.save(car_cache / "mask.png")
    background.save(car_cache / "background.png")
    guide_image.save(car_cache / "guide.png")
    ai_reference.save(car_cache / "ai_reference.png")
    if labeled_guide is not None:
        labeled_guide.save(car_cache / "labeled_guide.png")
    if ai_layout_guide is not None:
        ai_layout_guide.save(car_cache / "ai_layout_guide.png")
    ai_generation_guide.save(car_cache / "ai_generation_guide.png")
    if paintable_reference is not None:
        paintable_reference.save(car_cache / "paintable_reference.png")
    (car_cache / "panels.json").write_text(
        json.dumps(panel_regions, indent=2),
        encoding="utf-8",
    )
    (car_cache / "version.txt").write_text(str(CACHE_VERSION), encoding="utf-8")

    mask_pct = (np.array(paintable_mask) > 128).mean() * 100
    logger.info(
        "Template %s: mask %.1f%%, atlas regions %s",
        zip_name,
        mask_pct,
        len(uv_atlas.regions) if uv_atlas else "none",
    )

    return CarTemplate(
        car=car,
        wire=wire,
        paintable_mask=paintable_mask,
        background=background,
        guide_image=guide_image,
        source_zip=zip_name,
        paintable_reference=paintable_reference,
        panel_regions=panel_regions,
        labeled_guide=labeled_guide,
        ai_layout_guide=ai_layout_guide,
        ai_generation_guide=ai_generation_guide,
        ai_reference=ai_reference,
        uv_atlas=uv_atlas,
    )


def _load_cached(car: IRacingCar) -> CarTemplate | None:
    slug = _safe_slug(car.folder_path)
    car_cache = CACHE_DIR / slug
    required = ["wire.png", "mask.png", "background.png", "guide.png", "version.txt"]
    # labeled_guide.png is optional (added in cache v6)
    if not all((car_cache / f).exists() for f in required):
        return None
    try:
        if int((car_cache / "version.txt").read_text(encoding="utf-8").strip()) < CACHE_VERSION:
            return None
    except ValueError:
        return None

    zip_name = TEMPLATE_ZIP_BY_FOLDER.get(car.folder_path, "cached")
    paintable_reference = None
    pa_path = car_cache / "paintable_reference.png"
    if pa_path.exists():
        paintable_reference = Image.open(pa_path)

    panel_regions: dict[str, tuple[float, float, float, float]] = {}
    panels_path = car_cache / "panels.json"
    if panels_path.exists():
        raw = json.loads(panels_path.read_text(encoding="utf-8"))
        panel_regions = {k: tuple(v) for k, v in raw.items()}

    labeled_guide = None
    lg_path = car_cache / "labeled_guide.png"
    if lg_path.exists():
        labeled_guide = Image.open(lg_path)

    uv_atlas = load_atlas_for_car(car.folder_path)
    if uv_atlas is not None and not panel_regions:
        panel_regions = sync_atlas_panel_regions(uv_atlas)

    if labeled_guide is None and uv_atlas is not None:
        labeled_guide = build_labeled_guide(
            Image.open(car_cache / "mask.png"),
            uv_atlas,
            paintable_reference,
            car.resolution,
        )

    ai_layout_guide = None
    layout_path = car_cache / "ai_layout_guide.png"
    if layout_path.exists():
        ai_layout_guide = Image.open(layout_path)
    elif uv_atlas is not None:
        ai_layout_guide = build_ai_layout_guide(
            Image.open(car_cache / "mask.png"),
            uv_atlas,
            car.resolution,
        )

    gen_path = car_cache / "ai_generation_guide.png"
    if gen_path.exists():
        ai_generation_guide = Image.open(gen_path)
    else:
        ai_generation_guide = build_ai_generation_guide(
            Image.open(car_cache / "mask.png"),
            uv_atlas,
            car.resolution,
        )

    ai_reference = None
    ai_ref_path = car_cache / "ai_reference.png"
    if ai_ref_path.exists():
        ai_reference = Image.open(ai_ref_path)
    else:
        ai_reference = _build_clean_ai_reference(
            Image.open(car_cache / "mask.png"),
            (car.resolution, car.resolution),
            uv_atlas,
        )

    return CarTemplate(
        car=car,
        wire=Image.open(car_cache / "wire.png"),
        paintable_mask=Image.open(car_cache / "mask.png"),
        background=Image.open(car_cache / "background.png").convert("RGBA"),
        guide_image=Image.open(car_cache / "guide.png"),
        source_zip=zip_name,
        paintable_reference=paintable_reference,
        panel_regions=panel_regions,
        labeled_guide=labeled_guide,
        ai_layout_guide=ai_layout_guide,
        ai_generation_guide=ai_generation_guide,
        ai_reference=ai_reference,
        uv_atlas=uv_atlas,
    )


def get_car_template(car: IRacingCar, force_refresh: bool = False) -> CarTemplate:
    """Return cached template or download from iRacing CDN."""
    if force_refresh:
        slug = _safe_slug(car.folder_path)
        import shutil
        cache_path = CACHE_DIR / slug
        if cache_path.exists():
            shutil.rmtree(cache_path)

    if not force_refresh:
        cached = _load_cached(car)
        if cached is not None:
            return cached

    zip_name = TEMPLATE_ZIP_BY_FOLDER.get(car.folder_path)
    if not zip_name:
        raise RuntimeError(
            f"No official template mapping for {car.display_name} ({car.folder_path})"
        )
    return _extract_template(car, zip_name)


def preload_all_templates() -> dict[str, str]:
    """Download/cache all configured templates. Returns {car_name: status}."""
    results: dict[str, str] = {}
    for car_name, car in __import__("cars_config").CAR_BY_NAME.items():
        try:
            get_car_template(car, force_refresh=True)
            results[car_name] = "ok"
        except Exception as exc:
            results[car_name] = f"error: {exc}"
    return results