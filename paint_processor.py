"""
TGA export, spec-map generation, template compositing, and previews for iRacing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from cars_config import IRacingCar, get_output_filenames
from template_manager import CarTemplate
from uv_atlas import build_bbox_seam_mask, build_guide_strip_mask

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

# UV panel regions as fractions of 2048 canvas (x0, y0, x1, y1) per car class.
# Used to sample livery colors and map onto a side-view silhouette.
UV_PANELS: dict[str, dict[str, tuple[float, float, float, float]]] = {
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
    "openwheel": {
        "nose": (0.08, 0.05, 0.45, 0.25),
        "cockpit": (0.08, 0.24, 0.55, 0.45),
        "sidepod_l": (0.05, 0.42, 0.40, 0.68),
        "sidepod_r": (0.55, 0.42, 0.92, 0.68),
        "engine_cover": (0.30, 0.66, 0.88, 0.88),
        "rear_wing": (0.55, 0.86, 0.95, 0.98),
    },
}

# Map template panel names onto side-view silhouette zones.
PANEL_TO_SIDE_ZONE: dict[str, str] = {
    "hood": "hood",
    "roof": "roof",
    "driver_side": "driver_side",
    "passenger_side": "passenger_side",
    "accent_side": "driver_side",
    "rear": "rear",
    "front_bumper": "front_bumper",
    "rear_bumper": "rear_bumper",
    "front_splitter": "front_splitter",
}

# Side-view silhouette zones (polygon points) per class.
SIDE_VIEW_ZONES: dict[str, dict[str, list[tuple[int, int]]]] = {
    "nascar": {
        "hood": [(180, 175), (340, 130), (520, 130), (620, 175), (620, 230), (180, 230)],
        "roof": [(340, 95), (520, 95), (580, 130), (340, 130)],
        "driver_side": [(180, 230), (620, 230), (620, 310), (180, 310)],
        "passenger_side": [(620, 175), (900, 200), (900, 310), (620, 310)],
        "rear": [(180, 310), (900, 310), (900, 360), (180, 360)],
        "front_bumper": [(120, 230), (180, 230), (180, 360), (120, 360)],
        "rear_bumper": [(900, 280), (980, 280), (980, 360), (900, 360)],
    },
    "gt": {
        "hood": [(200, 170), (380, 120), (560, 120), (660, 170), (660, 240), (200, 240)],
        "roof": [(380, 90), (560, 90), (620, 120), (380, 120)],
        "driver_side": [(200, 240), (660, 240), (660, 320), (200, 320)],
        "passenger_side": [(660, 170), (920, 195), (920, 320), (660, 320)],
        "rear": [(200, 320), (920, 320), (920, 365), (200, 365)],
        "front_splitter": [(140, 240), (200, 240), (200, 365), (140, 365)],
    },
    "openwheel": {
        "nose": [(220, 200), (420, 150), (500, 200), (420, 260), (220, 260)],
        "cockpit": [(220, 260), (500, 260), (500, 300), (220, 300)],
        "sidepod_l": [(180, 300), (420, 300), (420, 340), (180, 340)],
        "sidepod_r": [(500, 300), (880, 300), (880, 340), (500, 340)],
        "engine_cover": [(420, 300), (880, 300), (880, 360), (420, 360)],
        "rear_wing": [(700, 140), (950, 120), (950, 180), (700, 200)],
    },
}


def _resize_to(image: Image.Image, size: int) -> Image.Image:
    if image.size == (size, size):
        return image.convert("RGBA")
    return image.resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")


def apply_paint_to_template(
    generated: Image.Image,
    template: CarTemplate,
) -> Image.Image:
    """
    Mask AI output to the car's paintable UV area and composite onto template BG.
    This keeps the livery inside the official iRacing wireframe boundaries.
    """
    return clip_paint_to_mask(generated, template)


def fill_unpainted_mask_areas(
    paint: Image.Image,
    template: CarTemplate,
    threshold: int = 18,
) -> Image.Image:
    """
    Fill mask regions the AI left black with the livery's base color.
    AI often paints only part of the UV sheet; this covers the full body.
    """
    size = template.resolution
    img = _resize_to(paint, size)
    mask = _resize_to(template.paintable_mask.convert("RGBA"), size).convert("L")
    arr = np.array(img.convert("RGBA"))
    mask_arr = np.array(mask) > 128
    lum = arr[:, :, :3].max(axis=2)

    painted = mask_arr & (lum > threshold)
    unpainted = mask_arr & (lum <= threshold)
    if not unpainted.any():
        return img

    if painted.any():
        colors = arr[:, :, :3][painted].astype(np.float32)
        saturation = colors.max(axis=1) - colors.min(axis=1)
        colorful = colors[saturation > 25]
        if len(colorful) > 50:
            fill = np.median(colorful, axis=0).astype(np.uint8)
        else:
            fill = np.median(colors, axis=0).astype(np.uint8)
    else:
        fill = np.array([35, 35, 40], dtype=np.uint8)

    arr[unpainted, :3] = fill
    arr[unpainted, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def clip_paint_to_mask(
    paint: Image.Image,
    template: CarTemplate,
) -> Image.Image:
    """Hard-clip paint to UV islands on a black background — required for in-sim use."""
    size = template.resolution
    gen = _resize_to(paint, size)
    mask = _resize_to(template.paintable_mask.convert("RGBA"), size).convert("L")
    bg = _resize_to(template.background, size)

    clipped = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    clipped.paste(gen, mask=mask)
    return Image.composite(clipped, bg, mask)


def create_template_preview(
    paint: Image.Image,
    template: CarTemplate,
    *,
    show_wire: bool = False,
) -> Image.Image:
    """Show final TGA layout. Wire overlay is off by default (matches exported paint)."""
    size = template.resolution
    base = _resize_to(paint, size)
    if show_wire:
        wire = _resize_to(template.wire, size).convert("RGBA")
        if wire.split()[3].getextrema()[1] > 0:
            preview = Image.alpha_composite(base, wire)
        else:
            preview = base
    else:
        preview = base

    # Readable thumbnail for UI.
    preview.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return preview


def _uv_crop(paint: Image.Image, rect: tuple[float, float, float, float]) -> Image.Image:
    w, h = paint.size
    x0, y0, x1, y1 = rect
    return paint.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def _dominant_colors(image: Image.Image, count: int = 3) -> list[tuple[int, int, int]]:
    """Return the most common RGB colors in an image patch."""
    thumb = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    arr = np.array(thumb).reshape(-1, 3)
    dark = arr.max(axis=1) < 24
    arr = arr[~dark]
    if len(arr) == 0:
        return [(40, 40, 44)]

    quantized = (arr // 24) * 24
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    colors: list[tuple[int, int, int]] = []
    for idx in order[: count * 2]:
        rgb = tuple(int(v) for v in unique[idx])
        if rgb not in colors:
            colors.append(rgb)
        if len(colors) >= count:
            break
    return colors or [(40, 40, 44)]


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def _sample_panel_style(
    paint: Image.Image,
    uv_rect: tuple[float, float, float, float],
) -> dict:
    """Summarize a UV region as clean fill/stripe colors for the side silhouette."""
    patch = _uv_crop(paint, uv_rect).convert("RGB")
    if patch.width < 2 or patch.height < 2:
        colors = [(40, 40, 44)]
    else:
        colors = _dominant_colors(patch, count=3)

    primary = colors[0]
    accent = colors[1] if len(colors) > 1 else primary
    use_accent = _color_distance(primary, accent) > 55

    # Detect strong vertical or horizontal banding (racing stripes).
    small = np.array(patch.resize((48, 48), Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32)
    row_var = small.std(axis=1).mean()
    col_var = small.std(axis=0).mean()
    striped = use_accent and (row_var > 28 or col_var > 28)

    return {
        "primary": primary,
        "accent": accent,
        "striped": striped,
        "use_accent": use_accent,
    }


def _draw_styled_zone(
    canvas: Image.Image,
    polygon: list[tuple[int, int]],
    style: dict,
) -> None:
    """Paint a silhouette zone using sampled livery colors — no UV stretching."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    dw, dh = max(x1 - x0, 8), max(y1 - y0, 8)

    zone = Image.new("RGBA", (dw, dh), style["primary"] + (255,))
    if style.get("striped") and style.get("use_accent"):
        stripe_layer = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
        stripe_draw = ImageDraw.Draw(stripe_layer)
        accent = style["accent"] + (190,)
        stripe_w = max(10, dw // 14)
        for x in range(-dh, dw + dh, stripe_w * 2):
            stripe_draw.polygon(
                [(x, 0), (x + stripe_w, 0), (x + stripe_w + dh, dh), (x + dh, dh)],
                fill=accent,
            )
        zone = Image.alpha_composite(zone, stripe_layer)

    mask = Image.new("L", (dw, dh), 0)
    shifted = [(px - x0, py - y0) for px, py in polygon]
    ImageDraw.Draw(mask).polygon(shifted, fill=255)
    canvas.paste(zone, (x0, y0), mask)


def _zone_styles_from_panels(
    paint: Image.Image,
    panels: dict[str, tuple[float, float, float, float]],
    zones: dict[str, list[tuple[int, int]]],
) -> dict[str, dict]:
    """Merge per-panel UV samples into one style per side-view zone."""
    styles: dict[str, dict] = {}
    for panel_name, uv_rect in panels.items():
        zone_name = PANEL_TO_SIDE_ZONE.get(panel_name)
        if zone_name is None or zone_name not in zones:
            continue
        style = _sample_panel_style(paint, uv_rect)
        if zone_name not in styles:
            styles[zone_name] = style
            continue
        # Blend accent hints when multiple panels map to the same zone.
        existing = styles[zone_name]
        if style.get("striped"):
            existing["striped"] = True
            existing["accent"] = style["accent"]
            existing["use_accent"] = True
    return styles


def create_car_side_preview(
    paint: Image.Image,
    car: IRacingCar,
    template: Optional[CarTemplate] = None,
) -> Image.Image:
    """
    Build a stylized side-view preview using dominant livery colors per body zone.
    Avoids stretching UV panels (which are flat unwraps) onto a 3D silhouette.
    """
    hint = car.template_hint if car.template_hint in SIDE_VIEW_ZONES else "gt"
    zones = SIDE_VIEW_ZONES[hint]

    if template is not None and template.panel_regions:
        panels = template.panel_regions
    else:
        panels = UV_PANELS.get(hint, UV_PANELS["gt"])

    zone_styles = _zone_styles_from_panels(paint, panels, zones)
    if not zone_styles:
        zone_styles = {
            zone_name: _sample_panel_style(paint, uv_rect)
            for zone_name, uv_rect in panels.items()
            if zone_name in zones
        }

    w, h = 1100, 480
    canvas = Image.new("RGBA", (w, h), (18, 20, 26, 255))
    draw = ImageDraw.Draw(canvas)

    draw.ellipse((80, 390, 1020, 450), fill=(10, 10, 12, 255))

    for zone_name in zones:
        style = zone_styles.get(zone_name)
        if style is None:
            continue
        _draw_styled_zone(canvas, zones[zone_name], style)

    for cx in (220, 860):
        draw.ellipse((cx - 55, 330, cx + 55, 440), fill=(12, 12, 14, 255))
        draw.ellipse((cx - 35, 350, cx + 35, 420), fill=(30, 30, 34, 255))

    if hint == "nascar":
        draw.polygon([(350, 130), (560, 130), (580, 200), (350, 200)], fill=(30, 40, 55, 220))
    elif hint == "gt":
        draw.polygon([(380, 120), (580, 120), (600, 210), (380, 210)], fill=(30, 40, 55, 220))
    elif hint == "openwheel":
        draw.polygon([(300, 170), (420, 150), (440, 220), (300, 240)], fill=(20, 30, 45, 200))

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 20), car.display_name, fill=(200, 200, 210, 255), font=font)
    draw.text(
        (20, 44),
        "Side-view color preview (see UV template for exact layout)",
        fill=(120, 130, 150, 255),
        font=font,
    )

    return canvas


def _guide_artifact_mask_from_atlas(template: CarTemplate, size: int) -> np.ndarray | None:
    """Thin ring along atlas bbox edges — only where AI copied frame lines."""
    if template.uv_atlas is None:
        return None
    return build_bbox_seam_mask(template.uv_atlas, size, band=6)


def _color_artifact_mask(
    mask_arr: np.ndarray,
    rgb: np.ndarray,
) -> np.ndarray:
    """Detect guide/wireframe colors the AI often copies into the paint."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = rgb.max(axis=2)

    # Guide wireframe is ~(60, 220, 255) — avoid stripping deep blues in livery art.
    cyan_wire = mask_arr & (
        (r >= 35)
        & (r <= 110)
        & (g >= 175)
        & (b >= 210)
        & (b > g - 20)
        & (g > r + 90)
    )
    # Guide yellow (~220,180,20) — exclude safety orange and other low-green livery colors.
    yellow_tag = mask_arr & (
        (b < 90) & (g > 120) & (r > 150) & (g >= r - 25)
        | (
            (np.abs(r - 220) < 45)
            & (np.abs(g - 180) < 50)
            & (b < 100)
        )
    )
    # Guide red outline (~220,60,60) — avoid stripping orange/red body paint.
    red_guide = mask_arr & (
        (np.abs(r - 220) < 45) & (np.abs(g - 60) < 40) & (np.abs(b - 60) < 40)
        | ((r > 185) & (g < 75) & (b < 75) & (r > g + 90))
    )
    guide_gray = mask_arr & (
        (np.abs(r - 48) < 18) & (np.abs(g - 50) < 18) & (np.abs(b - 55) < 18)
        | (np.abs(r - 52) < 14) & (np.abs(g - 54) < 14) & (np.abs(b - 58) < 14)
    )
    panel_hint = mask_arr & (
        (np.abs(r - 210) < 28) & (np.abs(g - 210) < 28) & (np.abs(b - 220) < 28)
    )
    return cyan_wire | yellow_tag | red_guide | guide_gray | panel_hint


def _dilate_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    if not mask.any():
        return mask
    filt = max(3, radius | 1)
    return (
        np.array(Image.fromarray(mask.astype(np.uint8) * 255).filter(
            ImageFilter.MaxFilter(filt)
        ))
        > 0
    )


def _reference_guide_artifact_mask(template: CarTemplate, size: int) -> np.ndarray | None:
    """
    Pixels that are labels, red zone boxes, or cyan wire on the labeled guide.
    Uses the human labeled guide so copied text/lines are removed at source positions.
    """
    guide_img = template.labeled_guide or template.ai_layout_guide
    if guide_img is None:
        return None

    guide = np.array(_resize_to(guide_img.convert("RGB"), size))
    r, g, b = guide[:, :, 0], guide[:, :, 1], guide[:, :, 2]

    yellow_tag = (
        (b < 105)
        & (g > 130)
        & (r > 170)
        & (g >= r - 55)
        & (r - g < 70)
    )
    red_box = (
        (np.abs(r.astype(int) - 220) < 40)
        & (np.abs(g.astype(int) - 60) < 40)
        & (np.abs(b.astype(int) - 60) < 40)
    )
    cyan_wire = (
        (np.abs(r.astype(int) - 60) < 40)
        & (g > 165)
        & (b > 200)
        & (g > r + 80)
    )
    # Label lettering only — exclude guide panel gray ~(48,50,55).
    label_text = (r < 28) & (g < 26) & (b < 32) & (r > 4)
    tag_border = (
        (np.abs(r.astype(int) - 40) < 22)
        & (np.abs(g.astype(int) - 35) < 22)
        & (b < 28)
    )

    labels_and_boxes = yellow_tag | red_box | label_text | tag_border
    # Generous dilation on labels so copied/warped text is fully removed.
    label_badges = _dilate_mask(yellow_tag | label_text | tag_border, radius=10)
    box_lines = _dilate_mask(red_box, radius=4)
    wire_lines = _dilate_mask(cyan_wire, radius=2)
    return label_badges | box_lines | wire_lines


def _paint_guide_color_mask(mask_arr: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Guide-like colors the AI copied — tight match to label/box/wire hues only."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    yellow_tag = mask_arr & (
        (np.abs(r - 220) < 45)
        & (np.abs(g - 180) < 50)
        & (b < 105)
        & (g > r - 50)
    )
    red_box = mask_arr & (
        (np.abs(r - 220) < 40)
        & (np.abs(g - 60) < 40)
        & (np.abs(b - 60) < 40)
    )
    cyan_wire = mask_arr & (
        (np.abs(r - 60) < 38)
        & (g > 165)
        & (b > 200)
        & (g > r + 75)
    )
    label_text = mask_arr & (r < 30) & (g < 28) & (b < 35) & ((r + g + b) < 80)
    return yellow_tag | red_box | cyan_wire | label_text


def _build_guide_artifact_mask(
    paint_arr: np.ndarray,
    template: CarTemplate,
    size: int,
) -> np.ndarray:
    """Union of geometric strip mask, reference-guide positions, and copied guide colors."""
    mask_arr = np.array(
        _resize_to(template.paintable_mask.convert("RGBA"), size).convert("L")
    ) > 128
    rgb = paint_arr[:, :, :3].astype(np.int16)

    artifact = np.zeros(mask_arr.shape, dtype=bool)

    if template.uv_atlas is not None:
        artifact |= mask_arr & build_guide_strip_mask(template.uv_atlas, size)

    ref_mask = _reference_guide_artifact_mask(template, size)
    if ref_mask is not None:
        artifact |= mask_arr & ref_mask

    artifact |= mask_arr & _paint_guide_color_mask(mask_arr, rgb)

    atlas_mask = _guide_artifact_mask_from_atlas(template, size)
    if atlas_mask is not None:
        artifact |= mask_arr & atlas_mask

    wire_band = _wire_overlay_band(template, size)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    wire_cyan_paint = (
        (np.abs(r - 60) < 38)
        & (g > 165)
        & (b > 200)
        & (g > r + 75)
    )
    artifact |= mask_arr & wire_band & wire_cyan_paint

    guide_gray = mask_arr & (
        (np.abs(r - 48) < 14)
        & (np.abs(g - 50) < 14)
        & (np.abs(b - 55) < 14)
        | (np.abs(r - 52) < 12)
        & (np.abs(g - 54) < 12)
        & (np.abs(b - 58) < 12)
    )
    artifact |= guide_gray

    return artifact & mask_arr


def _inpaint_texture(
    arr: np.ndarray,
    holes: np.ndarray,
    mask_arr: np.ndarray,
    source_rgb: np.ndarray,
    max_passes: int = 24,
) -> None:
    """Fill stripped guide pixels by propagating nearby livery texture inward."""
    unresolved_full = holes & mask_arr
    if not unresolved_full.any():
        return

    ys, xs = np.where(unresolved_full)
    margin = 28
    h, w = arr.shape[:2]
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + margin + 1)

    sub_mask = mask_arr[y0:y1, x0:x1]
    sub_holes = holes[y0:y1, x0:x1]
    sub_src = source_rgb[y0:y1, x0:x1].astype(np.float32)
    sh, sw = sub_mask.shape

    unresolved = sub_holes & sub_mask
    valid = (~sub_holes) & sub_mask
    out = sub_src.copy()

    for _ in range(max_passes):
        if not unresolved.any():
            break
        valid_f = valid.astype(np.float32)
        counts = np.zeros((sh, sw), dtype=np.float32)
        sums = np.zeros((sh, sw, 3), dtype=np.float32)
        valid_p = np.pad(valid_f, 1)
        for c in range(3):
            ch_p = np.pad(out[:, :, c], 1)
            ch_sum = np.zeros((sh, sw), dtype=np.float32)
            for dy in range(3):
                for dx in range(3):
                    if dy == 1 and dx == 1:
                        continue
                    neighbor_valid = valid_p[dy : dy + sh, dx : dx + sw]
                    ch_sum += ch_p[dy : dy + sh, dx : dx + sw] * neighbor_valid
            sums[:, :, c] = ch_sum
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                counts += valid_p[dy : dy + sh, dx : dx + sw]

        can_fill = unresolved & (counts > 0)
        if not can_fill.any():
            break
        for c in range(3):
            out[:, :, c][can_fill] = sums[:, :, c][can_fill] / counts[can_fill]
        valid |= can_fill
        unresolved &= ~can_fill

    filled = sub_holes & sub_mask
    arr[y0:y1, x0:x1][filled, :3] = np.clip(out[filled], 0, 255).astype(np.uint8)
    arr[y0:y1, x0:x1][filled, 3] = 255


def _fill_artifact_pixels(
    arr: np.ndarray,
    artifact: np.ndarray,
    mask_arr: np.ndarray,
    lum: np.ndarray,
    luminance_threshold: int,
    template: CarTemplate,
    source_rgb: np.ndarray,
) -> None:
    """Replace guide artifacts using texture propagated from the pre-strip livery."""
    _inpaint_texture(arr, artifact, mask_arr, source_rgb)


def smooth_bbox_seams(
    paint: Image.Image,
    template: CarTemplate,
    band: int = 8,
    blur_radius: int = 4,
    blend: float = 0.35,
) -> Image.Image:
    """Light feather only on thin bbox seams — preserves storm/lightning detail."""
    if template.uv_atlas is None:
        return paint
    size = template.resolution
    img = _resize_to(paint, size)
    arr = np.array(img.convert("RGBA"))
    mask_arr = np.array(
        _resize_to(template.paintable_mask.convert("RGBA"), size).convert("L")
    ) > 128
    seam = build_bbox_seam_mask(template.uv_atlas, size, band=band) & mask_arr
    if not seam.any():
        return img

    rgb = Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
    blurred = np.array(rgb.filter(ImageFilter.GaussianBlur(blur_radius)), dtype=np.float32)
    keep = 1.0 - blend
    for c in range(3):
        channel = arr[:, :, c].astype(np.float32)
        channel[seam] = channel[seam] * keep + blurred[:, :, c][seam] * blend
        arr[:, :, c] = channel.astype(np.uint8)
    arr[:, :, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def _wire_overlay_band(template: CarTemplate, size: int) -> np.ndarray:
    """Dilated alpha band from the template wire overlay PNG."""
    wire = _resize_to(template.wire.convert("RGBA"), size)
    alpha = np.array(wire.split()[3])
    return (
        np.array(Image.fromarray(alpha).filter(ImageFilter.MaxFilter(5))) > 32
    )


def _mask_edge_band(mask_arr: np.ndarray, radius: int = 5) -> np.ndarray:
    """Thin band along UV island edges (where wireframe lines sit)."""
    mask_img = Image.fromarray((mask_arr.astype(np.uint8) * 255))
    edges = np.array(mask_img.filter(ImageFilter.FIND_EDGES)) > 28
    return (
        np.array(Image.fromarray((edges * 255).astype(np.uint8)).filter(
            ImageFilter.MaxFilter(radius)
        ))
        > 0
    )


def strip_guide_overlays(
    paint: Image.Image,
    template: CarTemplate,
    luminance_threshold: int = 18,
    passes: int = 2,
) -> Image.Image:
    """
    Remove labels, red zone boxes, cyan wireframe, and guide text from final paint.

    Uses the AI layout guide image for pixel-accurate positions, then inpaints
    from neighboring livery colors.
    """
    size = template.resolution
    img = _resize_to(paint, size)
    pre_strip = np.array(img.convert("RGBA"))

    for _ in range(max(1, passes)):
        arr = np.array(img.convert("RGBA"))
        mask_arr = np.array(
            _resize_to(template.paintable_mask.convert("RGBA"), size).convert("L")
        ) > 128
        lum = arr[:, :, :3].max(axis=2)
        artifact = _build_guide_artifact_mask(arr, template, size)
        if not artifact.any():
            break
        _fill_artifact_pixels(
            arr,
            artifact,
            mask_arr,
            lum,
            luminance_threshold,
            template,
            pre_strip[:, :, :3],
        )
        img = Image.fromarray(arr, mode="RGBA")

    return smooth_bbox_seams(img, template, band=6, blur_radius=3, blend=0.3)


def strip_wireframe_artifacts(
    paint: Image.Image,
    template: CarTemplate,
    luminance_threshold: int = 18,
) -> Image.Image:
    """Backward-compatible alias for guide overlay removal."""
    return strip_guide_overlays(paint, template, luminance_threshold=luminance_threshold)


def strip_template_artifacts(
    paint: Image.Image,
    template: CarTemplate,
    luminance_threshold: int = 18,
    atlas_only: bool = False,
) -> Image.Image:
    """
    Remove wireframe/label colors the AI may have copied from old template references.
    Replaces cyan outlines, yellow tags, and red guide boxes with nearby livery color.

    Use atlas_only=True after regional paint overrides so livery yellows/reds are not
    mistaken for guide label colors.
    """
    size = template.resolution
    img = _resize_to(paint, size)
    arr = np.array(img.convert("RGBA"))
    mask_arr = np.array(_resize_to(template.paintable_mask.convert("RGBA"), size).convert("L")) > 128

    rgb = arr[:, :, :3].astype(np.int16)
    lum = rgb.max(axis=2)

    if atlas_only:
        artifact = _build_guide_artifact_mask(arr, template, size)
    else:
        artifact = np.zeros(mask_arr.shape, dtype=bool)
        atlas_mask = _guide_artifact_mask_from_atlas(template, size)
        if atlas_mask is not None:
            artifact |= mask_arr & atlas_mask
        artifact |= _color_artifact_mask(mask_arr, rgb)
        ref_mask = _reference_guide_artifact_mask(template, size)
        if ref_mask is not None:
            artifact |= mask_arr & ref_mask

    if not artifact.any():
        return img

    _fill_artifact_pixels(
        arr, artifact, mask_arr, lum, luminance_threshold, template, arr[:, :, :3].copy()
    )
    return Image.fromarray(arr, mode="RGBA")


def post_process_paint(image: Image.Image, sharpen: bool = True) -> Image.Image:
    """Light post-processing to crisp up AI output for in-sim use."""
    img = image.convert("RGBA")
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(1.12)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.04)
    alpha = img.split()[3]
    return Image.merge("RGBA", (*rgb.split(), alpha))


def generate_spec_map(
    paint: Image.Image,
    material_hints: Optional[dict] = None,
    template: Optional[CarTemplate] = None,
) -> Image.Image:
    """
    Auto-generate an iRacing PBR spec map from the paint design.
    R=metallic, G=roughness, B=clearcoat, A=mask
    """
    hints = material_hints or {}
    rgba = paint.convert("RGBA")
    rgb = np.array(rgba.convert("RGB"), dtype=np.float32)
    alpha = np.array(rgba.split()[3], dtype=np.float32)
    h, w = rgb.shape[:2]

    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    max_c = rgb.max(axis=2)
    min_c = rgb.min(axis=2)
    saturation = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-5), 0)

    metallic = np.zeros((h, w), dtype=np.float32)
    if hints.get("chrome"):
        metallic = np.where((luminance > 170) & (saturation < 0.25), 255.0, metallic)
        metallic = np.where(luminance > 210, 255.0, metallic)
    elif hints.get("metallic"):
        metallic = np.where((luminance > 140) & (saturation < 0.45), 200.0, metallic)
        metallic = np.where(luminance > 200, 255.0, metallic)
    else:
        metallic = np.where((luminance > 200) & (saturation < 0.2), 80.0, 0.0)

    if hints.get("matte"):
        base_rough = 210.0
    elif hints.get("gloss"):
        base_rough = 40.0
    else:
        base_rough = 90.0

    roughness = np.full((h, w), base_rough, dtype=np.float32)
    roughness = np.where(luminance < 40, np.minimum(roughness + 60, 255), roughness)
    roughness = np.where((luminance > 220) & (saturation < 0.15), 25.0, roughness)
    roughness = np.where(metallic > 128, np.minimum(roughness, 15.0), roughness)

    clearcoat = np.full((h, w), 255.0, dtype=np.float32)
    if hints.get("matte"):
        clearcoat = np.where(luminance < 80, 80.0, clearcoat)

    spec_mask = np.where((luminance < 8) & (alpha < 20), 0.0, 255.0)
    spec_mask = np.where(alpha < 30, 0.0, spec_mask)

    if template is not None:
        tmask = np.array(
            _resize_to(template.paintable_mask.convert("RGBA"), w).convert("L"),
            dtype=np.float32,
        )
        spec_mask = np.where(tmask > 128, spec_mask, 0.0)

    spec = np.stack(
        [
            np.clip(metallic, 0, 255),
            np.clip(roughness, 0, 255),
            np.clip(clearcoat, 0, 255),
            np.clip(spec_mask, 0, 255),
        ],
        axis=2,
    ).astype(np.uint8)

    return Image.fromarray(spec, mode="RGBA")


def save_tga(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="TGA")
    return path


def export_paint_files(
    paint: Image.Image,
    spec: Image.Image,
    car: IRacingCar,
    customer_id: str,
    install_to_iracing: bool = False,
    template: Optional[CarTemplate] = None,
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paint_name, spec_name = get_output_filenames(customer_id)
    session_dir = OUTPUT_DIR / f"{customer_id}_{car.folder_path.replace(chr(92), '_')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    paint_path = session_dir / paint_name
    spec_path = session_dir / spec_name

    if template is not None:
        paint = clip_paint_to_mask(paint, template)
        spec = clip_paint_to_mask(spec, template)

    save_tga(paint.convert("RGBA"), paint_path)
    save_tga(spec.convert("RGBA"), spec_path)

    result = {"paint": paint_path, "spec": spec_path, "session_dir": session_dir}

    if install_to_iracing:
        from cars_config import get_paint_install_path

        install_dir = get_paint_install_path(car, customer_id)
        install_dir.mkdir(parents=True, exist_ok=True)
        install_paint = install_dir / paint_name
        install_spec = install_dir / spec_name
        save_tga(paint.convert("RGBA"), install_paint)
        save_tga(spec.convert("RGBA"), install_spec)
        result["install_paint"] = install_paint
        result["install_spec"] = install_spec

    return result


# Backwards-compatible alias (old preview function name).
def create_preview_3d_style(paint: Image.Image, car: IRacingCar) -> Image.Image:
    return create_car_side_preview(paint, car, template=None)