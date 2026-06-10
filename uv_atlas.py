"""
Per-car UV atlas: semantic body-part labels mapped to UV island coordinates.

Enables the AI to understand where "rear bumper", "driver door", etc. live on the
flat iRacing template for each vehicle.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

ATLAS_DIR = Path(__file__).parent / "templates" / "atlas"
MEGA_BLOB_AREA_RATIO = 0.45


@dataclass
class UVRegion:
    """One labeled paintable zone on the UV template."""

    id: str
    label: str
    display_name: str
    bbox: tuple[float, float, float, float]
    aliases: list[str] = field(default_factory=list)
    in_game_hint: str = ""
    island_ids: list[int] = field(default_factory=list)
    label_anchor: tuple[float, float] = (0.0, 0.0)

    def bbox_text(self) -> str:
        x0, y0, x1, y1 = self.bbox
        return f"UV {x0:.0%}-{x1:.0%} horizontal, {y0:.0%}-{y1:.0%} vertical"


@dataclass
class CarUVAtlas:
    """Semantic map of UV regions for one iRacing car."""

    car_name: str
    folder_path: str
    resolution: int
    regions: list[UVRegion]
    alias_groups: dict[str, list[str]] = field(default_factory=dict)
    version: int = 1
    reference_png: str = ""
    layout_lines: list[str] = field(default_factory=list)

    def region_by_id(self, region_id: str) -> Optional[UVRegion]:
        for region in self.regions:
            if region.id == region_id:
                return region
        return None

    def regions_for_alias(self, alias: str) -> list[UVRegion]:
        alias_l = alias.lower().strip()
        matched: list[UVRegion] = []

        if alias_l in self.alias_groups:
            for rid in self.alias_groups[alias_l]:
                region = self.region_by_id(rid)
                if region:
                    matched.append(region)
            if matched:
                return matched

        for region in self.regions:
            if alias_l == region.id.replace("_", " "):
                matched.append(region)
            elif alias_l in [a.lower() for a in region.aliases]:
                matched.append(region)
            elif alias_l in region.display_name.lower():
                matched.append(region)
        return matched


def _safe_slug(folder_path: str) -> str:
    return folder_path.replace("\\", "_").replace("/", "_")


def load_atlas_for_car(folder_path: str) -> Optional[CarUVAtlas]:
    """Load hand-authored atlas JSON for a car if it exists."""
    path = ATLAS_DIR / f"{_safe_slug(folder_path)}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        regions = [
            UVRegion(
                id=r["id"],
                label=r["label"],
                display_name=r["display_name"],
                bbox=tuple(r["bbox"]),
                aliases=r.get("aliases", []),
                in_game_hint=r.get("in_game_hint", ""),
                island_ids=r.get("island_ids", []),
                label_anchor=tuple(r.get("label_anchor", [0.0, 0.0])),
            )
            for r in raw["regions"]
        ]
        return CarUVAtlas(
            car_name=raw.get("car_name", ""),
            folder_path=raw.get("folder_path", folder_path),
            resolution=raw.get("resolution", 2048),
            regions=regions,
            alias_groups=raw.get("alias_groups", {}),
            version=raw.get("version", 1),
            reference_png=raw.get("reference_png", ""),
            layout_lines=raw.get("layout_lines", []),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to load UV atlas %s: %s", path, exc)
        return None


# Natural-language fragments → alias group keys.
_INTENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brear\s*bumpers?\b", re.I), "rear"),
    (re.compile(r"\brear\s*quarters?\b", re.I), "rear_quarter"),
    (re.compile(r"\bquarter\s*panels?\b", re.I), "quarter"),
    (re.compile(r"\bbehind\s+(?:the\s+)?doors?\b", re.I), "rear_quarter"),
    (re.compile(r"\bsail\s*panels?\b", re.I), "rear_quarter"),
    (re.compile(r"\brear\s*deck\b", re.I), "rear"),
    (re.compile(r"\btrunks?\b", re.I), "trunk"),
    (re.compile(r"\btail\s*lights?\b", re.I), "rear"),
    (re.compile(r"\b(?:back|rear)\s+of\s+(?:the\s+)?car\b", re.I), "rear"),
    (re.compile(r"\brear\b", re.I), "rear"),
    (re.compile(r"\bfront\s*bumpers?\b", re.I), "front"),
    (re.compile(r"\bfront\s*bumper\s*corners?\b", re.I), "front_bumper_corner"),
    (re.compile(r"\bfront\s*corners?\b", re.I), "corner"),
    (re.compile(r"\bfront\s*quarters?\b", re.I), "corner"),
    (re.compile(r"\bfront\s*splitters?\b", re.I), "front"),
    (re.compile(r"\bgrilles?\b", re.I), "front"),
    (re.compile(r"\brear\s*diffusers?\b", re.I), "diffuser"),
    (re.compile(r"\bdiffusers?\b", re.I), "diffuser"),
    (re.compile(r"\bbacks?\b", re.I), "back"),
    (re.compile(r"\bhoods?\b", re.I), "hood"),
    (re.compile(r"\broofs?\b", re.I), "roof"),
    (re.compile(r"\bpassenger\s+(?:side\s+)?doors?\b", re.I), "passenger_side"),
    (re.compile(r"\bright\s+(?:side\s+)?doors?\b", re.I), "passenger_side"),
    (re.compile(r"\bpassenger\s+sides?\b", re.I), "passenger_side"),
    (re.compile(r"\bdriver\s+(?:side\s+)?doors?\b", re.I), "driver_side"),
    (re.compile(r"\bleft\s+(?:side\s+)?doors?\b", re.I), "driver_side"),
    (re.compile(r"\bnumber\s+panels?\b", re.I), "driver_side"),
    (re.compile(r"\bdriver\s+sides?\b", re.I), "driver_side"),
    (re.compile(r"\bfenders?\b", re.I), "fender"),
    (re.compile(r"\brockers?\b", re.I), "rocker"),
]


def resolve_regions_from_prompt(prompt: str, atlas: CarUVAtlas) -> list[UVRegion]:
    """Map user prompt phrases to labeled UV regions."""
    found: list[UVRegion] = []
    seen: set[str] = set()
    prompt_l = prompt.lower()

    for pattern, group in _INTENT_PATTERNS:
        if pattern.search(prompt):
            for region in atlas.regions_for_alias(group):
                if region.id not in seen:
                    seen.add(region.id)
                    found.append(region)

    for region in atlas.regions:
        for alias in region.aliases:
            if len(alias) < 4:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", prompt_l):
                if region.id not in seen:
                    seen.add(region.id)
                    found.append(region)

    return found


def format_atlas_reference(atlas: CarUVAtlas, highlight: Optional[list[UVRegion]] = None) -> str:
    """Build a text block listing all UV regions for the AI prompt."""
    highlight_ids = {r.id for r in (highlight or [])}
    lines = [
        "OFFICIAL UV REGION MAP (2048x2048 flat template — NOT a side-view car):",
        "The reference image shows cyan wireframe outlines on gray UV panels — use them for placement.",
        "Layout on the flat sheet (do NOT rearrange panels):",
    ]
    if atlas.layout_lines:
        lines.extend(f"  {line}" for line in atlas.layout_lines)
    else:
        lines.extend([
            "  TOP-CENTER = front bumper | TOP-RIGHT = back + rear bumper + diffuser",
            "  UPPER-MIDDLE = driver door | CENTER = roof | RIGHT-MIDDLE = hood",
            "  BEHIND DOORS = rear quarter panel (between door and rear corner)",
            "  BOTTOM-CENTER = passenger door | LEFT SIDE = rear quarters + trunk",
            "  FRONT BUMPER ROW = center grille + FRONT BUMPER CORNER end caps (left/right)",
            "  FRONT CORNERS = side quarter panels only (upper right + lower right)",
        ])
    lines.append("")
    for region in atlas.regions:
        marker = ">>>" if region.id in highlight_ids else "   "
        hint = region.in_game_hint.strip()
        hint_suffix = f" — {hint}" if hint else ""
        lines.append(f"{marker} [{region.label}] {region.display_name}{hint_suffix}")
    if highlight:
        lines.append("")
        lines.append("USER TARGETED THESE REGIONS (apply their instructions HERE):")
        for region in highlight:
            lines.append(f"  - {region.label} ({region.display_name})")
    return "\n".join(lines)


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _subtle_outline_overlay(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Dark gray panel edge lines — visible to AI, unlikely to appear in final livery."""
    from template_manager import _align_layer

    aligned = _align_layer(size, mask.convert("L"))
    edges = aligned.filter(ImageFilter.FIND_EDGES).convert("L")
    edge_mask = edges.point(lambda p: 255 if p > 18 else 0)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    color = Image.new("RGBA", size, (88, 92, 100, 110))
    overlay.paste(color, mask=edge_mask)
    return overlay


def build_ai_paint_reference(
    paintable_mask: Image.Image,
    atlas: Optional[CarUVAtlas] = None,
    size: int = 2048,
) -> Image.Image:
    """
    AI layout reference: panel shapes and region boundaries without text labels,
    bright cyan wireframe, or sponsor decals. Keeps paint on the correct UV panels.
    """
    guide = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    mask_l = paintable_mask.convert("L").resize((size, size), Image.Resampling.LANCZOS)

    panels = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    panels.paste(Image.new("RGBA", (size, size), (52, 54, 58, 255)), mask=mask_l)
    guide = Image.alpha_composite(guide, panels)

    if atlas is not None and atlas.reference_png:
        ref_path = ATLAS_DIR / atlas.reference_png
        if ref_path.exists():
            ref = np.array(Image.open(ref_path).convert("RGB").resize((size, size)))
            white = (ref[:, :, 0] > 245) & (ref[:, :, 1] > 245) & (ref[:, :, 2] > 245)
            hint = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            hint_arr = np.array(hint)
            hint_arr[white] = [175, 178, 185, 95]
            guide = Image.alpha_composite(guide, Image.fromarray(hint_arr, mode="RGBA"))

    guide = Image.alpha_composite(guide, _subtle_outline_overlay(mask_l, (size, size)))

    if atlas is not None:
        draw = ImageDraw.Draw(guide)
        for region in atlas.regions:
            x0, y0, x1, y1 = region.bbox
            px0, py0 = int(x0 * size), int(y0 * size)
            px1, py1 = int(x1 * size), int(y1 * size)
            draw.rectangle((px0, py0, px1, py1), outline=(72, 76, 84, 70), width=1)

            if region.label_anchor[0] > 0 or region.label_anchor[1] > 0:
                ax = int(region.label_anchor[0] * size)
                ay = int(region.label_anchor[1] * size)
                r = 5
                draw.ellipse((ax - r, ay - r, ax + r, ay + r), fill=(68, 72, 80, 90))

    return guide


def _draw_region_labels(draw: ImageDraw.ImageDraw, atlas: CarUVAtlas, size: int) -> None:
    label_font = _load_font(16)
    pad = 4
    for region in atlas.regions:
        x0, y0, x1, y1 = region.bbox
        px0, py0 = int(x0 * size), int(y0 * size)
        px1, py1 = int(x1 * size), int(y1 * size)
        draw.rectangle((px0, py0, px1, py1), outline=(220, 60, 60, 140), width=2)

        tag = region.label
        tb = draw.textbbox((0, 0), tag, font=label_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        if region.label_anchor[0] > 0 or region.label_anchor[1] > 0:
            ax, ay = region.label_anchor
            lx = max(8, min(int(ax * size) - tw // 2 - pad, size - tw - pad * 2 - 8))
            ly = max(8, min(int(ay * size) - th - pad * 2, size - th - pad * 2 - 8))
        else:
            cx = (px0 + px1) // 2
            cy = (py0 + py1) // 2
            lx = max(8, min(cx - tw // 2 - pad, size - tw - pad * 2 - 8))
            ly = max(8, min(cy - th // 2 - pad, size - th - pad * 2 - 8))
        draw.rectangle(
            (lx, ly, lx + tw + pad * 2, ly + th + pad * 2),
            fill=(220, 180, 20, 235),
            outline=(40, 35, 10, 255),
        )
        draw.text((lx + pad, ly + pad), tag, fill=(20, 18, 8, 255), font=label_font)


def build_ai_layout_guide(
    paintable_mask: Image.Image,
    atlas: CarUVAtlas,
    size: int = 2048,
) -> Image.Image:
    """
    Reference image sent to the AI: full labeled UV map with cyan wireframe.
    No sponsor decals or title text — only layout cues the model must follow.
    """
    from template_manager import _mask_outline_overlay

    guide = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    mask_l = paintable_mask.convert("L").resize((size, size), Image.Resampling.LANCZOS)

    panels = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    panels.paste(Image.new("RGBA", (size, size), (48, 50, 55, 255)), mask=mask_l)
    guide = Image.alpha_composite(guide, panels)

    if atlas.reference_png:
        ref_path = ATLAS_DIR / atlas.reference_png
        if ref_path.exists():
            ref = np.array(Image.open(ref_path).convert("RGB").resize((size, size)))
            white = (ref[:, :, 0] > 245) & (ref[:, :, 1] > 245) & (ref[:, :, 2] > 245)
            hint = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            hint_arr = np.array(hint)
            hint_arr[white] = [210, 210, 220, 70]
            guide = Image.alpha_composite(guide, Image.fromarray(hint_arr, mode="RGBA"))

    guide = Image.alpha_composite(guide, _mask_outline_overlay(mask_l, (size, size)))
    draw = ImageDraw.Draw(guide)
    _draw_region_labels(draw, atlas, size)
    return guide


def build_ai_generation_guide(
    paintable_mask: Image.Image,
    atlas: Optional[CarUVAtlas] = None,
    size: int = 2048,
) -> Image.Image:
    """
    Reference image sent to the AI: cyan wireframe + gray panels only.
    No yellow labels, red zone boxes, or text — prevents the model copying guides.
    Atlas is optional (unused here) — same wire+panel guide for every car.
    """
    from template_manager import _mask_outline_overlay

    guide = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    mask_l = paintable_mask.convert("L").resize((size, size), Image.Resampling.LANCZOS)

    panels = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    panels.paste(Image.new("RGBA", (size, size), (48, 50, 55, 255)), mask=mask_l)
    guide = Image.alpha_composite(guide, panels)

    guide = Image.alpha_composite(guide, _mask_outline_overlay(mask_l, (size, size)))
    return guide


def build_bbox_seam_mask(atlas: CarUVAtlas, size: int = 2048, band: int = 22) -> np.ndarray:
    """Wide band along every atlas bbox edge where AI panel frames appear."""
    mask = np.zeros((size, size), dtype=bool)
    for region in atlas.regions:
        x0, y0, x1, y1 = region.bbox
        px0, py0 = int(x0 * size), int(y0 * size)
        px1, py1 = int(x1 * size), int(y1 * size)
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(size, px1), min(size, py1)
        if px1 <= px0 or py1 <= py0:
            continue
        for t in range(band):
            if py0 + t < py1:
                mask[py0 + t, px0:px1] = True
            if py1 - 1 - t >= py0:
                mask[py1 - 1 - t, px0:px1] = True
            if px0 + t < px1:
                mask[py0:py1, px0 + t] = True
            if px1 - 1 - t >= px0:
                mask[py0:py1, px1 - 1 - t] = True
    return mask


def build_guide_strip_mask(atlas: CarUVAtlas, size: int = 2048) -> np.ndarray:
    """
    Pixel mask of label badges, label text, and red zone-box outlines.
    Used to unconditionally erase guide elements from AI output.
    """
    mask_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask_img)
    label_font = _load_font(16)
    pad = 4
    box_width = 6

    for region in atlas.regions:
        x0, y0, x1, y1 = region.bbox
        px0, py0 = int(x0 * size), int(y0 * size)
        px1, py1 = int(x1 * size), int(y1 * size)
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(size, px1), min(size, py1)
        if px1 > px0 and py1 > py0:
            draw.rectangle(
                (px0, py0, px1, py1),
                outline=255,
                width=box_width,
            )

        tag = region.label
        tb = draw.textbbox((0, 0), tag, font=label_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        if region.label_anchor[0] > 0 or region.label_anchor[1] > 0:
            ax, ay = region.label_anchor
            lx = max(8, min(int(ax * size) - tw // 2 - pad, size - tw - pad * 2 - 8))
            ly = max(8, min(int(ay * size) - th - pad * 2, size - th - pad * 2 - 8))
        else:
            cx = (px0 + px1) // 2
            cy = (py0 + py1) // 2
            lx = max(8, min(cx - tw // 2 - pad, size - tw - pad * 2 - 8))
            ly = max(8, min(cy - th // 2 - pad, size - th - pad * 2 - 8))
        lx2 = min(size, lx + tw + pad * 2)
        ly2 = min(size, ly + th + pad * 2)
        draw.rectangle((lx, ly, lx2, ly2), fill=255)
        draw.text((lx + pad, ly + pad), tag, fill=255, font=label_font)

    dilated = np.array(mask_img.filter(ImageFilter.MaxFilter(9))) > 128
    return dilated


def build_labeled_guide(
    paintable_mask: Image.Image,
    atlas: CarUVAtlas,
    paintable_reference: Optional[Image.Image] = None,
    size: int = 2048,
) -> Image.Image:
    """
    Human-readable labeled UV map shown in the UI preview path.
    Includes sponsor decal overlay and title text.
    """
    from template_manager import _mask_outline_overlay

    guide = Image.new("RGBA", (size, size), (22, 24, 28, 255))
    mask_l = paintable_mask.convert("L").resize((size, size), Image.Resampling.LANCZOS)

    panels = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    panels.paste(Image.new("RGBA", (size, size), (48, 50, 55, 255)), mask=mask_l)
    guide = Image.alpha_composite(guide, panels)

    if atlas.reference_png:
        ref_path = ATLAS_DIR / atlas.reference_png
        if ref_path.exists():
            ref = np.array(Image.open(ref_path).convert("RGB").resize((size, size)))
            white = (ref[:, :, 0] > 245) & (ref[:, :, 1] > 245) & (ref[:, :, 2] > 245)
            hint = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            hint_arr = np.array(hint)
            hint_arr[white] = [210, 210, 220, 55]
            guide = Image.alpha_composite(guide, Image.fromarray(hint_arr, mode="RGBA"))

    if paintable_reference is not None:
        pa = paintable_reference.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
        pa_arr = np.array(pa)
        pa_arr[:, :, 3] = np.clip(pa_arr[:, :, 3] * 0.18, 0, 60).astype(np.uint8)
        guide = Image.alpha_composite(guide, Image.fromarray(pa_arr, mode="RGBA"))

    guide = Image.alpha_composite(guide, _mask_outline_overlay(mask_l, (size, size)))

    draw = ImageDraw.Draw(guide)
    title_font = _load_font(22)
    small_font = _load_font(12)
    draw.text(
        (16, 12),
        f"{atlas.car_name} — labeled UV map",
        fill=(200, 210, 220, 255),
        font=title_font,
    )
    draw.text(
        (16, 36),
        f"UV region map for {atlas.car_name}",
        fill=(130, 140, 155, 255),
        font=small_font,
    )
    _draw_region_labels(draw, atlas, size)
    return guide


def sync_atlas_panel_regions(atlas: CarUVAtlas) -> dict[str, tuple[float, float, float, float]]:
    """Convert atlas regions into panel_regions dict for demo generator compatibility."""
    mapping = {
        "hood": "hood",
        "roof": "roof",
        "left_side_door": "driver_side",
        "right_side_door": "passenger_side",
        "right_lower_door": "passenger_side",
        "front_bumper": "front_bumper",
        "trunk": "rear",
        "rear_corner_lower": "rear",
        "back": "rear",
        "rear_diffuser": "rear_diffuser",
        "front_corner_upper_left": "front_bumper",
        "front_corner_upper_bumper_right": "front_bumper",
        "front_corner_upper_right": "hood",
        "front_corner_lower_right": "passenger_side",
        "pit_sign": "driver_side",
        "front_splitter_right": "front_bumper",
    }
    panels: dict[str, tuple[float, float, float, float]] = {}
    for region in atlas.regions:
        key = mapping.get(region.id, region.id)
        if key not in panels:
            panels[key] = region.bbox
    return panels