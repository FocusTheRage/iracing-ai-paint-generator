"""
Apply per-region paint instructions from the user prompt onto the correct UV panels.

The AI backend often ignores multi-region color assignments. This module parses
prompt clauses and procedurally paints atlas regions after AI generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from template_manager import CarTemplate
from uv_atlas import CarUVAtlas, UVRegion, load_atlas_for_car

# Longest phrases first for color matching.
COLOR_PHRASES: list[tuple[str, tuple[int, int, int]]] = [
    ("safety orange", (255, 95, 0)),
    ("electric violet", (155, 70, 240)),
    ("dark purple", (55, 18, 95)),
    ("charcoal gray", (48, 48, 52)),
    ("charcoal grey", (48, 48, 52)),
    ("brushed aluminum", (175, 180, 188)),
    ("aluminum", (175, 180, 188)),
    ("orange", (255, 100, 0)),
    ("purple", (95, 35, 145)),
    ("violet", (140, 60, 220)),
    ("charcoal", (45, 45, 50)),
    ("silver", (185, 190, 200)),
    ("yellow", (230, 200, 30)),
    ("white", (245, 245, 245)),
    ("black", (12, 12, 14)),
    ("red", (200, 30, 30)),
    ("blue", (30, 80, 200)),
    ("green", (30, 160, 60)),
    ("magenta", (200, 30, 160)),
]

GRAPHICS_HINTS = (
    "lightning",
    "bolt",
    "cloud",
    "storm",
    "flame",
    "graphic",
    "logo",
    "sponsor",
    "design",
    "pattern",
    "burst",
    "swirl",
    "camo",
)

SOLID_HINTS = ("solid", "plain", "bright", "flat", "matte", "gloss", "entire", "all")

# Clause fragment → region ids (most specific first).
CLAUSE_REGION_RULES: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\brear\s+diffusers?\b", re.I), ["rear_diffuser"]),
    (re.compile(r"\bpit\s+signs?\b", re.I), ["pit_sign"]),
    (re.compile(r"\bright\s+lower\s+doors?\b", re.I), ["right_lower_door"]),
    (re.compile(r"\brear\s+bumpers?\b", re.I), ["back"]),
    (re.compile(r"\bback\s+panels?\b", re.I), ["back"]),
    (re.compile(r"\brear\s+body\b", re.I), ["back"]),
    (re.compile(r"(?<![a-z])rear(?![a-z])", re.I), ["back"]),
    (re.compile(r"(?<![a-z])backs?(?![a-z])", re.I), ["back"]),
    (re.compile(r"\bfront\s+bumper\s+corners?\b", re.I), [
        "front_corner_upper_left",
        "front_corner_upper_bumper_right",
    ]),
    (re.compile(r"\bleft\s+bumper\s+corner\b", re.I), ["front_corner_upper_left"]),
    (re.compile(r"\bright\s+bumper\s+corner\b", re.I), ["front_corner_upper_bumper_right"]),
    (re.compile(r"\bleft\s+front\s+corners?\b", re.I), ["front_corner_upper_left"]),
    (re.compile(r"\bright\s+front\s+corners?\b", re.I), [
        "front_corner_upper_right",
        "front_corner_lower_right",
    ]),
    (re.compile(r"\bfront\s+left\s+quarters?\b", re.I), ["front_corner_upper_right"]),
    (re.compile(r"\bfront\s+right\s+quarters?\b", re.I), ["front_corner_lower_right"]),
    (re.compile(r"\bleft\s+front\s+quarters?\b", re.I), ["front_corner_upper_right"]),
    (re.compile(r"\bright\s+front\s+quarters?\b", re.I), ["front_corner_lower_right"]),
    (re.compile(r"\bright\s+rear\s+quarters?\b", re.I), ["rear_corner_lower"]),
    (re.compile(r"\brear\s+quarters?\b", re.I), ["rear_corner_lower"]),
    (re.compile(r"\brear\s+corners?\b", re.I), ["rear_corner_lower"]),
    (re.compile(r"\bquarter\s+panels?\b", re.I), ["rear_corner_lower"]),
    (re.compile(r"\brear\s+decks?\b", re.I), ["trunk"]),
    (re.compile(r"\bdeck\s+lids?\b", re.I), ["trunk"]),
    (re.compile(r"\btrunks?\b", re.I), ["trunk"]),
    (re.compile(r"\bdriver(?:'?s)?\s+doors?\b", re.I), ["left_side_door"]),
    (re.compile(r"\bdriver\s+side\s+doors?\b", re.I), ["left_side_door"]),
    (re.compile(r"\bleft\s+side\s+doors?\b", re.I), ["left_side_door"]),
    (re.compile(r"\bleft\s+doors?\b", re.I), ["left_side_door"]),
    (re.compile(r"\bpassenger(?:'?s)?\s+doors?\b", re.I), ["right_side_door"]),
    (re.compile(r"\bpassenger\s+side\s+doors?\b", re.I), ["right_side_door"]),
    (re.compile(r"\bright\s+side\s+doors?\b", re.I), ["right_side_door"]),
    (re.compile(r"\bright\s+doors?\b", re.I), ["right_side_door"]),
    (re.compile(r"\bfront\s+corners?\b", re.I), [
        "front_corner_upper_right",
        "front_corner_lower_right",
    ]),
    (re.compile(r"\bfront\s+splitters?\b", re.I), ["front_splitter_right"]),
    (re.compile(r"\bdriver\s+sides?\b", re.I), ["left_side_door", "pit_sign"]),
    (re.compile(r"\bpassenger\s+sides?\b", re.I), ["right_side_door", "right_lower_door"]),
    (re.compile(r"\bleft\s+sides?\b", re.I), ["left_side_door"]),
    (re.compile(r"\bright\s+sides?\b", re.I), ["right_side_door"]),
    (re.compile(r"\bfront\s+bumper\b(?!\s+corners?)", re.I), ["front_bumper"]),
    (re.compile(r"\bhoods?\b", re.I), ["hood"]),
    (re.compile(r"\broofs?\b", re.I), ["roof"]),
]

# Most-specific regions claim pixels first (prevents bumper/corner color bleed).
REGION_CLAIM_PRIORITY: list[str] = [
    "front_corner_upper_right",
    "front_corner_lower_right",
    "rear_corner_lower",
    "left_side_door",
    "right_side_door",
    "right_lower_door",
    "pit_sign",
    "front_corner_upper_left",
    "front_corner_upper_bumper_right",
    "front_bumper",
    "front_splitter_right",
    "back",
    "rear_diffuser",
    "trunk",
    "hood",
    "roof",
]


@dataclass
class RegionStyle:
    fill_color: tuple[int, int, int] | None = None
    diagonal_stripes: bool = False
    stripe_color: tuple[int, int, int] = (12, 12, 14)
    door_number: str | None = None
    number_color: tuple[int, int, int] = (245, 245, 245)
    preserve_ai: bool = False
    solid: bool = False


def _color_in_text(text: str) -> tuple[int, int, int] | None:
    lower = text.lower()
    qualifier = re.search(
        r"\b(?:solid|plain|matte|bright|deep|flat)\s+((?:[\w]+\s+){0,2})(\w+)\b",
        lower,
    )
    if qualifier:
        phrase = f"{qualifier.group(1)}{qualifier.group(2)}".strip()
        for name, rgb in COLOR_PHRASES:
            if name in phrase or phrase in name:
                return rgb
    before_with = lower.split(" with ")[0]
    for name, rgb in COLOR_PHRASES:
        if name in before_with:
            return rgb
    return None


def _regions_in_clause(clause: str) -> list[str]:
    """Return regions from the first (most specific) matching rule only."""
    for pattern, region_ids in CLAUSE_REGION_RULES:
        if pattern.search(clause):
            return list(region_ids)
    return []


def _clause_style(clause: str) -> RegionStyle:
    style = RegionStyle()
    lower = clause.lower()
    num_match = re.search(r"\bnumber\s+(\d{1,3})\b", clause, re.I)

    # Body color comes from text before "number" so "88 in white" tints the digits only.
    color_source = clause.split("number", 1)[0] if num_match else clause
    style.fill_color = _color_in_text(color_source)
    style.diagonal_stripes = (
        ("diagonal stripe" in lower or "striped" in lower)
        and "pinstripe" not in lower
    )
    style.solid = any(h in lower for h in SOLID_HINTS)
    if "aluminum" in lower or "brushed" in lower:
        style.solid = True
        style.preserve_ai = False
    else:
        style.preserve_ai = any(h in lower for h in GRAPHICS_HINTS) and not style.solid

    if num_match:
        style.door_number = num_match.group(1)
        num_color = None
        if re.search(r"\bwhite\s+number\b", lower) or re.search(
            r"\bnumber\s+\d+.*\bin\s+white\b", lower
        ):
            num_color = (245, 245, 245)
        elif re.search(r"\bblack\s+number\b", lower) or re.search(
            r"\bnumber\s+\d+.*\bin\s+black\b", lower
        ):
            num_color = (12, 12, 14)
        style.number_color = num_color or _color_in_text(clause) or (245, 245, 245)
    return style


def parse_regional_instructions(prompt: str) -> dict[str, RegionStyle]:
    """Map atlas region ids to paint instructions parsed from the prompt."""
    assignments: dict[str, RegionStyle] = {}
    clauses = re.split(r"[.;\n]+", prompt)

    for clause in clauses:
        clause = clause.strip()
        if len(clause) < 4:
            continue
        region_ids = _regions_in_clause(clause)
        if not region_ids:
            continue
        style = _clause_style(clause)
        if style.preserve_ai and not style.solid:
            for rid in region_ids:
                assignments[rid] = RegionStyle(preserve_ai=True)
            continue
        if style.fill_color is None and not style.door_number and not style.diagonal_stripes:
            continue

        for rid in region_ids:
            prev = assignments.get(rid)
            if prev is None:
                assignments[rid] = style
            else:
                merged = RegionStyle(
                    fill_color=style.fill_color or prev.fill_color,
                    diagonal_stripes=style.diagonal_stripes or prev.diagonal_stripes,
                    stripe_color=style.stripe_color,
                    door_number=style.door_number or prev.door_number,
                    number_color=style.number_color,
                    preserve_ai=style.preserve_ai and prev.preserve_ai,
                    solid=style.solid or prev.solid,
                )
                assignments[rid] = merged

    return assignments


def _global_base_color(prompt: str) -> tuple[int, int, int] | None:
    base_match = re.search(
        r"(?:matte\s+)?(?:black|white|red|blue|orange|purple|charcoal|violet)\s+base\b",
        prompt,
        re.I,
    )
    if not base_match:
        return None
    return _color_in_text(base_match.group(0))


def apply_base_color(
    assignments: dict[str, RegionStyle],
    prompt: str,
    atlas: CarUVAtlas,
) -> dict[str, RegionStyle]:
    """Fill unassigned regions with a global base color from the prompt."""
    base_color = _global_base_color(prompt)
    if not base_color:
        return assignments
    merged = dict(assignments)
    for region in atlas.regions:
        existing = merged.get(region.id)
        if existing is not None:
            continue
        merged[region.id] = RegionStyle(fill_color=base_color, solid=True)
    return merged


def _apply_door_base_fill(
    assignments: dict[str, RegionStyle],
    prompt: str,
) -> dict[str, RegionStyle]:
    """Door clauses with only a number still get the global base body color."""
    base_color = _global_base_color(prompt)
    if not base_color:
        return assignments
    merged = dict(assignments)
    for rid, style in assignments.items():
        if not style.door_number or style.fill_color is not None or style.preserve_ai:
            continue
        if rid not in ("left_side_door", "right_side_door"):
            continue
        merged[rid] = RegionStyle(
            fill_color=base_color,
            door_number=style.door_number,
            number_color=style.number_color,
            solid=True,
        )
    return merged


def _raw_region_mask(
    template: CarTemplate,
    region: UVRegion,
    size: int,
) -> np.ndarray:
    paintable = np.array(
        template.paintable_mask.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    ) > 128
    x0, y0, x1, y1 = region.bbox
    box = np.zeros_like(paintable, dtype=bool)
    px0, py0 = int(x0 * size), int(y0 * size)
    px1, py1 = int(x1 * size), int(y1 * size)
    box[py0:py1, px0:px1] = True
    return paintable & box


def _exclusive_region_masks(
    template: CarTemplate,
    atlas: CarUVAtlas,
    region_ids: set[str],
    size: int,
) -> dict[str, np.ndarray]:
    """Assign each paintable pixel to at most one region (specific beats general)."""
    paintable = np.array(
        template.paintable_mask.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    ) > 128
    claimed = np.zeros_like(paintable, dtype=bool)
    masks: dict[str, np.ndarray] = {}

    priority = [rid for rid in REGION_CLAIM_PRIORITY if rid in region_ids]
    for rid in atlas.regions:
        if rid.id in region_ids and rid.id not in priority:
            priority.append(rid.id)

    for rid in priority:
        region = atlas.region_by_id(rid)
        if region is None:
            continue
        raw = _raw_region_mask(template, region, size)
        exclusive = raw & ~claimed
        if exclusive.any():
            masks[rid] = exclusive
            claimed |= exclusive
    return masks


def _apply_fill(arr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    arr[mask, 0] = color[0]
    arr[mask, 1] = color[1]
    arr[mask, 2] = color[2]
    arr[mask, 3] = 255


def _apply_diagonal_stripes(
    arr: np.ndarray,
    mask: np.ndarray,
    base: tuple[int, int, int],
    stripe: tuple[int, int, int],
    width: int = 34,
) -> None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    parity = ((xs + ys) // width) % 2 == 0
    arr[ys[parity], xs[parity], 0] = stripe[0]
    arr[ys[parity], xs[parity], 1] = stripe[1]
    arr[ys[parity], xs[parity], 2] = stripe[2]
    arr[ys[parity], xs[parity], 3] = 255
    arr[ys[~parity], xs[~parity], 0] = base[0]
    arr[ys[~parity], xs[~parity], 1] = base[1]
    arr[ys[~parity], xs[~parity], 2] = base[2]
    arr[ys[~parity], xs[~parity], 3] = 255


def _draw_number_on_region(
    image: Image.Image,
    region: UVRegion,
    number: str,
    color: tuple[int, int, int],
    size: int,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arialbd.ttf", 200)
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = region.bbox
    cx = int((x0 + x1) * size / 2)
    cy = int((y0 + y1) * size / 2)
    bbox = draw.textbbox((0, 0), number, font=font, stroke_width=6)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        (cx - tw // 2, cy - th // 2),
        number,
        fill=color + (255,),
        font=font,
        stroke_width=6,
        stroke_fill=(0, 0, 0, 255),
    )
    return image


def apply_regional_overrides(
    paint: Image.Image,
    prompt: str,
    template: CarTemplate,
    *,
    no_text: bool = False,
) -> Image.Image:
    """
    Paint atlas regions per the user's prompt after AI generation.
    Ensures region-specific colors, stripes, and door numbers land correctly.
    """
    if template.uv_atlas is None:
        atlas = load_atlas_for_car(template.car.folder_path)
    else:
        atlas = template.uv_atlas
    if atlas is None:
        return paint

    assignments = parse_regional_instructions(prompt)
    assignments = apply_base_color(assignments, prompt, atlas)
    assignments = _apply_door_base_fill(assignments, prompt)
    if not assignments:
        return paint

    size = template.resolution
    img = paint.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img)

    paint_ids = {
        rid
        for rid, style in assignments.items()
        if not (style.preserve_ai and not style.solid)
        and (style.fill_color is not None or style.diagonal_stripes)
    }
    region_masks = _exclusive_region_masks(template, atlas, paint_ids, size)

    for region_id, style in assignments.items():
        region = atlas.region_by_id(region_id)
        if region is None:
            continue
        if style.preserve_ai and not style.solid:
            continue
        if style.fill_color is None and not style.diagonal_stripes:
            if not style.door_number:
                continue

        mask = region_masks.get(region_id)
        if mask is None or not mask.any():
            continue

        if style.diagonal_stripes and style.fill_color:
            _apply_diagonal_stripes(arr, mask, style.fill_color, style.stripe_color)
        elif style.fill_color:
            _apply_fill(arr, mask, style.fill_color)

    result = Image.fromarray(arr, mode="RGBA")

    from ai_backend import parse_prompt_constraints

    constraints = parse_prompt_constraints(prompt, no_text_option=no_text)
    if constraints.allow_car_number:
        for region_id, style in assignments.items():
            if not style.door_number:
                continue
            region = atlas.region_by_id(region_id)
            if region is None:
                continue
            result = _draw_number_on_region(
                result, region, style.door_number, style.number_color, size
            )

    return result


def regional_override_summary(prompt: str, atlas: CarUVAtlas) -> str:
    """Short debug summary of parsed regional instructions."""
    assignments = parse_regional_instructions(prompt)
    if not assignments:
        return ""
    lines = ["PARSED REGIONAL OVERRIDES (applied after AI generation):"]
    for rid, style in sorted(assignments.items()):
        region = atlas.region_by_id(rid)
        name = region.label if region else rid
        parts = []
        if style.fill_color:
            parts.append(f"fill RGB{style.fill_color}")
        if style.diagonal_stripes:
            parts.append("diagonal stripes")
        if style.door_number:
            parts.append(f"number {style.door_number}")
        if style.preserve_ai:
            parts.append("AI graphics preserved")
        lines.append(f"  - {name}: {', '.join(parts)}")
    return "\n".join(lines)