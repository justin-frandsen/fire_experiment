"""
roi_overlay_figures.py

Generate figures showing detected ROIs overlaid on the source images.
Reads the saved *_roi_type.npy masks produced by roi_saliency_extractor.py
and draws semi-transparent fills + outlines. Does NOT re-detect ROIs —
it only visualizes the masks you already created.

Usage:
    - Requires the "images" directory (source images) and the
      "contour_masks" directory (*_roi_type.npy masks).
    - Run:  python roi_overlay_figures.py
    - Annotated images are written to "figures/rois".
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# DISPLAY OPTION
#   False (default): raw, matter-of-fact display. All ROI types are
#                    drawn identically (same fill alpha, same outline
#                    thickness, neutral draw order). Best for figures
#                    that document the data faithfully.
#   True:            presentation styling. Fire is emphasized with a
#                    thicker outline and drawn on top of other ROIs.
# ============================================================
HIGHLIGHT_FIRE = False

# ---- Config -------------------------------------------------
IMAGES_DIR = "images"
MASK_DIR = "contour_masks"
OUT_DIR = "figures/rois"

# Colors matched to the analysis graphs (RGB 0-1).
ROI_COLORS = {
    1: (0.843, 0.188, 0.122),   # Fire       -> #D7301F
    2: (0.302, 0.686, 0.290),   # Vegetation -> #4DAF4A
}
ROI_NAMES = {1: "Fire", 2: "Vegetation"}

FILL_ALPHA = 0.45          # transparency of the colored ROI fill
OUTLINE_THICKNESS = 2      # px thickness of ROI outlines (raw mode: all ROIs)
FIRE_OUTLINE_THICKNESS = 3 # px thickness for Fire when HIGHLIGHT_FIRE is True


def find_mask_for_image(image_name):
    """
    Locate the roi_type mask that matches a given image file.
    The extractor saves masks as '<image_name>_roi_type.npy' where
    <image_name> is the full original filename (including extension).
    """
    candidate = os.path.join(MASK_DIR, f"{image_name}_roi_type.npy")
    if os.path.exists(candidate):
        return candidate
    # Fallback: match on the stem in case extensions were stripped.
    stem = os.path.splitext(image_name)[0]
    for f in os.listdir(MASK_DIR):
        if f.endswith("_roi_type.npy") and f.startswith(stem):
            return os.path.join(MASK_DIR, f)
    return None


def build_overlay(img_rgb, roi_type):
    """
    Blend semi-transparent ROI fills onto the image and draw outlines.

    Raw mode (HIGHLIGHT_FIRE=False): every ROI type is treated identically.
    Highlight mode (HIGHLIGHT_FIRE=True): Fire gets a thicker outline and is
    drawn last so it sits on top.
    """
    overlay = img_rgb.astype(np.float32) / 255.0

    # Fill each ROI type with its color (alpha blend) — identical for all types.
    for roi_val, color in ROI_COLORS.items():
        region = roi_type == roi_val
        if not region.any():
            continue
        color_arr = np.array(color, dtype=np.float32)
        overlay[region] = (1 - FILL_ALPHA) * overlay[region] + FILL_ALPHA * color_arr

    overlay = (overlay * 255).clip(0, 255).astype(np.uint8).copy()

    # Determine outline draw order and thickness based on the display option.
    if HIGHLIGHT_FIRE:
        # Veg first, Fire last (on top), Fire thicker.
        draw_order = (2, 1)
    else:
        # Raw: neutral order, uniform thickness for every ROI type.
        draw_order = sorted(ROI_COLORS.keys())

    for roi_val in draw_order:
        region = (roi_type == roi_val).astype(np.uint8)
        if not region.any():
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        color_255 = tuple(int(c * 255) for c in ROI_COLORS[roi_val])
        if HIGHLIGHT_FIRE and roi_val == 1:
            thickness = FIRE_OUTLINE_THICKNESS
        else:
            thickness = OUTLINE_THICKNESS
        cv2.drawContours(overlay, contours, -1, color_255, thickness)

    return overlay


def present_roi_types(roi_type):
    """Return the ROI type values actually present (for the legend)."""
    return [v for v in ROI_COLORS if (roi_type == v).any()]


def save_figure(image_name, overlay, present):
    """Render the overlay with a title and legend, then save to OUT_DIR."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(overlay)
    ax.set_title(f"Detected ROIs — {image_name}", fontweight="bold")
    ax.axis("off")

    handles = [
        Patch(facecolor=ROI_COLORS[v], edgecolor="black", label=ROI_NAMES[v])
        for v in present
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=True)

    plt.tight_layout()
    stem = os.path.splitext(image_name)[0]
    suffix = "_roi_highlight" if HIGHLIGHT_FIRE else "_roi"
    out_path = os.path.join(OUT_DIR, f"{stem}{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mode = "HIGHLIGHT" if HIGHLIGHT_FIRE else "RAW"
    print(f"Display mode: {mode}")

    images = [f for f in os.listdir(IMAGES_DIR) if not f.startswith(".")]
    if not images:
        print(f"No images found in '{IMAGES_DIR}'.")
        return

    made, skipped = 0, 0
    for img_name in images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Skipping (could not read image): {img_name}")
            skipped += 1
            continue

        mask_path = find_mask_for_image(img_name)
        if mask_path is None:
            print(f"Skipping (no roi_type mask found): {img_name}")
            skipped += 1
            continue

        roi_type = np.load(mask_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Guard against any size mismatch between image and mask.
        if roi_type.shape[:2] != img_rgb.shape[:2]:
            roi_type = cv2.resize(
                roi_type, (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        overlay = build_overlay(img_rgb, roi_type)
        present = present_roi_types(roi_type)
        out_path = save_figure(img_name, overlay, present)
        print(f"Saved {out_path}")
        made += 1

    print(f"\nDone. {made} figure(s) written to '{OUT_DIR}', {skipped} skipped.")


if __name__ == "__main__":
    main()