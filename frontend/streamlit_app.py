import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import colorsys

from backend.services.pipeline.inference import AnnotationPipeline
from backend.utils.mask_utils import resize_mask
from backend.services.export.yolo_seg_export import (
    initialize_dataset_metadata,
    save_training_image,
    save_yolo_segmentation,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _distinct_colors(n):
    """Return n visually distinct BGR colors using HSV spread."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def build_all_objects_overlay(image_np, results, h, w, selected_idx=None):
    """
    Draw every mask with a unique color.
    The selected object gets a thick white border highlight.
    Returns a uint8 RGB image.
    """
    overlay = image_np.copy().astype(np.float32)
    colors = _distinct_colors(len(results))

    for i, obj in enumerate(results):
        mask = obj.get("segmentation")
        if mask is None:
            continue
        if mask.shape != (h, w):
            mask = resize_mask(mask, (h, w))
        mask = mask.astype(bool)
        color = np.array(colors[i], dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.45 + color * 0.55

    # Highlight selected object with a bright border
    if selected_idx is not None:
        sel_mask = results[selected_idx].get("segmentation")
        if sel_mask is not None:
            if sel_mask.shape != (h, w):
                sel_mask = resize_mask(sel_mask, (h, w))
            sel_mask_u8 = sel_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(sel_mask_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 3)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def crop_object(image_np, mask, h, w, pad=6):
    """Return a tight crop of image_np around the masked object."""
    if mask.shape != (h, w):
        mask = resize_mask(mask, (h, w))
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
    return image_np[y0:y1, x0:x1]


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="SAM3 Industrial Labeler")
st.title("🔬 SAM3 Industrial Labeling System")

# ── load model (cached across reruns) ────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SAM3 model…")
def load_pipeline():
    return AnnotationPipeline()

pipeline = load_pipeline()
initialize_dataset_metadata()

# ── session state ─────────────────────────────────────────────────────────────

for key, default in [("results", None), ("annotations", []),
                     ("last_file", None), ("selected", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Image Upload")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    st.divider()
    st.header("⚙️ Segmentation Tuning")
    recall_mode_label = st.selectbox(
        "Recall Mode",
        options=["Balanced", "High Recall", "Max Recall", "Industrial Parts"],
        index=1,
        help=(
            "**Industrial Parts** — tuned for cylinder head gaskets, carburetors "
            "and similar metallic components: denser point grid, lower thresholds, "
            "5× preprocessing variants (CLAHE + edge-enhanced + gamma + sharpen). "
            "Slower but finds thin and complex-shaped parts."
        ),
    )
    min_area_px = st.number_input(
        "Min object area (pixels)",
        min_value=5,
        max_value=5000,
        value=15 if recall_mode_label == "Industrial Parts" else 30,
        step=5,
        help="Lower this to keep tiny fragments. Raise to remove noise.",
    )
    max_masks = st.slider(
        "Max objects per image",
        min_value=100,
        max_value=2000,
        value=900,
        step=50,
        help="Upper cap to protect memory/time on dense scrap scenes.",
    )

    with st.expander("🔬 Advanced SAM Parameters", expanded=False):
        st.caption(
            "Override the defaults for the chosen mode. "
            "Lower thresholds → more masks (useful for gaskets / carburetors)."
        )
        adv_pred_iou = st.slider(
            "Pred IoU threshold",
            min_value=0.40, max_value=0.95, step=0.01,
            value=0.50 if recall_mode_label == "Industrial Parts" else
                  0.80 if recall_mode_label == "Max Recall" else
                  0.84 if recall_mode_label == "High Recall" else 0.88,
            help="Lower → SAM accepts masks whose own IoU prediction is uncertain. "
                 "Gaskets and thin parts need ≤ 0.60.",
        )
        adv_stability = st.slider(
            "Stability score threshold",
            min_value=0.50, max_value=0.98, step=0.01,
            value=0.60 if recall_mode_label == "Industrial Parts" else
                  0.90 if recall_mode_label == "Max Recall" else
                  0.93 if recall_mode_label == "High Recall" else 0.95,
            help="Lower → keeps less-stable masks (thin gasket edges often score low).",
        )
        adv_points = st.slider(
            "Points per side",
            min_value=16, max_value=128, step=4,
            value=128 if recall_mode_label == "Industrial Parts" else
                  40 if recall_mode_label == "Max Recall" else
                  32 if recall_mode_label == "High Recall" else 24,
            help="Dense grid catches small passages and fine gasket boundaries. "
                 "128 = maximum density for Industrial Parts.",
        )
        adv_dedup_contains = st.slider(
            "Containment dedup ratio",
            min_value=0.60, max_value=0.99, step=0.01,
            value=0.75 if recall_mode_label == "Industrial Parts" else 0.97,
            help="Lower → keeps masks that are nested inside larger ones "
                 "(e.g. gasket inside engine head).",
        )

    st.divider()
    st.header("📊 Stats")
    total = len(st.session_state.results) if st.session_state.results else 0
    saved = len(st.session_state.annotations)
    st.metric("Objects detected", total)
    st.metric("Annotations saved", saved)

    if st.session_state.annotations:
        st.divider()
        st.subheader("Saved labels")
        for ann in st.session_state.annotations:
            st.write(f"• `{ann['label']}`")

# ── main area ─────────────────────────────────────────────────────────────────

if not uploaded:
    st.info("Upload an image in the sidebar to begin.")
    st.stop()

# Reset when a new file is uploaded
if uploaded.name != st.session_state.last_file:
    st.session_state.results = None
    st.session_state.annotations = []
    st.session_state.selected = 0
    st.session_state.last_file = uploaded.name

try:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
except Exception as e:
    st.error(f"Image loading failed: {e}")
    st.stop()

h, w = image_np.shape[:2]

# ── segmentation trigger ──────────────────────────────────────────────────────

col_img, col_btn = st.columns([4, 1])
with col_img:
    st.image(image_np, caption="Uploaded image", width="stretch")
with col_btn:
    st.write("")
    st.write("")
    run = st.button("🚀 Segment All Objects", width="stretch", type="primary")

if run:
    mode_map = {
        "Balanced": "balanced",
        "High Recall": "high",
        "Max Recall": "max",
        "Industrial Parts": "industrial",
    }
    seg_config = {
        "recall_mode": mode_map[recall_mode_label],
        "min_area_px": int(min_area_px),
        "max_masks": int(max_masks),
        # Advanced overrides (always passed so the backend uses the slider values)
        "pred_iou_thresh": adv_pred_iou,
        "stability_score_thresh": adv_stability,
        "points_per_side": adv_points,
        "dedup_contains": adv_dedup_contains,
    }

    with st.spinner("SAM3 segmenting all objects…"):
        results = pipeline.run(image_np, config=seg_config)
    if not results:
        st.error("No objects detected. Try a different image.")
        st.stop()
    st.session_state.results = results
    st.session_state.selected = 0
    st.success(f"✅ Found **{len(results)}** objects")

results = st.session_state.results

if results is None:
    st.stop()

# ── full segmentation map ─────────────────────────────────────────────────────

st.divider()
st.subheader(f"🧠 All {len(results)} Objects — Full Segmentation Map")

seg_map = build_all_objects_overlay(image_np, results, h, w,
                                    selected_idx=st.session_state.selected)
st.image(seg_map, caption="Each color = one object  |  White border = selected",
         width="stretch")

# ── object gallery ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("📦 Object Gallery")

COLS = 6
colors = _distinct_colors(len(results))
gallery_cols = st.columns(COLS)

for i, obj in enumerate(results):
    mask = obj.get("segmentation")
    if mask is None:
        continue
    crop = crop_object(image_np, mask, h, w)
    if crop is None or crop.size == 0:
        continue

    with gallery_cols[i % COLS]:
        border_color = "#ffffff" if i == st.session_state.selected else "#333333"
        st.markdown(
            f'<div style="border:3px solid {border_color};border-radius:6px;padding:2px">',
            unsafe_allow_html=True
        )
        st.image(crop, caption=f"#{i}", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

# ── labeling panel ─────────────────────────────────────────────────────────────

st.divider()
sel = st.session_state.selected
st.subheader(f"🎯 Label Object #{sel}")

left, right = st.columns(2)

with left:
    # Show selected object highlighted on full image
    highlight = build_all_objects_overlay(image_np, results, h, w, selected_idx=sel)
    st.image(highlight, caption=f"Object #{sel} highlighted (white border)",
             width="stretch")

with right:
    sel_mask = results[sel].get("segmentation")
    if sel_mask is not None:
        crop = crop_object(image_np, sel_mask, h, w, pad=10)
        if crop is not None:
            st.image(crop, caption=f"Object #{sel} crop", width="stretch")

    pick_col, label_col = st.columns([1, 2])
    with pick_col:
        selected_from_dropdown = st.selectbox(
            "Object",
            options=list(range(len(results))),
            index=min(st.session_state.selected, len(results) - 1),
            format_func=lambda idx: f"#{idx}",
            key="selected_object_dropdown",
        )
    with label_col:
        label = st.text_input("Enter label", placeholder="e.g. nut, bolt, gear",
                              key="label_input")

    if selected_from_dropdown != st.session_state.selected:
        st.session_state.selected = selected_from_dropdown
        st.rerun()

    clean_label = label.strip()

    if st.button("💾 Save Annotation", type="primary",
                 disabled=not clean_label):
        mask = results[sel].get("segmentation")
        if mask is not None:
            if mask.shape != (h, w):
                mask = resize_mask(mask, (h, w))
            mask = mask.astype(bool)
            ys, xs = np.where(mask)

            if len(xs) > 0:
                st.session_state.annotations.append({
                    "object_id": sel,
                    "label": clean_label,
                    "bbox": [int(xs.min()), int(ys.min()),
                             int(xs.max() - xs.min()), int(ys.max() - ys.min())]
                })

                saved_name, split = save_training_image(
                    image_name=uploaded.name, image_np=image_np)
                save_yolo_segmentation(
                    image_name=saved_name, image_shape=(h, w),
                    mask=mask, label=clean_label, split=split)

                st.success(f"✅ Saved `{clean_label}` → {split} set")
                st.rerun()