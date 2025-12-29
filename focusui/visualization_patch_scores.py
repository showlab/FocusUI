"""Visualization utilities for patch scores."""

import numpy as np
from PIL import Image
import matplotlib.cm as cm


def colorize_heatmap(arr_2d, out_size=None, cmap_name='jet', vmin=None, vmax=None):
    """Map a 2D array to a color heatmap (RGB Image) and return normalized values."""
    arr = np.asarray(arr_2d, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - vmin) / (vmax - vmin + 1e-12)
        norm = np.clip(norm, 0.0, 1.0)

    rgba = (cm.get_cmap(cmap_name)(norm) * 255).astype(np.uint8)
    rgb = rgba[..., :3]
    img = Image.fromarray(rgb, mode='RGB')
    if out_size is not None:
        img = img.resize(out_size, resample=Image.Resampling.NEAREST)
    return img, norm


def load_and_resize_image(image_path, out_w, out_h):
    """Load image from path or accept a PIL.Image and resize to target size."""
    if isinstance(image_path, Image.Image):
        img = image_path
    else:
        img = Image.open(image_path)
    img = img.resize((out_w, out_h), resample=Image.Resampling.NEAREST)
    return img


def _heatmap_1d_to_img(arr_1d, h_grid_half, w_grid_half, out_w, out_h, cmap_name='jet'):
    arr_2d = np.asarray(arr_1d, dtype=np.float32).reshape(h_grid_half, w_grid_half)
    img, _ = colorize_heatmap(arr_2d, out_size=(out_w, out_h), cmap_name=cmap_name)
    return img


def vis_ps_heatmap(arr_1d, h_grid_half, w_grid_half, out_w, out_h, cmap_name='jet'):
    """Return a color heatmap image from a 1D score array."""
    return _heatmap_1d_to_img(arr_1d, h_grid_half, w_grid_half, out_w, out_h, cmap_name=cmap_name)


def vis_ps_overlay(screenshot_path, ps_1d, h_grid_half, w_grid_half, out_w, out_h, alpha=0.5, cmap_name='jet'):
    """Return overlay of screenshot (PIL or path) and predicted heatmap image."""
    heatmap_img = vis_ps_heatmap(ps_1d, h_grid_half, w_grid_half, out_w, out_h, cmap_name=cmap_name)
    screenshot_img = load_and_resize_image(screenshot_path, out_w, out_h)
    blended = Image.blend(screenshot_img.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=alpha)
    return blended
