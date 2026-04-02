from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from mouse_actions import _accumulate_positions, _actions_from_target_text


def _draw_cursor(
    img: Image.Image,
    x: int,
    y: int,
    color: tuple[int, int, int],
    radius: int = 4,
    outline_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    draw = ImageDraw.Draw(img)
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bbox, fill=color, outline=outline_color, width=1)


def _draw_trail(
    img: Image.Image,
    positions: list[tuple[int, int]],
    up_to_idx: int,
    color: tuple[int, int, int, int],
    width: int = 2,
) -> None:
    """Draw a fading trail of recent positions onto the image."""
    draw = ImageDraw.Draw(img, "RGBA")
    trail_len = min(up_to_idx + 1, 8)
    start_idx = max(0, up_to_idx - trail_len + 1)
    for i in range(start_idx, up_to_idx):
        alpha = int(60 + 140 * ((i - start_idx) / max(trail_len - 1, 1)))
        c = (*color[:3], alpha)
        draw.line(
            [positions[i], positions[i + 1]],
            fill=c,
            width=width,
        )


def _draw_frame_label(
    img: Image.Image,
    frame_idx: int,
    gt_action: str,
    pred_action: str,
    upscale: int,
) -> None:
    """Draw frame index and action labels on the top of the frame."""
    draw = ImageDraw.Draw(img)
    font_size = max(10, 3 * upscale)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/custom/Inter.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    y_offset = 2
    draw.text((4, y_offset), f"F{frame_idx}", fill=(255, 255, 255), font=font)
    y_offset += font_size + 2

    gt_short = gt_action if len(gt_action) < 20 else gt_action[:18] + ".."
    pred_short = pred_action if len(pred_action) < 20 else pred_action[:18] + ".."
    draw.text((4, y_offset), f"GT:{gt_short}", fill=(0, 220, 0), font=font)
    y_offset += font_size + 1
    draw.text((4, y_offset), f"P:{pred_short}", fill=(220, 60, 60), font=font)


def _render_cursor_frames(
    jpeg_frames: list[bytes],
    gt_actions: list[str],
    pred_actions: list[str],
    img_w: int,
    img_h: int,
    max_frames: int = 16,
    upscale: int = 4,
) -> list[Image.Image]:
    """Render up to `max_frames` frames with GT (green) and pred (red) cursors,
    trails, and action labels."""
    n = min(len(jpeg_frames), len(gt_actions), max_frames)
    gt_pos = _accumulate_positions(gt_actions[:n], img_w, img_h)
    pred_pos = _accumulate_positions(pred_actions[:n], img_w, img_h)
    # Pad pred_pos to length n if the model produced fewer actions than n
    if len(pred_pos) < n:
        last = pred_pos[-1] if pred_pos else (img_w // 2, img_h // 2)
        pred_pos = pred_pos + [last] * (n - len(pred_pos))

    rendered: list[Image.Image] = []
    radius = max(2, 3 * upscale // 4)
    for i in range(n):
        frame_img = Image.open(io.BytesIO(jpeg_frames[i])).convert("RGBA")
        if upscale > 1:
            frame_img = frame_img.resize(
                (img_w * upscale, img_h * upscale), Image.NEAREST,
            )

        # Scale positions
        gt_scaled = [(p[0] * upscale, p[1] * upscale) for p in gt_pos[:i + 1]]
        pred_scaled = [(p[0] * upscale, p[1] * upscale) for p in pred_pos[:i + 1]]

        # Draw trails
        _draw_trail(frame_img, gt_scaled, i, color=(0, 220, 0, 180), width=max(1, upscale // 2))
        _draw_trail(frame_img, pred_scaled, i, color=(220, 60, 60, 180), width=max(1, upscale // 2))

        # Convert back to RGB for cursor drawing
        frame_rgb = frame_img.convert("RGB")

        gx, gy = gt_scaled[i]
        px, py = pred_scaled[i]

        _draw_cursor(frame_rgb, gx, gy, color=(0, 220, 0), radius=radius)   # GT green
        _draw_cursor(frame_rgb, px, py, color=(220, 60, 60), radius=radius)  # pred red

        # Draw labels
        gt_a = gt_actions[i] if i < len(gt_actions) else "N/A"
        pr_a = pred_actions[i] if i < len(pred_actions) else "N/A"
        _draw_frame_label(frame_rgb, i, gt_a, pr_a, upscale)

        rendered.append(frame_rgb)
    return rendered


def _make_grid(
    images: list[Image.Image],
    cols: int = 8,
) -> Image.Image:
    """Tile a list of PIL images into a single grid image."""
    if not images:
        return Image.new("RGB", (1, 1))
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    return grid


def _render_trajectory_overview(
    jpeg_frames: list[bytes],
    gt_actions: list[str],
    pred_actions: list[str],
    img_w: int,
    img_h: int,
    upscale: int = 4,
) -> Image.Image:
    """Render a single overview image showing the full GT and pred trajectories
    overlaid on the first frame."""
    n = min(len(jpeg_frames), len(gt_actions))
    if n == 0:
        return Image.new("RGB", (img_w * upscale, img_h * upscale))
    frame_img = Image.open(io.BytesIO(jpeg_frames[0])).convert("RGBA")
    if upscale > 1:
        frame_img = frame_img.resize(
            (img_w * upscale, img_h * upscale), Image.NEAREST,
        )

    gt_pos = _accumulate_positions(gt_actions[:n], img_w, img_h)
    pred_pos = _accumulate_positions(pred_actions[:n], img_w, img_h)
    gt_scaled = [(p[0] * upscale, p[1] * upscale) for p in gt_pos]
    pred_scaled = [(p[0] * upscale, p[1] * upscale) for p in pred_pos]

    draw = ImageDraw.Draw(frame_img, "RGBA")
    # Draw full trajectory lines
    if len(gt_scaled) > 1:
        for i in range(len(gt_scaled) - 1):
            alpha = int(80 + 175 * (i / max(len(gt_scaled) - 1, 1)))
            draw.line([gt_scaled[i], gt_scaled[i + 1]], fill=(0, 220, 0, alpha), width=max(2, upscale // 2))
    if len(pred_scaled) > 1:
        for i in range(len(pred_scaled) - 1):
            alpha = int(80 + 175 * (i / max(len(pred_scaled) - 1, 1)))
            draw.line([pred_scaled[i], pred_scaled[i + 1]], fill=(220, 60, 60, alpha), width=max(2, upscale // 2))

    frame_rgb = frame_img.convert("RGB")
    radius = max(3, upscale)
    # Mark start
    _draw_cursor(frame_rgb, gt_scaled[0][0], gt_scaled[0][1], color=(0, 255, 0), radius=radius + 2, outline_color=(255, 255, 0))
    # Mark end
    if len(gt_scaled) > 1:
        _draw_cursor(frame_rgb, gt_scaled[-1][0], gt_scaled[-1][1], color=(0, 180, 0), radius=radius)
    if len(pred_scaled) > 1:
        _draw_cursor(frame_rgb, pred_scaled[-1][0], pred_scaled[-1][1], color=(180, 0, 0), radius=radius)

    # Legend
    ldraw = ImageDraw.Draw(frame_rgb)
    font_size = max(12, 4 * upscale)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/custom/Inter.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    lx = 6
    ly = frame_rgb.height - font_size * 3 - 10
    ldraw.text((lx, ly), "Green = GT trajectory", fill=(0, 220, 0), font=font)
    ldraw.text((lx, ly + font_size + 2), "Red = Pred trajectory", fill=(220, 60, 60), font=font)
    ldraw.text((lx, ly + 2 * (font_size + 2)), f"{n} frames", fill=(200, 200, 200), font=font)

    return frame_rgb
