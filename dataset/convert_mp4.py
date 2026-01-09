import pandas as pd
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

# --- Helper Functions ---

def offset_to_yx(content, offset):
    """Converts a 1D string offset to 2D (y, x) coordinates."""
    offset = min(len(content), int(offset))
    y = content.count('\n', 0, offset)
    last_newline_pos = content.rfind('\n', 0, offset)
    if last_newline_pos == -1:
        x = offset
    else:
        x = offset - last_newline_pos - 1
    return y, x

def apply_change(content, offset, length, new_text):
    """Applies a text change to the content string."""
    content = str(content)
    new_text = str(new_text) if pd.notna(new_text) else ""
    offset, length = int(offset), int(length)
    new_text = new_text.replace('\\n', '\n').replace('\\r', '\r')
    if offset > len(content):
        content += ' ' * (offset - len(content))
    return content[:offset] + new_text + content[offset + length:]

# --- Video Rendering Functions ---

def get_monospaced_font(size=14):
    """Attempts to load a nice monospaced font, falls back to default."""
    common_fonts = [
        "Consolas.ttf", "Courier New.ttf", "Lucon.ttf", 
        "LiberationMono-Regular.ttf", "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Monaco.ttf"
    ]
    
    for font_name in common_fonts:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
            
    print("Warning: Could not find a standard TTF font. Using PIL default (might look pixelated).")
    return ImageFont.load_default()

def get_font_metrics(font):
    """Calculate consistent character dimensions for a monospaced font."""
    char_w = int(font.getlength("A"))
    ascent, descent = font.getmetrics()
    char_h = ascent + descent
    bbox = font.getbbox("A")
    baseline_offset = bbox[1]
    return char_w, char_h, ascent, baseline_offset

def create_frame(width, height, content, cursor_pos, scroll_y, active_file, status_text, 
                 font, char_w, char_h, ascent, baseline_offset, pause_message=None):
    """Draws a single video frame using PIL."""
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    lines = content.split('\n')
    max_visible_lines = (height // char_h) - 2
    
    text_y_offset = -baseline_offset
    
    for i in range(max_visible_lines):
        line_idx = scroll_y + i
        if line_idx < len(lines):
            y_pos = i * char_h + text_y_offset
            draw.text((0, y_pos), lines[line_idx], font=font, fill=(200, 200, 200))
    
    cursor_y, cursor_x = cursor_pos
    display_y = cursor_y - scroll_y
    
    if 0 <= display_y < max_visible_lines:
        cursor_px_x = cursor_x * char_w
        cursor_px_y = display_y * char_h
        
        draw.rectangle(
            [cursor_px_x, cursor_px_y, cursor_px_x + char_w, cursor_px_y + char_h], 
            fill=(255, 255, 255)
        )
        
        if cursor_y < len(lines):
            line = lines[cursor_y]
            if cursor_x < len(line):
                char_under = line[cursor_x]
                draw.text((cursor_px_x, cursor_px_y + text_y_offset), char_under, font=font, fill=(0, 0, 0))

    bar_y = height - (2 * char_h)
    draw.rectangle([0, bar_y, width, bar_y + char_h], fill=(255, 255, 255))
    draw.text((0, bar_y + text_y_offset), status_text, font=font, fill=(0, 0, 0))
    
    if pause_message:
        pause_bar_y = bar_y - char_h
        draw.rectangle([0, pause_bar_y, width, pause_bar_y + char_h], fill=(255, 165, 0))
        draw.text((0, pause_bar_y + text_y_offset), pause_message, font=font, fill=(0, 0, 0))
    
    return np.array(img)

def render_video(filepath, output_file, speed_factor, width=1280, height=720, 
                 fps=30, long_pause_threshold=120000, save_labels=True, labels_only=False):
    """Main loop to process data and write MP4 with keystroke labels."""
    
    print(f"Processing {filepath}...")
    try:
        df = pd.read_csv(filepath).sort_values('Time').reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return

    font = get_monospaced_font(size=18)
    char_w, char_h, ascent, baseline_offset = get_font_metrics(font)
    
    if not labels_only:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    file_states = {}
    scroll_states = {}
    active_file = "Unknown"
    max_visible_lines = (height // char_h) - 2
    
    labels = []
    label_idx = 0
    current_video_frame = 0
    
    if save_labels:
        output_path = Path(output_file)
        labels_dir = output_path.parent / f"{output_path.stem}_labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("Processing events..." if labels_only else "Rendering frames... this may take a while.")
    
    for i in range(len(df)):
        event = df.iloc[i]
        next_event = df.iloc[i+1] if i + 1 < len(df) else None
        
        active_file = event['File']
        
        if active_file not in file_states:
            file_states[active_file] = ""
            scroll_states[active_file] = 0
        
        offset = int(event['RangeOffset'])
        length = int(event['RangeLength'])
        new_text = str(event['Text']) if pd.notna(event['Text']) else ""
        new_text_clean = new_text.replace('\\n', '\n').replace('\\r', '\r')
        
        video_frame_before = current_video_frame
        
        if active_file == "TERMINAL":
            file_states[active_file] += new_text_clean + '\n'
        else:
            file_states[active_file] = apply_change(
                file_states[active_file], event['RangeOffset'], 
                event['RangeLength'], event['Text']
            )
            
        content = file_states[active_file]
        cursor_y, cursor_x = offset_to_yx(content, event['RangeOffset'])
        scroll_y = scroll_states[active_file]
        
        if active_file == "TERMINAL":
            lines = content.split('\n')
            if len(lines) > max_visible_lines:
                scroll_y = max(0, len(lines) - max_visible_lines)
        else:
            if cursor_y < scroll_y:
                scroll_y = cursor_y
            elif cursor_y >= scroll_y + max_visible_lines:
                scroll_y = cursor_y - max_visible_lines + 1
        
        scroll_states[active_file] = scroll_y
        
        action = event["Type"]
        
        if next_event is not None:
            real_delta_ms = next_event['Time'] - event['Time']
            is_long_pause = real_delta_ms > long_pause_threshold
            
            if is_long_pause:
                pause_display_frames = fps * 3
                current_video_frame += pause_display_frames
                frames_to_write = fps
            else:
                video_delta_ms = real_delta_ms / speed_factor
                frames_to_write = max(1, int((video_delta_ms / 1000.0) * fps))
                if frames_to_write < 1 and video_delta_ms > 10: 
                    frames_to_write = 1
        else:
            frames_to_write = fps * 2
            is_long_pause = False

        if not labels_only:
            status_text = f"File: {active_file} | Time: {event['Time']/1000:.1f}s | Speed: {speed_factor}x"
            
            if next_event is not None and is_long_pause:
                pause_message = "Long pause detected. User might be googling, thinking or might have gone for a coffee..."
                frame_with_pause = create_frame(
                    width, height, content, (cursor_y, cursor_x), 
                    scroll_y, active_file, status_text, font, char_w, char_h, 
                    ascent, baseline_offset, pause_message=pause_message
                )
                for _ in range(fps * 3):
                    video_out.write(frame_with_pause)
            
            frame_image = create_frame(
                width, height, content, (cursor_y, cursor_x), 
                scroll_y, active_file, status_text, font, char_w, char_h, 
                ascent, baseline_offset
            )
            for _ in range(frames_to_write):
                video_out.write(frame_image)
        
        current_video_frame += frames_to_write
        
        if save_labels:
            labels.append({
                "label_idx": label_idx,
                "event_idx": i,
                "timestamp_ms": int(event['Time']),
                "file": active_file,
                "action_type": action,
                "text": new_text,
                "range_offset": offset,
                "range_length": length,
                "cursor": {"y": cursor_y, "x": cursor_x},
                "video_frame_before": video_frame_before,
                "video_frame_after": current_video_frame,
            })
            label_idx += 1
            
        if i % 100 == 0:
            print(f"Processed {i}/{len(df)} events...", end='\r')

    if not labels_only:
        video_out.release()
    
    if save_labels and labels:
        with open(labels_dir / "keystrokes.jsonl", 'w') as f:
            for label in labels:
                f.write(json.dumps(label) + '\n')
          
        print(f"\nSaved {len(labels)} labels to {labels_dir}")
        print(f"Total video frames: {current_video_frame}")