"""
Meme Image Generator for the Fake Internet Ecosystem.

Generates actual images for memes using PIL/Pillow.
Each meme gets a unique, stylized image based on:
- Personality type (different color schemes, layouts)
- Content text (rendered as overlay)
- Generation/mutation (visual evolution indicators)
- Topic (color accent)

Image styles per personality:
- curious: cool blues, question marks
- toxic: red/dark, spiky borders
- viral_chaser: hot pink/yellow, explosion effects
- niche_thinker: deep purple, clean lines
- lurker: muted grey, subtle
- influencer: gold/white, glamorous
- meme_lord: neon green, chaotic
- echo_seeker: purple/pink, mirror effects
- doom_scroller: dark blue/black, ominous
- hot_taker: orange/red, fire-like
"""

import numpy as np
import os
import hashlib
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Image output directory
MEME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "memes")

# Ensure meme directory exists
os.makedirs(MEME_DIR, exist_ok=True)

# Color schemes per personality type
PERSONALITY_COLORS = {
    "curious": {"bg1": (29, 58, 92), "bg2": (99, 111, 250), "accent": (130, 170, 255), "text": (220, 230, 255)},
    "toxic": {"bg1": (58, 28, 28), "bg2": (180, 40, 40), "accent": (255, 80, 80), "text": (255, 200, 200)},
    "viral_chaser": {"bg1": (58, 28, 92), "bg2": (255, 20, 128), "accent": (255, 220, 50), "text": (255, 255, 220)},
    "niche_thinker": {"bg1": (45, 28, 58), "bg2": (120, 60, 200), "accent": (171, 99, 250), "text": (230, 210, 255)},
    "lurker": {"bg1": (40, 40, 45), "bg2": (80, 80, 90), "accent": (150, 150, 160), "text": (200, 200, 210)},
    "influencer": {"bg1": (58, 28, 45), "bg2": (200, 160, 50), "accent": (255, 215, 0), "text": (255, 245, 220)},
    "meme_lord": {"bg1": (28, 58, 28), "bg2": (50, 200, 50), "accent": (100, 255, 100), "text": (220, 255, 220)},
    "echo_seeker": {"bg1": (58, 28, 58), "bg2": (180, 50, 180), "accent": (255, 150, 255), "text": (255, 220, 255)},
    "doom_scroller": {"bg1": (15, 15, 40), "bg2": (30, 30, 80), "accent": (80, 80, 180), "text": (150, 150, 200)},
    "hot_taker": {"bg1": (58, 42, 28), "bg2": (220, 100, 20), "accent": (255, 160, 50), "text": (255, 230, 200)},
    "bot_spam": {"bg1": (40, 40, 28), "bg2": (100, 100, 20), "accent": (200, 200, 50), "text": (220, 220, 150)},
}

# Meme template backgrounds
MEME_BG_PATTERNS = ["gradient", "noise", "geometric", "radial"]


def _get_font(size: int = 20, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a font, trying system fonts first."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/english/Carlito-Bold.ttf" if bold else "/usr/share/fonts/truetype/english/Carlito-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _gradient_bg(width: int, height: int, color1: Tuple, color2: Tuple) -> Image.Image:
    """Create a gradient background."""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        r = int(color1[0] + (color2[0] - color1[0]) * y / height)
        g = int(color1[1] + (color2[1] - color1[1]) * y / height)
        b = int(color1[2] + (color2[2] - color1[2]) * y / height)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img


def _noise_bg(width: int, height: int, color1: Tuple, color2: Tuple) -> Image.Image:
    """Create a noisy background."""
    img = _gradient_bg(width, height, color1, color2)
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise, 'RGB')
    img = Image.blend(img, noise_img, 0.15)
    return img


def _geometric_bg(width: int, height: int, color1: Tuple, color2: Tuple, accent: Tuple) -> Image.Image:
    """Create a geometric pattern background."""
    img = _gradient_bg(width, height, color1, color2)
    draw = ImageDraw.Draw(img)
    
    # Draw random geometric shapes
    np.random.seed(hash((color1, color2)) % 2**31)
    for _ in range(8):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        size = np.random.randint(20, 80)
        shape_type = np.random.randint(0, 3)
        alpha_color = tuple(min(255, c + 30) for c in accent)
        
        if shape_type == 0:  # Rectangle
            draw.rectangle([x1, y1, x1 + size, y1 + size], outline=alpha_color, width=2)
        elif shape_type == 1:  # Circle
            draw.ellipse([x1, y1, x1 + size, y1 + size], outline=alpha_color, width=2)
        else:  # Line
            x2 = np.random.randint(0, width)
            y2 = np.random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=alpha_color, width=2)
    
    return img


def _radial_bg(width: int, height: int, color1: Tuple, color2: Tuple, accent: Tuple) -> Image.Image:
    """Create a radial pattern background."""
    img = Image.new('RGB', (width, height), color1)
    draw = ImageDraw.Draw(img)
    
    cx, cy = width // 2, height // 2
    max_r = int(np.sqrt(cx**2 + cy**2))
    
    for r in range(max_r, 0, -8):
        frac = r / max_r
        cr = int(color2[0] * (1 - frac) + color1[0] * frac)
        cg = int(color2[1] * (1 - frac) + color1[1] * frac)
        cb = int(color2[2] * (1 - frac) + color1[2] * frac)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(cr, cg, cb))
    
    # Accent ring
    ring_r = max_r // 3
    draw.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r], outline=accent, width=3)
    
    return img


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        try:
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]
        except Exception:
            line_width = len(test_line) * 10
        
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


def generate_meme_image(text: str, personality: str = "curious",
                        generation: int = 0, mutation_types: List[str] = None,
                        topic_idx: int = 0, post_id: str = "",
                        is_meme: bool = True) -> str:
    """
    Generate a meme image and save it.
    
    Returns the relative path to the generated image (from static/).
    """
    WIDTH, HEIGHT = 400, 300
    
    # Get color scheme
    colors = PERSONALITY_COLORS.get(personality, PERSONALITY_COLORS["curious"])
    color1 = colors["bg1"]
    color2 = colors["bg2"]
    accent = colors["accent"]
    text_color = colors["text"]
    
    # Add mutation visual effects
    if generation > 0:
        # Each generation shifts colors slightly
        shift = min(generation * 15, 60)
        color2 = tuple(min(255, c + shift) for c in color2)
        accent = tuple(min(255, c + shift // 2) for c in accent)
    
    # Choose background pattern
    pattern_idx = hash(post_id) % len(MEME_BG_PATTERNS) if post_id else np.random.randint(0, len(MEME_BG_PATTERNS))
    pattern = MEME_BG_PATTERNS[pattern_idx]
    
    if pattern == "gradient":
        img = _gradient_bg(WIDTH, HEIGHT, color1, color2)
    elif pattern == "noise":
        img = _noise_bg(WIDTH, HEIGHT, color1, color2)
    elif pattern == "geometric":
        img = _geometric_bg(WIDTH, HEIGHT, color1, color2, accent)
    else:  # radial
        img = _radial_bg(WIDTH, HEIGHT, color1, color2, accent)
    
    draw = ImageDraw.Draw(img)
    
    # Draw border based on personality
    border_color = accent
    border_width = 3
    if personality == "toxic":
        border_width = 4
    elif personality == "viral_chaser":
        border_width = 5
    elif personality == "influencer":
        border_width = 4
    draw.rectangle([0, 0, WIDTH - 1, HEIGHT - 1], outline=border_color, width=border_width)
    
    # Draw top label bar
    label_colors = {
        "curious": "THOUGHT",
        "toxic": "HOT TAKE",
        "viral_chaser": "TRENDING",
        "niche_thinker": "INSIGHT",
        "lurker": "OBSERVATION",
        "influencer": "ANNOUNCEMENT",
        "meme_lord": "MEME",
        "echo_seeker": "VINDICATED",
        "doom_scroller": "CONCERN",
        "hot_taker": "HOT TAKE",
        "bot_spam": "SPAM",
    }
    label = label_colors.get(personality, "POST")
    
    # Top bar background
    draw.rectangle([0, 0, WIDTH, 32], fill=(*accent, 200))
    label_font = _get_font(14, bold=True)
    draw.text((10, 8), label, fill=(0, 0, 0), font=label_font)
    
    # Generation indicator
    if generation > 0:
        gen_text = f"GEN {generation}"
        draw.text((WIDTH - 70, 8), gen_text, fill=(0, 0, 0), font=label_font)
    
    # Main text
    text_font = _get_font(18, bold=True)
    wrapped = _wrap_text(text, text_font, WIDTH - 40)
    
    y_offset = 50
    for line in wrapped[:8]:  # Max 8 lines
        # Shadow
        draw.text((22, y_offset + 2), line, fill=(0, 0, 0), font=text_font)
        draw.text((20, y_offset), line, fill=text_color, font=text_font)
        y_offset += 26
    
    # Mutation indicators at bottom
    if mutation_types:
        mut_font = _get_font(11)
        mut_colors = {
            "shorten": (100, 150, 255),
            "shift": (100, 255, 100),
            "slang": (255, 150, 255),
            "distort": (150, 200, 255),
            "merge": (255, 200, 100),
        }
        mut_x = 10
        for mt in mutation_types[:5]:
            mc = mut_colors.get(mt, accent)
            draw.rectangle([mut_x, HEIGHT - 24, mut_x + 55, HEIGHT - 6], fill=mc)
            draw.text((mut_x + 4, HEIGHT - 22), mt[:6].upper(), fill=(0, 0, 0), font=mut_font)
            mut_x += 60
    
    # Add subtle noise for "realistic" look
    if is_meme:
        noise_overlay = np.random.randint(0, 15, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise_overlay, 'RGB')
        img = Image.blend(img, noise_img, 0.05)
    
    # Slight blur for smoothness
    img = img.filter(ImageFilter.SMOOTH)
    
    # Save image
    safe_id = post_id.replace("/", "_").replace("\\", "_") if post_id else "unknown"
    filename = f"meme_{safe_id}.png"
    filepath = os.path.join(MEME_DIR, filename)
    img.save(filepath, "PNG")
    
    return f"memes/{filename}"


def generate_post_image(text: str, personality: str = "curious",
                        topic_idx: int = 0, post_id: str = "") -> str:
    """Generate a simpler image for regular (non-meme) posts."""
    WIDTH, HEIGHT = 400, 200
    
    colors = PERSONALITY_COLORS.get(personality, PERSONALITY_COLORS["curious"])
    color1 = colors["bg1"]
    color2 = colors["bg2"]
    text_color = colors["text"]
    accent = colors["accent"]
    
    # Simple gradient background
    img = _gradient_bg(WIDTH, HEIGHT, color1, color2)
    draw = ImageDraw.Draw(img)
    
    # Thin accent bar on left
    draw.rectangle([0, 0, 4, HEIGHT], fill=accent)
    
    # Text
    text_font = _get_font(15)
    wrapped = _wrap_text(text, text_font, WIDTH - 30)
    
    y_offset = 20
    for line in wrapped[:6]:
        draw.text((15, y_offset), line, fill=text_color, font=text_font)
        y_offset += 22
    
    # Save
    safe_id = post_id.replace("/", "_").replace("\\", "_") if post_id else "unknown"
    filename = f"post_{safe_id}.png"
    filepath = os.path.join(MEME_DIR, filename)
    img.save(filepath, "PNG")
    
    return f"memes/{filename}"


def cleanup_old_images(max_images: int = 1000):
    """Remove oldest images if we have too many."""
    if not os.path.exists(MEME_DIR):
        return
    
    files = []
    for f in os.listdir(MEME_DIR):
        if f.endswith('.png'):
            fp = os.path.join(MEME_DIR, f)
            files.append((fp, os.path.getmtime(fp)))
    
    if len(files) > max_images:
        files.sort(key=lambda x: x[1])  # Sort by modification time
        for fp, _ in files[:len(files) - max_images]:
            try:
                os.remove(fp)
            except OSError:
                pass
