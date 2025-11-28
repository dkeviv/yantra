#!/usr/bin/env python3
"""
Generate Yantra app icons using PIL only (no Cairo dependency)
"""

import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Installing Pillow...")
    os.system("pip3 install pillow")
    from PIL import Image, ImageDraw, ImageFont

def create_icon(size):
    """Create a modern Yantra icon with Y letter"""
    # Create image with white to light blue gradient background
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw rounded rectangle background with gradient effect
    radius = int(size * 0.225)  # 22.5% of size for rounded corners
    
    # Create gradient background (white to light blue)
    for y in range(size):
        # Gradient from white (255, 255, 255) to light blue (240, 244, 255)
        r = 255
        g = int(255 - (y / size) * 11)
        b = int(255 - (y / size) * 0)
        draw.rectangle([(0, y), (size, y + 1)], fill=(r, g, b, 255))
    
    # Create rounded corners mask
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(0, 0), (size, size)], radius=radius, fill=255)
    
    # Apply mask to create rounded corners
    img.putalpha(mask)
    
    # Draw the Y letter with blue gradient
    # Calculate letter dimensions
    letter_width = int(size * 0.375)  # 37.5% of icon size
    letter_height = int(size * 0.488)  # 48.8% of icon size
    left_margin = int(size * 0.3125)  # 31.25% - center the Y
    top_margin = int(size * 0.273)    # 27.3% from top
    
    # Y letter consists of two parts:
    # 1. V-shaped top part
    # 2. Vertical stem at bottom
    
    # Create blue gradient color (from #3b82f6 to #1d4ed8)
    blue_start = (59, 130, 246)    # Light blue
    blue_end = (29, 78, 216)       # Dark blue
    
    # Draw V part of Y
    v_points = [
        (left_margin, top_margin),                                    # Top left
        (left_margin + letter_width // 2, top_margin + letter_height // 2),  # Middle point
        (left_margin + letter_width, top_margin),                     # Top right
        (left_margin + letter_width - int(size * 0.0625), top_margin),  # Inner top right
        (left_margin + letter_width // 2, top_margin + letter_height // 2 - int(size * 0.049)),  # Inner middle
        (left_margin + int(size * 0.0625), top_margin),               # Inner top left
    ]
    
    draw.polygon(v_points, fill=blue_start)
    
    # Draw vertical stem of Y
    stem_width = int(size * 0.0625)  # 6.25% of size
    stem_left = left_margin + letter_width // 2 - stem_width // 2
    stem_top = top_margin + letter_height // 2 - int(size * 0.0195)
    stem_bottom = top_margin + letter_height
    
    draw.rectangle(
        [(stem_left, stem_top), (stem_left + stem_width, stem_bottom)],
        fill=blue_end
    )
    
    # Add subtle shadow/glow effect
    # Create a slightly larger Y with transparency for glow
    shadow_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_img)
    
    # Draw shadow Y (slightly offset and transparent)
    offset = max(1, size // 128)
    shadow_v_points = [(x + offset, y + offset) for x, y in v_points]
    shadow_draw.polygon(shadow_v_points, fill=(29, 78, 216, 100))
    shadow_draw.rectangle(
        [(stem_left + offset, stem_top + offset), 
         (stem_left + stem_width + offset, stem_bottom + offset)],
        fill=(29, 78, 216, 100)
    )
    
    # Blend shadow with main image
    img = Image.alpha_composite(shadow_img, img)
    
    return img

def generate_all_icons():
    """Generate all required icon sizes"""
    script_dir = Path(__file__).parent
    icons_dir = script_dir / "src-tauri" / "icons"
    icons_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating Yantra app icons...")
    
    # Required sizes
    sizes = [32, 128, 256, 512, 1024]
    
    for size in sizes:
        print(f"  Creating {size}x{size} icon...")
        img = create_icon(size)
        
        # Save standard size
        if size == 512:
            img.save(icons_dir / "icon.png", "PNG")
            print(f"    ✓ icon.png")
        
        img.save(icons_dir / f"{size}x{size}.png", "PNG")
        print(f"    ✓ {size}x{size}.png")
        
        # Create @2x version for 128
        if size == 128:
            img.save(icons_dir / f"{size}x{size}@2x.png", "PNG")
            print(f"    ✓ {size}x{size}@2x.png")
    
    # Create base.png (copy of icon.png)
    base_img = create_icon(512)
    base_img.save(icons_dir / "base.png", "PNG")
    print(f"    ✓ base.png")
    
    # Generate macOS .icns file
    print("\n🍎 Generating macOS .icns file...")
    iconset_dir = icons_dir / "icon.iconset"
    iconset_dir.mkdir(exist_ok=True)
    
    macos_sizes = [
        (16, "16x16"),
        (32, "16x16@2x"),
        (32, "32x32"),
        (64, "32x32@2x"),
        (128, "128x128"),
        (256, "128x128@2x"),
        (256, "256x256"),
        (512, "256x256@2x"),
        (512, "512x512"),
        (1024, "512x512@2x"),
    ]
    
    for size, name in macos_sizes:
        img = create_icon(size)
        img.save(iconset_dir / f"icon_{name}.png", "PNG")
        print(f"    ✓ icon_{name}.png")
    
    # Convert to .icns using iconutil (macOS only)
    icns_path = icons_dir / "icon.icns"
    result = os.system(f"iconutil -c icns -o '{icns_path}' '{iconset_dir}' 2>/dev/null")
    
    if result == 0:
        print(f"\n    ✓ icon.icns created")
        # Clean up iconset directory
        import shutil
        shutil.rmtree(iconset_dir)
    else:
        print(f"\n    ⚠️  iconutil not available, keeping iconset directory")
    
    print("\n✅ All icons generated successfully!")
    print(f"📁 Icons saved to: {icons_dir}")

if __name__ == "__main__":
    generate_all_icons()
