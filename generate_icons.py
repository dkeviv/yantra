#!/usr/bin/env python3
"""
Generate Yantra app icons from SVG template
Requires: pip install cairosvg pillow
"""

import os
from pathlib import Path

try:
    import cairosvg
    from PIL import Image
    import io
except ImportError:
    print("Installing required packages...")
    os.system("pip3 install cairosvg pillow")
    import cairosvg
    from PIL import Image
    import io

# Icon sizes needed
SIZES = [32, 128, 256, 512, 1024]

def generate_icons():
    script_dir = Path(__file__).parent
    icons_dir = script_dir / "src-tauri" / "icons"
    svg_path = icons_dir / "icon-template.svg"
    
    if not svg_path.exists():
        print(f"Error: SVG template not found at {svg_path}")
        return
    
    print(f"Generating icons from {svg_path}...")
    
    # Read SVG
    with open(svg_path, 'r') as f:
        svg_data = f.read()
    
    for size in SIZES:
        # Generate PNG from SVG
        png_data = cairosvg.svg2png(
            bytestring=svg_data.encode('utf-8'),
            output_width=size,
            output_height=size
        )
        
        # Open with Pillow for further processing
        img = Image.open(io.BytesIO(png_data))
        
        # Save main sizes
        if size == 512:
            output_path = icons_dir / "icon.png"
            img.save(output_path, "PNG")
            print(f"✓ Created {output_path}")
            
            output_path = icons_dir / f"{size}x{size}.png"
            img.save(output_path, "PNG")
            print(f"✓ Created {output_path}")
        elif size == 32 or size == 128 or size == 256:
            output_path = icons_dir / f"{size}x{size}.png"
            img.save(output_path, "PNG")
            print(f"✓ Created {output_path}")
            
            if size == 128:
                # Create @2x version
                output_path = icons_dir / f"{size}x{size}@2x.png"
                img.save(output_path, "PNG")
                print(f"✓ Created {output_path}")
    
    # Create base.png (same as icon.png)
    base_img = Image.open(icons_dir / "icon.png")
    base_img.save(icons_dir / "base.png", "PNG")
    print(f"✓ Created base.png")
    
    print("\n✅ All icons generated successfully!")
    print("\nNow generating .icns for macOS...")
    
    # Create iconset directory for macOS
    iconset_dir = icons_dir / "icon.iconset"
    iconset_dir.mkdir(exist_ok=True)
    
    # macOS icon sizes
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
        png_data = cairosvg.svg2png(
            bytestring=svg_data.encode('utf-8'),
            output_width=size,
            output_height=size
        )
        img = Image.open(io.BytesIO(png_data))
        output_path = iconset_dir / f"icon_{name}.png"
        img.save(output_path, "PNG")
        print(f"✓ Created {name}.png")
    
    # Convert iconset to icns using iconutil
    icns_path = icons_dir / "icon.icns"
    result = os.system(f"iconutil -c icns -o {icns_path} {iconset_dir}")
    
    if result == 0:
        print(f"\n✅ Created {icns_path}")
        # Clean up iconset directory
        os.system(f"rm -rf {iconset_dir}")
    else:
        print("\n⚠️  Warning: Could not create .icns file. iconutil may not be available.")
    
    print("\n🎉 Icon generation complete!")

if __name__ == "__main__":
    generate_icons()
