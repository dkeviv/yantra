#!/bin/bash
# Create Yantra app icon using ImageMagick or Python PIL

# Check if we have magick/convert
if command -v magick &> /dev/null; then
    # Create a 1024x1024 icon with Yantra branding
    magick -size 1024x1024 xc:"#1e293b" \
        -gravity center \
        -fill "#3b82f6" \
        -font Arial-Bold -pointsize 200 \
        -annotate +0-100 "Y" \
        -fill white \
        -font Arial -pointsize 80 \
        -annotate +0+100 "ANTRA" \
        -fill "#3b82f6" \
        -draw "circle 512,512 512,300" \
        -fill none \
        -stroke "#3b82f6" -strokewidth 20 \
        -draw "path 'M 350,600 L 450,700 L 650,500'" \
        base.png
    
    echo "✅ Created base icon with ImageMagick"
elif command -v python3 &> /dev/null; then
    # Use Python PIL to create icon
    python3 << 'PYTHON'
from PIL import Image, ImageDraw, ImageFont
import os

# Create 1024x1024 image with dark background
img = Image.new('RGB', (1024, 1024), color='#1e293b')
draw = ImageDraw.Draw(img)

# Draw circle
draw.ellipse([256, 256, 768, 768], fill='#3b82f6', outline='#60a5fa', width=10)

# Draw "Y" for Yantra
try:
    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 300)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Draw Y
draw.text((512, 400), "Y", fill='white', font=font_large, anchor='mm')

# Draw checkmark/code symbol
draw.line([(350, 650), (450, 750), (650, 550)], fill='white', width=30)

img.save('base.png')
print("✅ Created base icon with Python PIL")
PYTHON
else
    echo "⚠️ Neither ImageMagick nor Python PIL available"
    echo "Creating simple placeholder icon..."
    # Create a simple colored square as fallback
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" | base64 -d > base.png
fi

# Generate all required sizes
for size in 32 128 256 512; do
    if command -v sips &> /dev/null; then
        sips -z $size $size base.png --out ${size}x${size}.png 2>/dev/null
        echo "Created ${size}x${size}.png"
    fi
done

# Create @2x versions
if command -v sips &> /dev/null; then
    sips -z 256 256 base.png --out 128x128@2x.png 2>/dev/null
    echo "Created 128x128@2x.png"
fi

# Copy base as icon.png
cp base.png icon.png

echo "✅ Icon generation complete!"
