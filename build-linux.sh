#!/bin/bash
# Build script for Linux AppImage and deb packages
# Usage: ./build-linux.sh

set -e  # Exit on error

echo "🐧 Building Yantra for Linux..."

# Check prerequisites
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: cargo not found. Install Rust from https://rustup.rs/"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm not found. Install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  Warning: This script should be run on Linux for best results"
    echo "Current OS: $OSTYPE"
fi

# Check for required Linux dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 Checking Linux dependencies..."
    MISSING_DEPS=""
    
    if ! dpkg -l | grep -q libwebkit2gtk-4.0-dev; then
        MISSING_DEPS="$MISSING_DEPS libwebkit2gtk-4.0-dev"
    fi
    if ! dpkg -l | grep -q build-essential; then
        MISSING_DEPS="$MISSING_DEPS build-essential"
    fi
    if ! dpkg -l | grep -q libssl-dev; then
        MISSING_DEPS="$MISSING_DEPS libssl-dev"
    fi
    if ! dpkg -l | grep -q libgtk-3-dev; then
        MISSING_DEPS="$MISSING_DEPS libgtk-3-dev"
    fi
    if ! dpkg -l | grep -q libayatana-appindicator3-dev; then
        MISSING_DEPS="$MISSING_DEPS libayatana-appindicator3-dev"
    fi
    if ! dpkg -l | grep -q librsvg2-dev; then
        MISSING_DEPS="$MISSING_DEPS librsvg2-dev"
    fi
    
    if [ -n "$MISSING_DEPS" ]; then
        echo "⚠️  Missing dependencies: $MISSING_DEPS"
        echo "Install with: sudo apt-get install $MISSING_DEPS"
        exit 1
    fi
    echo "✅ All dependencies installed"
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd "$(dirname "$0")"
npm install

# Build for Linux
echo "🔨 Building Tauri app for Linux..."
npm run tauri:build -- --target x86_64-unknown-linux-gnu

# Check build success
APPIMAGE_PATH="src-tauri/target/release/bundle/appimage/yantra_0.1.0_amd64.AppImage"
DEB_PATH="src-tauri/target/release/bundle/deb/yantra_0.1.0_amd64.deb"

BUILD_SUCCESS=true

if [ -f "$APPIMAGE_PATH" ]; then
    echo "✅ AppImage build successful!"
    echo "📦 AppImage: src-tauri/$APPIMAGE_PATH"
    SIZE=$(du -h "$APPIMAGE_PATH" | cut -f1)
    echo "📊 AppImage size: $SIZE"
    
    # Make AppImage executable
    chmod +x "$APPIMAGE_PATH"
    
    # Generate checksum
    echo "🔐 Generating AppImage checksum..."
    cd "$(dirname "$APPIMAGE_PATH")"
    shasum -a 256 "$(basename "$APPIMAGE_PATH")" > "$(basename "$APPIMAGE_PATH").sha256"
    cd - > /dev/null
else
    echo "❌ AppImage build failed"
    BUILD_SUCCESS=false
fi

if [ -f "$DEB_PATH" ]; then
    echo "✅ Debian package build successful!"
    echo "📦 Deb package: src-tauri/$DEB_PATH"
    SIZE=$(du -h "$DEB_PATH" | cut -f1)
    echo "📊 Deb size: $SIZE"
    
    # Generate checksum
    echo "🔐 Generating deb checksum..."
    cd "$(dirname "$DEB_PATH")"
    shasum -a 256 "$(basename "$DEB_PATH")" > "$(basename "$DEB_PATH").sha256"
    cd - > /dev/null
else
    echo "❌ Debian package build failed"
    BUILD_SUCCESS=false
fi

if [ "$BUILD_SUCCESS" = true ]; then
    echo ""
    echo "🎉 Linux build complete!"
    echo ""
    echo "To test:"
    echo "  AppImage: ./src-tauri/$APPIMAGE_PATH"
    echo "  Deb:      sudo dpkg -i src-tauri/$DEB_PATH"
else
    exit 1
fi
