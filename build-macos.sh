#!/bin/bash
# Build script for macOS DMG installer
# Usage: ./build-macos.sh

set -e  # Exit on error

echo "🍎 Building Yantra for macOS..."

# Check prerequisites
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: cargo not found. Install Rust from https://rustup.rs/"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm not found. Install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script should be run on macOS for best results"
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd "$(dirname "$0")"
npm install

# Build for universal binary (Intel + Apple Silicon)
echo "🔨 Building Tauri app (Universal binary)..."
npm run tauri:build -- --target universal-apple-darwin

# Check build success
if [ -f "src-tauri/target/release/bundle/dmg/Yantra_0.1.0_universal.dmg" ]; then
    echo "✅ Build successful!"
    echo "📦 DMG installer: src-tauri/target/release/bundle/dmg/Yantra_0.1.0_universal.dmg"
    
    # Show file size
    SIZE=$(du -h "src-tauri/target/release/bundle/dmg/Yantra_0.1.0_universal.dmg" | cut -f1)
    echo "📊 Package size: $SIZE"
    
    # Optional: Create checksums
    echo "🔐 Generating checksums..."
    cd src-tauri/target/release/bundle/dmg
    shasum -a 256 Yantra_0.1.0_universal.dmg > Yantra_0.1.0_universal.dmg.sha256
    echo "📝 Checksum saved: Yantra_0.1.0_universal.dmg.sha256"
else
    echo "❌ Build failed. DMG not found."
    exit 1
fi

echo ""
echo "🎉 macOS build complete!"
echo "To test: open src-tauri/target/release/bundle/dmg/Yantra_0.1.0_universal.dmg"
