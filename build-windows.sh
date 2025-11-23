#!/bin/bash
# Build script for Windows NSIS installer
# Usage: ./build-windows.sh
# Note: Best run on Windows, but can cross-compile from macOS/Linux with proper setup

set -e  # Exit on error

echo "🪟 Building Yantra for Windows..."

# Check prerequisites
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: cargo not found. Install Rust from https://rustup.rs/"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm not found. Install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if Windows target is installed
if ! rustup target list | grep -q "x86_64-pc-windows-msvc (installed)"; then
    echo "📦 Installing Windows target..."
    rustup target add x86_64-pc-windows-msvc
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd "$(dirname "$0")"
npm install

# Build for Windows
echo "🔨 Building Tauri app for Windows..."

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Running on Windows
    npm run tauri:build -- --target x86_64-pc-windows-msvc
    INSTALLER_PATH="src-tauri/target/release/bundle/nsis/Yantra_0.1.0_x64-setup.exe"
else
    # Cross-compiling (requires additional setup)
    echo "⚠️  Warning: Cross-compiling from $OSTYPE"
    echo "For best results, build on Windows natively."
    echo "Attempting cross-compile..."
    npm run tauri:build -- --target x86_64-pc-windows-msvc || {
        echo "❌ Cross-compilation failed."
        echo "💡 To cross-compile from Unix, you need:"
        echo "   1. Install mingw-w64: brew install mingw-w64 (macOS)"
        echo "   2. Configure cargo for Windows target"
        echo "   3. Or use a Windows VM/CI for native builds"
        exit 1
    }
    INSTALLER_PATH="src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/Yantra_0.1.0_x64-setup.exe"
fi

# Check build success
if [ -f "$INSTALLER_PATH" ]; then
    echo "✅ Build successful!"
    echo "📦 Installer: src-tauri/$INSTALLER_PATH"
    
    # Show file size
    SIZE=$(du -h "$INSTALLER_PATH" | cut -f1)
    echo "📊 Package size: $SIZE"
    
    # Optional: Create checksums
    echo "🔐 Generating checksums..."
    cd "$(dirname "$INSTALLER_PATH")"
    shasum -a 256 "$(basename "$INSTALLER_PATH")" > "$(basename "$INSTALLER_PATH").sha256"
    echo "📝 Checksum saved"
else
    echo "❌ Build failed. Installer not found at: $INSTALLER_PATH"
    exit 1
fi

echo ""
echo "🎉 Windows build complete!"
echo "To test: Run installer on Windows machine"
