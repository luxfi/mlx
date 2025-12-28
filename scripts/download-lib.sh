#!/bin/bash
# Download pre-built MLX library for Go bindings
# Usage: ./scripts/download-lib.sh [version]

set -e

VERSION="${1:-latest}"
REPO="luxfi/mlx"
LIB_DIR="$(dirname "$0")/../lib"

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    darwin)
        case "$ARCH" in
            arm64) PLATFORM="macos-arm64" ;;
            x86_64) PLATFORM="macos-x64" ;;
            *) echo "Unsupported macOS architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    linux)
        case "$ARCH" in
            x86_64) PLATFORM="linux-x64" ;;
            aarch64) PLATFORM="linux-arm64" ;;
            *) echo "Unsupported Linux architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    mingw*|msys*|cygwin*)
        PLATFORM="windows-x64"
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

FILENAME="libmlx-${PLATFORM}.tar.gz"

echo "=== Downloading MLX library for $PLATFORM ==="
mkdir -p "$LIB_DIR"

if [ "$VERSION" = "latest" ]; then
    URL="https://github.com/${REPO}/releases/latest/download/${FILENAME}"
else
    URL="https://github.com/${REPO}/releases/download/${VERSION}/${FILENAME}"
fi

echo "Downloading from: $URL"
curl -fsSL "$URL" -o "/tmp/${FILENAME}" || {
    echo "Failed to download. Available releases:"
    echo "  https://github.com/${REPO}/releases"
    exit 1
}

echo "Extracting to: $LIB_DIR"
tar -xzf "/tmp/${FILENAME}" -C "$LIB_DIR"
rm "/tmp/${FILENAME}"

echo "=== MLX library installed successfully ==="
ls -la "$LIB_DIR"
