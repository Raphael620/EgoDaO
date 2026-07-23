#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$PROJECT_DIR/dist/main.dist"
ICON_SRC="$PROJECT_DIR/DaO/icon.png"

if [ ! -d "$DIST_DIR" ]; then
  echo "ERROR: $DIST_DIR not found. Run build.sh first."
  exit 1
fi

if [ ! -f "$DIST_DIR/main.bin" ]; then
  echo "ERROR: $DIST_DIR/main.bin not found. Nuitka build may have failed or output name differs."
  exit 1
fi

# ─── Read version from pyproject.toml ───
VERSION=$(grep -Po '(?<=^version = ")[^"]*' "$PROJECT_DIR/pyproject.toml")
PACKAGE="egodao"
PKG_ROOT="$PROJECT_DIR/dist/${PACKAGE}_${VERSION}_amd64"
DEB_FILE="${PACKAGE}_${VERSION}_amd64.deb"

echo "=== Packaging $PACKAGE v$VERSION ==="

# ─── Clean and create package root under dist/ ───
rm -rf "$PKG_ROOT"
mkdir -p "$PKG_ROOT/DEBIAN"
mkdir -p "$PKG_ROOT/usr/lib/$PACKAGE"
mkdir -p "$PKG_ROOT/usr/bin"
mkdir -p "$PKG_ROOT/usr/share/applications"
mkdir -p "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps"

# ─── Copy compiled files ───
echo "Copying compiled files..."
if ! cp -a "$DIST_DIR"/* "$PKG_ROOT/usr/lib/$PACKAGE/"; then
  echo "ERROR: failed to copy $DIST_DIR to $PKG_ROOT/usr/lib/$PACKAGE/"
  exit 1
fi
chmod 755 "$PKG_ROOT/usr/lib/$PACKAGE/main.bin"

# ─── Launcher script ───
cat > "$PKG_ROOT/usr/bin/$PACKAGE" << 'LAUNCHER'
#!/bin/bash
PROGDIR="/usr/lib/egodao"
exec "$PROGDIR/main.bin" "$@"
LAUNCHER
chmod 755 "$PKG_ROOT/usr/bin/$PACKAGE"

# ─── .desktop file ───
cat > "$PKG_ROOT/usr/share/applications/${PACKAGE}.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=Ego Daq-O
GenericName=Ego Data Acquisition
Comment=Ego-centric data acquisition and real-time processing
Icon=$PACKAGE
Exec=$PACKAGE
Terminal=false
Categories=Science;DataVisualization;
DESKTOP

# ─── Icon (resize 1024→256) ───
if [ -f "$ICON_SRC" ]; then
  if command -v convert &>/dev/null; then
    convert "$ICON_SRC" -resize 256x256 "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps/${PACKAGE}.png"
  elif command -v ffmpeg &>/dev/null; then
    ffmpeg -y -v quiet -i "$ICON_SRC" -vf scale=256:256 "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps/${PACKAGE}.png"
  else
    cp "$ICON_SRC" "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps/${PACKAGE}.png"
  fi
else
  echo "WARNING: icon not found at $ICON_SRC"
fi

# ─── DEBIAN control ───
cat > "$PKG_ROOT/DEBIAN/control" << CONTROL
Package: $PACKAGE
Version: $VERSION
Architecture: amd64
Maintainer: EgoDaO Team
Depends: libxcb-cursor0, libxcb-shape0, libxcb-xfixes0, libxcb-render-util0, libxcb-image0, libxcb-keysyms1, libxcb-randr0, libxcb-glx0, libxkbcommon-x11-0, libxcb-icccm4, libxcb-xkb1
Section: science
Priority: optional
Description: Ego Daq-O — Ego 数据采集与实时处理系统
  A data acquisition system for embodied AI ego-centric data collection.
  Supports OAK depth cameras with 3-camera + IMU + VIO pipeline,
  dual recording format (raw .mp4 + HumanEgo-compatible format),
  and headless mode with global hotkey control.
CONTROL

# ─── Post-install script ───
cat > "$PKG_ROOT/DEBIAN/postinst" << 'POSTINST'
#!/bin/bash
set -e

# Give main.bin CAP_SYS_RAWIO for keyboard to work without sudo
if [ -f /usr/lib/egodao/main.bin ]; then
  setcap cap_sys_rawio+ep /usr/lib/egodao/main.bin 2>/dev/null || true
fi

# Install OAK udev rules (linux device permissions)
UDEV_RULES="/etc/udev/rules.d/99-depthai.rules"
if [ ! -f "$UDEV_RULES" ]; then
  cat > "$UDEV_RULES" << 'UDEV'
SUBSYSTEM=="usb", ATTR{idVendor}=="03e7", MODE="0666"
UDEV
  udevadm control --reload-rules 2>/dev/null || true
  udevadm trigger 2>/dev/null || true
fi

exit 0
POSTINST
chmod 755 "$PKG_ROOT/DEBIAN/postinst"

# ─── Build .deb ───
echo "Building .deb..."
dpkg-deb --build "$PKG_ROOT" "$PROJECT_DIR/dist/$DEB_FILE"

echo
echo "=== Package created ==="
echo "  dist/$DEB_FILE"
echo
echo "Install: sudo dpkg -i dist/$DEB_FILE"
echo "Run:     egodao"
echo "Headless: egodao --no-gui"
echo
echo "NOTE: headless mode (--no-gui) hotkey requires running with sudo unless"
echo "      setcap succeeded (CAP_SYS_RAWIO on main.bin)."
