#!/bin/bash

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

DESKTOP_FILE="$HOME/Desktop/qaeye.desktop"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Name=Q-AEye
Comment=Run Drowsiness AI Docker
Exec=$APP_DIR/run_mainui.sh
Icon=$APP_DIR/logo/icon.png
Terminal=false
Type=Application
Categories=Utility;
EOF


chmod +x "$DESKTOP_FILE"