#!/usr/bin/env bash
# Runs on every codespace start (postStartCommand), including resuming a stopped one.
# PyVista/VTK need an X display to render; DISPLAY=:99 is set in devcontainer.json.
set -u
if pgrep -x Xvfb >/dev/null; then
    exit 0
fi
# A stopped codespace can leave a stale lock behind, which makes Xvfb refuse to start.
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
setsid nohup Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp >/tmp/xvfb.log 2>&1 < /dev/null &
echo "start-xvfb: Xvfb started on :99"
