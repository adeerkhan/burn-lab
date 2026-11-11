#!/usr/bin/env bash
# Run a burn-lab command inside a tmux session.
#
# Burn's TUI dashboard only renders when stdout is a terminal, so a plain
# `cargo run` from a script or an SSH session silently falls back to the plain
# CLI renderer. tmux gives the process a real pty, which turns the dashboard on.
#
# Usage:
#   ./scripts/dashboard.sh                                  # train with the TUI
#   ./scripts/dashboard.sh cargo run --release -- test       # any command
#
# Then, in another shell:
#   tmux capture-pane -p -t burnlab      # snapshot the dashboard as text
#   tmux attach -t burnlab               # attach interactively

set -euo pipefail

cd "$(dirname "$0")/.."

SESSION="${SESSION:-burnlab}"
WIDTH="${WIDTH:-140}"
HEIGHT="${HEIGHT:-42}"

if [ "$#" -eq 0 ]; then
    set -- cargo run --release -- train
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x "$WIDTH" -y "$HEIGHT"
# Keep the pane alive after the command exits, so the last dashboard stays visible.
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
tmux send-keys -t "$SESSION" "$*" Enter

echo "tmux session '$SESSION' started."
echo "  snapshot : tmux capture-pane -p -t $SESSION"
echo "  attach   : tmux attach -t $SESSION"
echo "  stop     : tmux kill-session -t $SESSION"
