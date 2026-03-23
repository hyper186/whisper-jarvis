#!/usr/bin/env bash
# Jarvis voice assistant — start / stop / status
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PIDFILE="$DIR/jarvis.pid"
VENV_PY="$DIR/venv/bin/python"
export JARVIS_LOGFILE="${JARVIS_LOGFILE:-$DIR/jarvis.log}"

# Optional local secrets (file is gitignored); does not override vars already set in the shell.
ENV_FILE="$DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

cmd="${1:-}"

case "$cmd" in
  start)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Already running (PID $(cat "$PIDFILE"))."
      exit 0
    fi
    if [[ -z "${VENICE_API_KEY:-}" ]]; then
      echo "Set VENICE_API_KEY first, e.g. export VENICE_API_KEY='...'"
      exit 1
    fi
    [[ -x "$VENV_PY" ]] || { echo "Missing venv: $VENV_PY"; exit 1; }
    # Python already appends to JARVIS_LOGFILE; avoid duplicating lines in the same file
    nohup env JARVIS_LOGFILE="$JARVIS_LOGFILE" "$VENV_PY" -u "$DIR/jarvis.py" >/dev/null 2>&1 &
    echo $! > "$PIDFILE"
    echo "Started PID $(cat "$PIDFILE"). Log: $JARVIS_LOGFILE"
    ;;
  stop)
    if [[ ! -f "$PIDFILE" ]]; then
      echo "No PID file — not started with this script."
      exit 1
    fi
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "Sent SIGTERM to $pid."
    else
      echo "Process $pid not running."
    fi
    rm -f "$PIDFILE"
    ;;
  status)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Running PID $(cat "$PIDFILE")"
    else
      echo "Not running (or stale PID file)."
      [[ -f "$PIDFILE" ]] && rm -f "$PIDFILE"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|status}"
    echo ""
    echo "Before start: export VENICE_API_KEY='...'"
    echo "Foreground (Ctrl+C to stop):"
    echo "  cd $DIR && export VENICE_API_KEY='...' && ./venv/bin/python -u jarvis.py"
    exit 1
    ;;
esac
