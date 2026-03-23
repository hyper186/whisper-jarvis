"""
Voice assistant: openWakeWord (Hey Jarvis) → faster-whisper (local) → Venice AI chat + TTS.
Runs one interaction at a time; debounces wake and resets the wake model after each turn.

Logs: stderr + optional file. See jarvis.sh / env JARVIS_LOGFILE.

Secrets: set VENICE_API_KEY in the environment, or put it in a .env file next to jarvis.py
(gitignored). Do not commit API keys or .synthesis-credentials.

THIS IS HARD CODED TO WORK WITH HDMI FOR SOUND OUTPUT. CHANGE BACK TO SPEAKER IF NEEDED
"""
import getpass
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel
import openwakeword

# === CONFIG ===
def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, str(default)).strip())
        return max(1, v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw not in ("0", "false", "no", "off")


_stats_lock = threading.Lock()
_stats = {"chat_ok": 0, "chat_err": 0, "tts_ok": 0, "tts_err": 0, "turns": 0}
CHAT_URL = "https://api.venice.ai/api/v1/chat/completions"
TTS_URL = "https://api.venice.ai/api/v1/audio/speech"
IMAGE_URL = "https://api.venice.ai/api/v1/image/generate"

# Project directory (for bundled assets next to jarvis.py).
_JARVIS_DIR = Path(__file__).resolve().parent
_BUNDLED_LOADING_VIDEO = str(_JARVIS_DIR / "Venice Loading Video.mp4")
_BUNDLED_OPENSCREEN_PNG = str(_JARVIS_DIR / "wj_openscreen.png")
# Turn id 0 = idle openscreen only; first user turn uses 1+ and replaces it (loading → image).
IMAGE_OPENSCREEN_TURN_ID = 0


def _load_env_file() -> None:
    """Load KEY=value lines from .env beside this script. Does not override existing os.environ."""
    path = _JARVIS_DIR / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].lstrip()
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        if key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


_load_env_file()

VENICE_API_KEY = (os.environ.get("VENICE_API_KEY") or "").strip()

SAMPLE_RATE = 16000
# openWakeWord expects chunks of 1280 samples (80 ms) at 16 kHz
OWW_CHUNK = 1280
# webrtcvad at 16 kHz: valid frame sizes are 320, 480, 960
VAD_FRAME = 480

WHISPER_MODEL_SIZE = "base"
CHAT_MODEL = "minimax-m27" #zai-org-glm-5
SYSTEM_PROMPT = (
    "You are Jarvis, a private voice assistant powered by Grok. "
    "Answers are read aloud in one shot — keep them brief and complete: "
    "aim for one tight paragraph (about 40–80 spoken words), always end on a full sentence. "
    "Only use two short paragraphs if the user clearly asks for more detail. "
    "Plain language only: no markdown, no bullet lists, no numbered lists unless trivial. "
    "Never use footnotes, [[1]]-style citations, links, URLs, 'Sources:', or 'References:'. "
    "Be witty and truthful within that length."
)

# Only load the Jarvis wake word to cut CPU and false triggers from other built-ins
WAKE_MODEL_PATHS = [openwakeword.models["hey_jarvis"]["model_path"]]
WAKE_THRESHOLD = 0.65
# Require a few consecutive hot frames so noise does not instantly trigger
WAKE_PATIENCE = {"hey_jarvis_v0.1": 3}
WAKE_THRESHOLD_PATIENCE = {"hey_jarvis_v0.1": WAKE_THRESHOLD}

# After wake, this many seconds of captured audio are discarded (wake-word tail).
# Recommend users pause ~0.5–1.0 s after "Hey Jarvis" before the question —
# about POST_WAKE_SKIP_S + a small buffer (see POST_WAKE_HINT_S below).
POST_WAKE_SKIP_S = 0.45
POST_WAKE_HINT_S = 0.7
# End command after this much continuous silence (seconds)
SILENCE_END_S = 1.2
MAX_COMMAND_S = 12.0
# Ignore new wakes for this long after finishing speech (stops TTS/heavy breathing re-triggers)
COOLDOWN_S = 2.0

# Venice Kokoro TTS is typically faster on-device than Qwen 3 voices.
# Kokoro voice IDs look like: `af_sky`, `af_river`, etc.
TTS_MODEL = os.environ.get("JARVIS_TTS_MODEL", "tts-kokoro").strip()
TTS_VOICE = os.environ.get("JARVIS_TTS_VOICE", "af_sky").strip()
# Venice: mp3 by default. On Pi, try: export JARVIS_TTS_FORMAT=wav (often easier for aplay/pw-play).
TTS_RESPONSE_FORMAT = os.environ.get("JARVIS_TTS_FORMAT", "mp3").strip().lower()
# Completion budget (Venice/OpenAI-style APIs may read max_completion_tokens).
MAX_COMPLETION_TOKENS = _env_int("JARVIS_MAX_TOKENS", 2048)
# TTS playback speed; ~1.05–1.15 reads a bit faster without chipmunk effect.
TTS_SPEED = max(0.25, min(4.0, _env_float("JARVIS_TTS_SPEED", 1.09)))
# Long inputs need a long read timeout (seconds). Was 90; 2.4k+ chars can exceed that.
TTS_HTTP_TIMEOUT = _env_int("JARVIS_TTS_TIMEOUT", 240)
# Cap characters sent to TTS (full reply still in logs). 0 = no cap.
try:
    _tmc = int(os.environ.get("JARVIS_TTS_MAX_CHARS", "2800").strip())
    TTS_MAX_INPUT_CHARS = max(0, _tmc)
except ValueError:
    TTS_MAX_INPUT_CHARS = 2800

# If true, pass `streaming: true` to Venice TTS so audio can begin playing
# before the whole response is generated.
#
# Note: on this Pi setup, FIFO streaming with ffplay appears unreliable for
# mp3 (decoder errors / silence). Default to False so audio plays reliably.
TTS_STREAMING = bool(_env_int("JARVIS_TTS_STREAMING", 0))
TTS_STREAM_CHUNK_BYTES = _env_int("JARVIS_TTS_STREAM_CHUNK_BYTES", 8192)
# If streaming plays almost instantly (often means decoder couldn't build a full frame),
# retry once using non-streaming TTS for that same chunk so the user still hears audio.
TTS_STREAMING_FALLBACK = bool(_env_int("JARVIS_TTS_STREAMING_FALLBACK", 1))
TTS_STREAMING_FALLBACK_IF_DT_S = max(0.0, float(os.environ.get("JARVIS_TTS_STREAMING_FALLBACK_DT_S", "2.0").strip()))

# If Venice TTS truncates long inputs, chunking avoids cutting mid-reply.
# Set 0 to disable. Default picks a conservative size so even ~300–400 char
# replies get split across a couple TTS requests instead of one.
TTS_CHUNK_CHARS = _env_int("JARVIS_TTS_CHUNK_CHARS", 300)
# When multiple TTS chunks are queued, start downloading the next chunk while the
# current one plays (cuts dead air between chunks). Set 0 to disable.
TTS_PREFETCH_NEXT = bool(_env_int("JARVIS_TTS_PREFETCH", 1))

# === MONITOR IMAGE (nano-banana-2) ===
IMAGE_ENABLED = bool(_env_int("JARVIS_IMAGE_ENABLED", 1))
IMAGE_MODEL = os.environ.get("JARVIS_IMAGE_MODEL", "nano-banana-2").strip()
IMAGE_FORMAT = os.environ.get("JARVIS_IMAGE_FORMAT", "webp").strip().lower()
IMAGE_RESOLUTION = os.environ.get("JARVIS_IMAGE_RESOLUTION", "2K").strip()
# Venice image API: e.g. "16:9", "1:1", "9:16" (see aspect_ratio on image/generate).
IMAGE_ASPECT_RATIO = os.environ.get("JARVIS_IMAGE_ASPECT_RATIO", "16:9").strip()
IMAGE_RETURN_BINARY = bool(_env_int("JARVIS_IMAGE_RETURN_BINARY", 0))
IMAGE_HTTP_TIMEOUT = _env_int("JARVIS_IMAGE_TIMEOUT", 240)
IMAGE_DISPLAY_WIDTH = max(1, _env_int("JARVIS_IMAGE_WIDTH", 1920))
IMAGE_DISPLAY_HEIGHT = max(1, _env_int("JARVIS_IMAGE_HEIGHT", 1080))
# Many Venice image models cap dimensions at 1280 and require a divisor (often 16).
IMAGE_API_MAX_DIM = max(1, _env_int("JARVIS_IMAGE_API_MAX_DIM", 1280))
IMAGE_API_DIM_DIVISOR = max(1, _env_int("JARVIS_IMAGE_API_DIVISOR", 16))
IMAGE_TMP_DIR = os.environ.get("JARVIS_IMAGE_TMP_DIR", "/tmp/jarvis_images").strip()
IMAGE_PROMPT_AGENT_MAX_TOKENS = _env_int("JARVIS_IMAGE_PROMPT_MAX_TOKENS", 220)
# Second LLM call to refine (question,reply) → image prompt. Disable for a much faster path (uses reply text only).
IMAGE_PROMPT_AGENT_ENABLED = _env_bool("JARVIS_IMAGE_PROMPT_AGENT", True)
# If 1 (default), build the image prompt from the main chat reply and call /image/generate immediately (no extra LLM
# wait). If 0 and JARVIS_IMAGE_PROMPT_AGENT=1, wait for the image-prompt LLM before /image/generate (slower, often
# better prompts).
IMAGE_FAST_IMAGE_PROMPT = _env_bool("JARVIS_IMAGE_FAST_IMAGE_PROMPT", True)
# Web search on that prompt call adds a lot of latency; reply already reflects chat. Enable only if you need it.
IMAGE_PROMPT_WEB_SEARCH = _env_bool("JARVIS_IMAGE_PROMPT_WEB_SEARCH", False)
IMAGE_PROMPT_MODEL = os.environ.get("JARVIS_IMAGE_PROMPT_MODEL", "").strip()
IMAGE_FBDEV = os.environ.get("JARVIS_IMAGE_FBDEV", "/dev/fb0").strip()
IMAGE_VT = _env_int("JARVIS_IMAGE_VT", 1)
IMAGE_CHVT = bool(_env_int("JARVIS_IMAGE_CHVT", 1))
# How to paint images on the HDMI monitor:
# - auto: prefers desktop mpv when X11/Wayland session detected (see JARVIS_IMAGE_TRY_DRM_FIRST); else DRM-first.
# - mpv-drm / fbi / feh: force one backend.
IMAGE_VIEWER = os.environ.get("JARVIS_IMAGE_VIEWER", "auto").strip().lower()
IMAGE_DRM_DEVICE = os.environ.get("JARVIS_IMAGE_DRM_DEVICE", "").strip()
IMAGE_DRM_CONNECTOR = os.environ.get("JARVIS_IMAGE_DRM_CONNECTOR", "").strip()
# If 1, mpv logs to stderr (also captured in jarvis.log if logging configured); use for debugging HDMI.
IMAGE_MPV_VERBOSE = bool(_env_int("JARVIS_IMAGE_MPV_VERBOSE", 0))
# -1 = auto: if a GUI session is detected (X11 socket / Wayland), try desktop mpv before DRM. 1 = DRM first always. 0 = desktop before DRM always.
IMAGE_TRY_DRM_FIRST = _env_int("JARVIS_IMAGE_TRY_DRM_FIRST", -1)

IMAGE_PLACEHOLDER_ENABLED = bool(_env_int("JARVIS_IMAGE_PLACEHOLDER_ENABLED", 1))
# While the model renders: looped video (mpv) or PNG. Default = bundled "Venice Loading Video.mp4" if present, else ffmpeg-built /tmp clip.
_DEFAULT_LOADING_MEDIA = (
    _BUNDLED_LOADING_VIDEO if os.path.isfile(_BUNDLED_LOADING_VIDEO) else "/tmp/jarvis_monitor_loading.mp4"
)
IMAGE_PLACEHOLDER_PATH = os.environ.get("JARVIS_IMAGE_PLACEHOLDER_PATH", _DEFAULT_LOADING_MEDIA).strip()
# Length of the auto-generated loading clip (seconds); mpv loops it until the real image is ready.
IMAGE_LOADING_CLIP_S = max(2, min(30, _env_int("JARVIS_IMAGE_LOADING_CLIP_SECONDS", 4)))
# X11: set root to solid black once so brief gaps show black instead of the desktop wallpaper (0 to disable).
JARVIS_X11_SOLID_ROOT = bool(_env_int("JARVIS_X11_SOLID_ROOT", 1))
# Extra fullscreen mpv (no --ontop) under Jarvis so IPC/overlay gaps show black, not the desktop wallpaper.
JARVIS_MONITOR_BLACK_UNDERLAY = _env_bool("JARVIS_MONITOR_BLACK_UNDERLAY", True)


def _loading_media_is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".mp4", ".webm", ".mkv", ".mov", ".m4v")


def _monitor_handoff_loadfile_risky(old_path: str | None, new_path: str) -> bool:
    """True when swapping still image ↔ video via IPC loadfile tends to expose the desktop for a frame."""
    if not old_path:
        return False
    return _loading_media_is_video(old_path) != _loading_media_is_video(new_path)


def _monitor_path_is_still_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (
        ".png",
        ".jpg",
        ".jpeg",
        ".jpe",
        ".webp",
        ".gif",
    )


def _mpv_overlay_spawn_settle_for_path(media_path: str) -> float:
    """Short settle is fine for video; large PNG/JPEG decode needs longer before we trust mpv stayed up."""
    base = JARVIS_MPV_OVERLAY_SPAWN_SETTLE_S
    if _monitor_path_is_still_image(media_path):
        return max(base, 0.55)
    return base


def _dejavu_sans_path() -> str | None:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        if os.path.isfile(p):
            return p
    return None


def _ensure_loading_media() -> str:
    """Ensure loading placeholder exists: looping MP4/WebM (default) or single PNG if path ends in .png."""
    path = IMAGE_PLACEHOLDER_PATH
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    try:
        if os.path.exists(path) and os.path.getsize(path) > 2000:
            return path
    except OSError:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot create loading placeholder")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return _ensure_loading_png(path, None, None, ffmpeg)

    if _loading_media_is_video(path):
        # 16:9 clip, dark background + subtle vertical motion on the caption (mpv loops with --loop-file=yes).
        w = min(1280, IMAGE_API_MAX_DIM)
        w = (w // IMAGE_API_DIM_DIVISOR) * IMAGE_API_DIM_DIVISOR
        h = max(IMAGE_API_DIM_DIVISOR, int((w * 9 / 16 // IMAGE_API_DIM_DIVISOR) * IMAGE_API_DIM_DIVISOR))
        font = _dejavu_sans_path()
        dur = IMAGE_LOADING_CLIP_S
        if font:
            ff_esc = font.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            vf = (
                f"drawtext=fontfile={ff_esc}:text='Generating...':fontsize=44:fontcolor=white@0.92:"
                f"x=(w-text_w)/2:y=(h-text_h)/2+18*sin(2*PI*t/{dur}):"
                f"box=1:boxcolor=black@0.5:boxborderw=14"
            )
        else:
            vf = (
                f"drawtext=text='Generating...':fontsize=40:fontcolor=white@0.9:"
                f"x=(w-text_w)/2:y=(h-text_h)/2+16*sin(2*PI*t/{dur})"
            )
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=#1a2030:s={w}x{h}:d={dur}:r=30",
            "-vf",
            vf,
            "-t",
            str(dur),
        ]
        if ext == ".webm":
            cmd.extend(["-c:v", "libvpx-vp9", "-crf", "37", "-b:v", "0", "-an", "-pix_fmt", "yuv420p", path])
        else:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-tune",
                    "stillimage",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    path,
                ]
            )
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        except subprocess.CalledProcessError as e:
            logging.warning(
                "Loading video encode failed (%s), falling back to PNG: %s",
                e,
                (e.stderr or "")[:500],
            )
            png_fallback = os.path.splitext(path)[0] + ".png"
            return _ensure_loading_png(png_fallback, w, h, ffmpeg)
        return path

    return _ensure_loading_png(path, None, None, ffmpeg)


def _ensure_loading_png(path: str, w: int | None, h: int | None, ffmpeg: str) -> str:
    if w is None or h is None:
        w = int(IMAGE_API_MAX_DIM // IMAGE_API_DIM_DIVISOR * IMAGE_API_DIM_DIVISOR)
        h = max(IMAGE_API_DIM_DIVISOR, int((w * 9 / 16 // IMAGE_API_DIM_DIVISOR) * IMAGE_API_DIM_DIVISOR))
    font = _dejavu_sans_path()
    if font:
        ff_esc = font.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filt = (
            f"drawtext=fontfile={ff_esc}:text='Generating...':fontsize=40:"
            "fontcolor=white@0.9:x=(w-text_w)/2:y=(h-text_h)/2:"
            "box=1:boxcolor=black@0.45:boxborderw=12"
        )
    else:
        filt = (
            "drawtext=text='Generating...':fontsize=36:fontcolor=white@0.9:"
            "x=(w-text_w)/2:y=(h-text_h)/2"
        )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=#1a2030:s=%dx%d" % (w, h),
            "-vf",
            filt,
            "-frames:v",
            "1",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return path


def _drm_connected_connectors() -> list[tuple[str, str]]:
    """Return [(drm_device, connector), ...] for KMS connectors with status=connected."""
    out: list[tuple[str, str]] = []
    base = "/sys/class/drm"
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        if "-" not in name or not name.startswith("card"):
            continue
        card_id, conn_name = name.split("-", 1)
        if not card_id.startswith("card") or not card_id[4:].isdigit():
            continue
        if "WRITEBACK" in conn_name.upper():
            continue
        status_path = os.path.join(base, name, "status")
        try:
            with open(status_path, encoding="utf-8") as f:
                if f.read().strip() != "connected":
                    continue
        except OSError:
            continue
        dev = f"/dev/dri/{card_id}"
        if os.path.exists(dev):
            out.append((dev, conn_name))
    return out


def _mpv_drm_attempt_specs() -> list[tuple[str, str]]:
    """Ordered (drm_device, connector); connector may be '' for device-default."""
    if IMAGE_DRM_DEVICE:
        return [(IMAGE_DRM_DEVICE, IMAGE_DRM_CONNECTOR)]

    attempts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    discovered = _drm_connected_connectors()

    def sort_key(item: tuple[str, str]) -> tuple:
        dev, conn = item
        u = conn.upper()
        pri = 0 if "HDMI" in u else (1 if "DP" in u else 2)
        m = re.search(r"card(\d+)$", dev)
        card_idx = int(m.group(1)) if m else 999
        # Prefer lower card index first (Pi: the cable you use is often card0; card1 was wrongly tried first before).
        return (pri, card_idx, conn, dev)

    for item in sorted(discovered, key=sort_key):
        if item not in seen:
            attempts.append(item)
            seen.add(item)

    for cand in ("/dev/dri/card0", "/dev/dri/card1"):
        if os.path.exists(cand) and (cand, "") not in seen:
            attempts.append((cand, ""))
            seen.add((cand, ""))

    return attempts if attempts else [("", "")]


def _desktop_session_likely() -> bool:
    """True if a local graphical session probably owns the display (DRM master unavailable for vo=drm)."""
    if os.path.exists("/tmp/.X11-unix/X0"):
        return True
    rt = (os.environ.get("XDG_RUNTIME_DIR") or "").strip()
    if not rt:
        uid_path = f"/run/user/{os.getuid()}"
        if os.path.isdir(uid_path):
            rt = uid_path
    if rt:
        for wn in ("wayland-0", "wayland-1"):
            if os.path.exists(os.path.join(rt, wn)):
                return True
    return False


def _auto_try_drm_before_desktop() -> bool:
    if IMAGE_TRY_DRM_FIRST == 1:
        return True
    if IMAGE_TRY_DRM_FIRST == 0:
        return False
    return not _desktop_session_likely()


IMAGE_PROMPT_SYSTEM_PROMPT = (
    "You are an image prompt designer. Convert the user's question and the assistant's "
    "spoken reply into a single image concept that can be rendered by an image model. "
    "The output will be shown on a wide 16:9 monitor; favor a horizontal composition (landscape framing). "
    "Pick an appropriate visualization style (infographic, comparison chart, skyline illustration, "
    "recipe cards, diagram, etc.) based on the content. Prefer clear shapes and icons. "
    "Avoid small or unreadable text; do not include paragraphs of text. If numbers are needed, "
    "encode them as bar lengths, podium sizes, or relative scales instead of labels. "
    "Return ONLY the final image prompt text with no JSON and no extra commentary."
)

_image_display_state_lock = threading.Lock()
_image_latest_turn_id = 0
_image_display_proc: subprocess.Popen | None = None
_image_display_path: str | None = None
# Unix socket for the running mpv --input-ipc-server=… (seamless loadfile: loading clip → final image).
_image_mpv_ipc_socket: str | None = None
# True if the visible fullscreen window is X11/Wayland (mpv/feh), not KMS vo=drm.
_image_display_via_desktop: bool = False
_jarvis_x11_root_painted: bool = False
_monitor_black_underlay_proc: subprocess.Popen | None = None


def _remove_transient_monitor_file(path: str | None) -> None:
    """Delete only generated turn_* files under IMAGE_TMP_DIR — never the bundled loading video."""
    if not path:
        return
    ap = os.path.abspath(path)
    if ap == os.path.abspath(IMAGE_PLACEHOLDER_PATH):
        return
    if ap == os.path.abspath(_BUNDLED_LOADING_VIDEO):
        return
    tmp = os.path.abspath(IMAGE_TMP_DIR)
    if not ap.startswith(tmp + os.sep):
        return
    bn = os.path.basename(ap)
    if not (
        re.match(r"^turn_\d+\.(webp|png|jpe?g|jpeg)$", bn, re.I)
        or re.match(r"^turn_\d+_monitor\.png$", bn, re.I)
    ):
        return
    try:
        os.remove(ap)
    except OSError:
        pass


def _monitor_raster_copy_for_playback(src_path: str) -> str:
    """Some desktop mpv builds mishandle WebP; feh/mpv are reliable with a flat PNG."""
    ext = os.path.splitext(src_path)[1].lower()
    if ext not in (".webp",):
        return src_path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src_path
    os.makedirs(IMAGE_TMP_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out = os.path.join(IMAGE_TMP_DIR, f"{stem}_monitor.png")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                src_path,
                "-frames:v",
                "1",
                out,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if os.path.isfile(out) and os.path.getsize(out) > 200:
            return out
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as e:
        logging.debug("Monitor WebP→PNG raster copy skipped: %s", e)
    return src_path


def _ensure_black_underlay_png() -> str:
    """Solid black 1920×1080 PNG for the underlay mpv (ffmpeg) or tiny fallback."""
    path = os.path.join(tempfile.gettempdir(), "jarvis_monitor_underlay_black.png")
    if os.path.isfile(path) and os.path.getsize(path) > 80:
        return path
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=black:s=1920x1080",
                    "-frames:v",
                    "1",
                    path,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            if os.path.isfile(path) and os.path.getsize(path) > 80:
                return path
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as e:
            logging.warning("Monitor underlay: ffmpeg black frame failed: %s", e)
    # Minimal 1×1 black PNG
    try:
        with open(path, "wb") as f:
            f.write(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                )
            )
    except OSError as e:
        logging.warning("Monitor underlay: cannot write %s: %s", path, e)
    return path


def _stop_monitor_underlay() -> None:
    global _monitor_black_underlay_proc
    p = _monitor_black_underlay_proc
    _monitor_black_underlay_proc = None
    _terminate_display_proc(p)


def _start_monitor_black_underlay_mpv() -> None:
    """Fullscreen black mpv without --ontop: stays under Jarvis mpv; hides wallpaper during transitions."""
    global _monitor_black_underlay_proc
    if not IMAGE_ENABLED or not JARVIS_MONITOR_BLACK_UNDERLAY:
        return
    if not _desktop_session_likely() or not shutil.which("mpv"):
        return
    if _monitor_black_underlay_proc is not None and _monitor_black_underlay_proc.poll() is None:
        return
    png = _ensure_black_underlay_png()
    if not os.path.isfile(png):
        return
    mpv = shutil.which("mpv")
    if not mpv:
        return
    abspath = os.path.abspath(png)
    last_err = b""
    for env in _mpv_desktop_env_candidates():
        has_disp = bool((env.get("DISPLAY") or "").strip())
        vos_try = ("x11", "gpu") if has_disp else ("gpu", "x11")
        for vo in vos_try:
            if vo == "x11" and not has_disp:
                continue
            cmd = [
                mpv,
                "--no-terminal",
                "--no-audio",
                "--mute=yes",
                "--fullscreen",
                "--keep-open=always",
                "--force-window=immediate",
                "--loop-file=yes",
                "--image-display-duration=inf",
                "--background=color",
                "--background-color=#FF000000",
                f"--vo={vo}",
                "--title=jarvis-black-underlay",
                abspath,
            ]
            if not IMAGE_MPV_VERBOSE:
                cmd.insert(2, "--msg-level=all:error")
            else:
                cmd[2:2] = ["-v", "--msg-level=all=warn"]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
            )
            ok, err_b = _wait_display_proc_alive(
                proc, label=f"mpv black underlay vo={vo}", settle_s=0.22
            )
            if ok:
                _monitor_black_underlay_proc = proc
                logging.info(
                    "Monitor: black underlay mpv pid=%s (under Jarvis; set JARVIS_MONITOR_BLACK_UNDERLAY=0 to disable)",
                    proc.pid,
                )
                return
            _terminate_display_proc(proc)
            last_err = err_b
    logging.warning(
        "Monitor: black underlay mpv could not start (desktop may flash between clips). err=%r",
        (last_err or b"")[:800],
    )


def _stop_image_display() -> None:
    """Stop the monitor image viewer process if it is running."""
    global _image_display_proc, _image_display_path, _image_mpv_ipc_socket, _image_display_via_desktop
    with _image_display_state_lock:
        proc = _image_display_proc
        old_path = _image_display_path
        ipc_sock = _image_mpv_ipc_socket
        _image_display_proc = None
        _image_display_path = None
        _image_mpv_ipc_socket = None
        _image_display_via_desktop = False
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if ipc_sock:
        try:
            os.unlink(ipc_sock)
        except OSError:
            pass
    _remove_transient_monitor_file(old_path)


def _new_mpv_ipc_socket_path() -> str:
    """Fresh path for a new mpv instance (avoids bind conflicts while an old mpv still holds its socket)."""
    return os.path.join(tempfile.gettempdir(), f"jarvis_mpv_{os.getuid()}_{uuid.uuid4().hex}.sock")


def _unlink_mpv_ipc_quiet(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _mpv_ipc_request(socket_path: str, command: list) -> bool:
    """Send one mpv JSON IPC command; return True if error field is success."""
    req = {"command": command}
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(socket_path)
        sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf and len(buf) < 65536:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf += chunk
        sock.close()
        line = buf.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
        if not line:
            return False
        resp = json.loads(line)
        return resp.get("error") == "success"
    except Exception as e:
        logging.debug("mpv IPC %s failed: %s", command[:1], e)
        return False


def _mpv_ipc_loadfile(socket_path: str, path: str) -> bool:
    """Swap the current fullscreen mpv media without killing the process (no compositor gap)."""
    return _mpv_ipc_request(socket_path, ["loadfile", os.path.abspath(path), "replace"])


def _mpv_ipc_get_property(socket_path: str, name: str):
    """Read one mpv property via JSON IPC; None if unavailable or error."""
    req = {"command": ["get_property", name]}
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect(socket_path)
        sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf and len(buf) < 65536:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf += chunk
        sock.close()
        line = buf.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
        if not line:
            return None
        resp = json.loads(line)
        if resp.get("error") != "success":
            return None
        return resp.get("data")
    except Exception as e:
        logging.debug("mpv IPC get_property %s failed: %s", name, e)
        return None


def _mpv_ipc_still_to_video_seamless(socket_path: str, video_abs: str) -> bool:
    """Append the loading video, then playlist-next so demux runs while the still stays visible.

    loadfile … replace clears the VO and often flashes black; append keeps the image until the next entry plays.
    """
    video_abs = os.path.abspath(video_abs)
    if not _mpv_ipc_request(socket_path, ["loadfile", video_abs, "append"]):
        return False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        n = _mpv_ipc_get_property(socket_path, "playlist-count")
        if isinstance(n, int) and n >= 2:
            break
        time.sleep(0.025)
    else:
        logging.debug("Monitor: still→video append: playlist-count never reached 2")
        return False
    if _mpv_ipc_request(socket_path, ["playlist-next"]) or _mpv_ipc_request(
        socket_path, ["playlist-next", "force"]
    ):
        return True
    logging.debug("Monitor: playlist-next failed after append; using loadfile replace")
    return _mpv_ipc_loadfile(socket_path, video_abs)


def _mpv_ipc_wait_loading_video_painted(socket_path: str | None, *, timeout_s: float) -> bool:
    """True once mpv has started outputting the loading video (avoids topmost black-window flash)."""
    if not socket_path or not os.path.exists(socket_path):
        return False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        efn = _mpv_ipc_get_property(socket_path, "estimated-frame-number")
        if isinstance(efn, (int, float)) and int(efn) >= 1:
            return True
        vp = _mpv_ipc_get_property(socket_path, "video-params")
        tp = _mpv_ipc_get_property(socket_path, "time-pos")
        if isinstance(vp, dict) and vp.get("w") and isinstance(tp, (int, float)):
            return True
        time.sleep(0.035)
    return False


def _mpv_ipc_socket_path_runtime() -> str:
    """Prefer XDG_RUNTIME_DIR for the IPC socket (often more reliable than /tmp for desktop mpv)."""
    rt = (os.environ.get("XDG_RUNTIME_DIR") or "").strip()
    if rt and os.path.isdir(rt):
        return os.path.join(rt, f"jarvis_mpv_{os.getuid()}_{uuid.uuid4().hex}.sock")
    return _new_mpv_ipc_socket_path()


def _wait_mpv_ipc_socket_ready(
    socket_path: str, proc: subprocess.Popen, *, timeout_s: float = 4.0
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if os.path.exists(socket_path):
            try:
                test = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                test.settimeout(0.15)
                test.connect(socket_path)
                test.close()
                return True
            except OSError:
                pass
        time.sleep(0.04)
    return False


# Extra settle after IPC loadfile so the compositor paints before we consider the handoff done.
JARVIS_MPV_IPC_LOADFILE_SETTLE_S = max(0.0, min(1.5, _env_float("JARVIS_MPV_IPC_LOADFILE_SETTLE_S", 0.28)))
# Added after loadfile when swapping loading MP4 → final still in the same mpv (helps first frame paint).
JARVIS_MPV_IPC_VIDEO_TO_STILL_EXTRA_SETTLE_S = max(
    0.0, min(1.2, _env_float("JARVIS_MPV_IPC_VIDEO_TO_STILL_EXTRA_SETTLE_S", 0.5))
)
# After loadfile when swapping openscreen/final PNG → loading MP4 in the same mpv (first video frame).
JARVIS_MPV_IPC_STILL_TO_VIDEO_EXTRA_SETTLE_S = max(
    0.0, min(1.0, _env_float("JARVIS_MPV_IPC_STILL_TO_VIDEO_EXTRA_SETTLE_S", 0.48))
)
# After spawning on-top replacement mpv, wait before closing the old window so the new surface is actually painted.
JARVIS_MPV_OVERLAY_HANDOFF_SETTLE_S = max(
    0.0, min(2.0, _env_float("JARVIS_MPV_OVERLAY_HANDOFF_SETTLE_S", 0.1))
)
# Video → still: keep loading mpv alive until overlay mpv has decoded the PNG/JPEG (removes ~1s desktop flash).
JARVIS_MPV_PRE_KILL_STILL_SETTLE_S = max(
    0.0, min(1.5, _env_float("JARVIS_MPV_PRE_KILL_STILL_SETTLE_S", 0.68))
)
# Still → loading MP4 (overlay fallback): keep openscreen/final PNG until loading mpv is playing (same idea as pre-kill still).
JARVIS_MPV_PRE_KILL_LOADING_VIDEO_SETTLE_S = max(
    0.0, min(1.5, _env_float("JARVIS_MPV_PRE_KILL_LOADING_VIDEO_SETTLE_S", 0.52))
)
# How long to wait after Popen for direct-file desktop mpv before trusting it stayed up (overlay path; keep small).
JARVIS_MPV_OVERLAY_SPAWN_SETTLE_S = max(
    0.06, min(0.8, _env_float("JARVIS_MPV_OVERLAY_SPAWN_SETTLE_S", 0.18))
)
# idle+IPC path: after JSON loadfile, confirm process still alive (was 0.9s — felt like a multi-second “gap”).
JARVIS_MPV_IDLE_POST_LOAD_SETTLE_S = max(
    0.12, min(1.2, _env_float("JARVIS_MPV_IDLE_POST_LOAD_SETTLE_S", 0.35))
)
# Defer-ontop loading mpv: poll IPC until first video frame before raising window above the still image.
JARVIS_MPV_LOADING_FIRST_FRAME_WAIT_S = max(
    0.2, min(3.5, _env_float("JARVIS_MPV_LOADING_FIRST_FRAME_WAIT_S", 2.0))
)
# Same-mpv still→loading: append + playlist-next instead of replace (avoids VO black clear). Set 0 to use replace only.
JARVIS_MPV_STILL_TO_VIDEO_APPEND = _env_bool("JARVIS_MPV_STILL_TO_VIDEO_APPEND", True)
# Desktop monitor mpv hwdec: default no — Pi often blinks black when HW tears down between still and H.264.
# Use JARVIS_MPV_MONITOR_HWDEC=auto or default to restore mpv's normal hwdec choice.
JARVIS_MPV_MONITOR_HWDEC = (os.environ.get("JARVIS_MPV_MONITOR_HWDEC") or "no").strip()


def _mpv_monitor_hwdec_cmd_args() -> list[str]:
    v = JARVIS_MPV_MONITOR_HWDEC.lower()
    if v in ("default", "auto", "mpv"):
        return []
    if v in ("no", "off", "false", "0"):
        return ["--hwdec=no"]
    return [f"--hwdec={JARVIS_MPV_MONITOR_HWDEC}"]


def _spawn_mpv_desktop_idle_ipc_then_load(
    media_path: str, *, post_load_settle_s: float | None = None
) -> tuple[subprocess.Popen, str] | None:
    """Start fullscreen desktop mpv in --idle=yes mode with IPC, then load media via loadfile.

    Same process + socket is reused for loadfile(handoff) → final image (no desktop flash).
    """
    settle_after_load = (
        JARVIS_MPV_IDLE_POST_LOAD_SETTLE_S if post_load_settle_s is None else post_load_settle_s
    )
    mpv = shutil.which("mpv")
    if not mpv or not _desktop_session_likely():
        return None
    abspath = os.path.abspath(media_path)
    envs = _mpv_desktop_env_candidates()
    if not any(
        (e.get("DISPLAY") or "").strip() or (e.get("WAYLAND_DISPLAY") or "").strip() for e in envs
    ) and not os.path.exists("/tmp/.X11-unix/X0"):
        return None

    last_err = b""
    for env in envs:
        has_disp = bool((env.get("DISPLAY") or "").strip())
        vos_try = ("x11", "gpu") if has_disp else ("gpu", "x11")
        for vo in vos_try:
            if vo == "x11" and not (env.get("DISPLAY") or "").strip():
                continue
            ipc_sock = _mpv_ipc_socket_path_runtime()
            _unlink_mpv_ipc_quiet(ipc_sock)
            cmd = [
                mpv,
                "--no-terminal",
                "--no-audio",
                "--mute=yes",
                *_mpv_monitor_hwdec_cmd_args(),
                "--fullscreen",
                "--ontop",
                "--background=color",
                "--background-color=#FF000000",
                "--keep-open=always",
                "--force-window=immediate",
                "--idle=yes",
                f"--vo={vo}",
                "--loop-file=yes",
                "--image-display-duration=inf",
                "--input-ipc-server",
                ipc_sock,
            ]
            if not IMAGE_MPV_VERBOSE:
                cmd.insert(2, "--msg-level=all:error")
            else:
                cmd[2:2] = ["-v", "--msg-level=all=warn"]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
            )
            label = f"mpv idle+ipc {vo} DISPLAY={env.get('DISPLAY', '')!r}"
            if not _wait_mpv_ipc_socket_ready(ipc_sock, proc, timeout_s=2.0):
                try:
                    proc.terminate()
                    proc.wait(timeout=1.5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                _unlink_mpv_ipc_quiet(ipc_sock)
                last_err = b"ipc socket timeout"
                continue
            if not _mpv_ipc_loadfile(ipc_sock, abspath):
                try:
                    proc.terminate()
                    proc.wait(timeout=1.5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                _unlink_mpv_ipc_quiet(ipc_sock)
                last_err = b"loadfile failed"
                continue
            ok, err_b = _wait_display_proc_alive(proc, label=label, settle_s=settle_after_load)
            if ok:
                logging.info(
                    "Monitor: desktop mpv IPC session vo=%s pid=%s (loading → final via loadfile, no gap)",
                    vo,
                    proc.pid,
                )
                return proc, ipc_sock
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            _unlink_mpv_ipc_quiet(ipc_sock)
            last_err = err_b
    if last_err:
        logging.debug("mpv desktop idle+ipc session not available; last err=%r", (last_err or b"")[:500])
    return None


def _sanitize_single_prompt(prompt: str) -> str:
    """Best-effort cleanup so the image model doesn't get prompt garbage."""
    p = prompt.strip()
    # Remove markdown-ish leading hashes and any following whitespace.
    p = re.sub(r"^#+\s*", "", p)
    p = p.replace("\r", " ").replace("\n", " ").strip()
    # Avoid wrapping quotes that some models return.
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return p


def _image_prompt_without_llm(question: str, reply: str) -> str:
    """Build an image prompt from the spoken reply only (no extra chat round-trip)."""
    snippet = (reply or "").strip().replace("\r", " ")
    if len(snippet) > 1600:
        snippet = snippet[:1600] + "…"
    out = _sanitize_single_prompt(
        f"{snippet} — wide 16:9 infographic or editorial illustration, bold readable shapes, minimal text."
    )
    if not out:
        out = _sanitize_single_prompt(
            f"{(question or '').strip()[:800]} — wide 16:9 illustration, clear visual metaphor, minimal text."
        )
    return out or "abstract wide 16:9 gradient landscape, calm modern illustration"


def _get_image_prompt(question: str, reply: str) -> str:
    """Ask the LLM to convert (question, reply) into an image model prompt."""
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    model = IMAGE_PROMPT_MODEL or CHAT_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": IMAGE_PROMPT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User question:\n{question}\n\n"
                    f"Assistant spoken reply:\n{reply}\n\n"
                    "Create the best possible image prompt for the reply."
                ),
            },
        ],
        "temperature": 0.4,
        "max_tokens": IMAGE_PROMPT_AGENT_MAX_TOKENS,
        "stream": False,
        "venice_parameters": {
            "enable_web_search": "on" if IMAGE_PROMPT_WEB_SEARCH else "off",
            "enable_x_search": False,
            "enable_web_scraping": False,
            "enable_web_citations": False,
            "include_venice_system_prompt": False,
        },
        "parallel_tool_calls": False,
        "tools": [],
    }
    try:
        resp = requests.post(CHAT_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        body = resp.json()
        raw = body["choices"][0]["message"]["content"]
        out = _sanitize_single_prompt(strip_citations_and_links(raw))
        if not out:
            out = _sanitize_single_prompt(f"{reply} (visual summary infographic, no readable text)")
        return out
    except Exception as e:
        logging.warning("Image prompt agent failed: %s", e)
        # Fallback: make a generic image prompt from the reply itself.
        out = _sanitize_single_prompt(f"{reply} (visual infographic, no readable text)")
        if not out:
            out = _sanitize_single_prompt(f"{question} (visual summary infographic, no readable text)")
        return out


def _venice_api_dimensions() -> tuple[int, int]:
    """Width/height for POST /image/generate, within caps and JARVIS_IMAGE_ASPECT_RATIO."""
    divisor = IMAGE_API_DIM_DIVISOR
    max_dim = IMAGE_API_MAX_DIM
    ar = (IMAGE_ASPECT_RATIO or "").lower().replace(" ", "").replace("x", ":")

    if ar == "16:9":
        api_w = (max_dim // divisor) * divisor
        api_h = max(divisor, int((api_w * 9 / 16 // divisor) * divisor))
        return api_w, api_h
    if ar == "9:16":
        api_h = (max_dim // divisor) * divisor
        api_w = max(divisor, int((api_h * 9 / 16 // divisor) * divisor))
        return api_w, api_h
    if ar == "1:1":
        s = (max_dim // divisor) * divisor
        return s, s

    w = float(IMAGE_DISPLAY_WIDTH)
    h = float(IMAGE_DISPLAY_HEIGHT)
    md = float(max_dim)
    if max(w, h) > md:
        scale = md / max(w, h)
        w *= scale
        h *= scale
    dv = float(divisor)
    return max(1, int((w // dv) * dv)), max(1, int((h // dv) * dv))


def _generate_monitor_image(prompt: str, out_path: str) -> str:
    """Generate an image via Venice image API and write it to out_path."""
    if not prompt or not prompt.strip():
        raise ValueError("image prompt is empty/whitespace")
    logging.info("Monitor image prompt chars=%d", len(prompt.strip()))

    fmt = IMAGE_FORMAT
    if fmt == "jpg":
        fmt = "jpeg"

    if fmt not in ("jpeg", "png", "webp"):
        raise ValueError(f"Unsupported IMAGE_FORMAT={IMAGE_FORMAT!r}; expected jpeg/png/webp.")

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    if IMAGE_RETURN_BINARY:
        headers["Accept"] = f"image/{fmt}"

    resolution = (IMAGE_RESOLUTION or "").strip().upper()
    api_width, api_height = _venice_api_dimensions()

    req = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "format": fmt,
        "width": api_width,
        "height": api_height,
        "return_binary": IMAGE_RETURN_BINARY,
        "safe_mode": True,
    }
    if IMAGE_ASPECT_RATIO:
        req["aspect_ratio"] = IMAGE_ASPECT_RATIO
    logging.info(
        "Venice image generate: %dx%d aspect_ratio=%r model=%s",
        api_width,
        api_height,
        IMAGE_ASPECT_RATIO or "(unset)",
        IMAGE_MODEL,
    )
    # Some Venice image models are picky about resolution values.
    if resolution in ("1K", "2K", "4K"):
        req["resolution"] = resolution
    resp = requests.post(
        IMAGE_URL,
        json=req,
        headers=headers,
        timeout=(30, IMAGE_HTTP_TIMEOUT),
    )
    try:
        resp.raise_for_status()
    except Exception:
        # Include the response body snippet: Venice returns JSON with useful 400 details.
        body_snip = (resp.text or "").strip()
        logging.warning(
            "Venice image API error status=%s body_snip=%r",
            resp.status_code,
            body_snip[:800],
        )
        raise

    content_type = (resp.headers.get("content-type") or "").lower()
    image_bytes: bytes
    if "application/json" in content_type:
        body = resp.json()
        images = body.get("images") or []
        if not images:
            raise ValueError("Venice image API returned no images in JSON.")
        timing = body.get("timing")
        if isinstance(timing, dict) and timing:
            logging.info("Venice image API timing (server ms): %s", timing)
        # When return_binary=false, images are base64-encoded strings.
        image_bytes = base64.b64decode(images[0])
    else:
        image_bytes = resp.content

    if len(image_bytes) < 100:
        raise ValueError(f"Venice image API returned too few bytes: {len(image_bytes)}")

    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return out_path


def _wait_display_proc_alive(
    proc: subprocess.Popen, *, label: str, settle_s: float = 0.45
) -> tuple[bool, bytes]:
    """Return True only if the viewer is still running after settle_s.

    mpv vo=drm often stays alive for a few ms then exits with Permission denied; a single
    poll() would falsely count that as success.
    """
    err_b = b""
    deadline = time.monotonic() + settle_s
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            try:
                _, err_b = proc.communicate(timeout=1.0)
            except Exception:
                err_b = b""
            logging.warning(
                "%s exited during settle (%.2fs) rc=%s err=%r",
                label,
                settle_s,
                rc,
                (err_b or b"")[:4000],
            )
            return False, err_b
        time.sleep(0.05)
    rc = proc.poll()
    if rc is not None:
        try:
            _, err_b = proc.communicate(timeout=1.0)
        except Exception:
            err_b = b""
        logging.warning("%s exited at settle rc=%s err=%r", label, rc, (err_b or b"")[:4000])
        return False, err_b
    return True, err_b


def _log_image_file_magic(path: str, context: str) -> None:
    """Log size and format signature so we can confirm bytes on disk are a real image."""
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError as e:
        logging.warning("%s: cannot read %s: %s", context, path, e)
        return
    kind = "?"
    if head.startswith(b"\xff\xd8\xff"):
        kind = "jpeg"
    elif head.startswith(b"\x89PNG\r\n\x1a\n"):
        kind = "png"
    elif len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        kind = "webp"
    elif head.startswith((b"GIF87a", b"GIF89a")):
        kind = "gif"
    elif len(head) >= 8 and head[4:8] == b"ftyp":
        kind = "mp4"
    elif len(head) >= 4 and head.startswith(b"\x1a\x45\xdf\xa3"):
        kind = "webm/mkv"
    logging.info("%s bytes=%d format=%s magic16=%s path=%s", context, sz, kind, head[:16].hex(), path)


def _spawn_mpv_drm(image_path: str) -> subprocess.Popen | None:
    """Show image fullscreen via KMS/DRM (works when Jarvis is started from SSH).

    No --input-ipc-server: on Pi 5, vo=drm + IPC can leave mpv running with a blank screen on the active cable.
    """
    mpv = shutil.which("mpv")
    if not mpv:
        return None
    abspath = os.path.abspath(image_path)
    base_cmd = [
        mpv,
        "--no-terminal",
        "--no-audio",
        "--mute=yes",
        "--fullscreen",
        "--keep-open=always",
        "--force-window=immediate",
        "--vo=drm",
        "--loop-file=yes",
        "--image-display-duration=inf",
    ]
    if not IMAGE_MPV_VERBOSE:
        # Leave errors visible on stderr; --really-quiet hides the reason mpv exits (rc=1) during settle.
        base_cmd.insert(2, "--msg-level=all:error")
    else:
        base_cmd.extend(["-v", "--msg-level=all=warn"])

    last_err = b""
    for dev, conn in _mpv_drm_attempt_specs():
        cmd = list(base_cmd)
        if dev:
            cmd.append(f"--drm-device={dev}")
        if conn:
            cmd.append(f"--drm-connector={conn}")
        cmd.append(abspath)
        label = f"mpv drm ({dev or 'default'} {conn or 'default-connector'})"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )
        ok, err_b = _wait_display_proc_alive(proc, label=label)
        if ok:
            logging.info(
                "Monitor image shown via mpv drm pid=%s device=%s connector=%s image=%s",
                proc.pid,
                dev or "default",
                conn or "(unset)",
                abspath,
            )
            return proc
        last_err = err_b
    if last_err:
        logging.warning("mpv drm failed on all tried DRM devices; last err=%r", last_err[:2000])
    return None


def _mpv_desktop_env_candidates() -> list[dict[str, str]]:
    """DISPLAY / Wayland environments to try when DRM is busy (desktop compositor)."""
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    base = os.environ.copy()

    def add(e: dict[str, str]) -> None:
        key = (
            (e.get("DISPLAY") or "").strip(),
            (e.get("WAYLAND_DISPLAY") or "").strip(),
            (e.get("XAUTHORITY") or "").strip(),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(e)

    # Order matters: bare SSH env + vo=gpu probes DRM and fails while compositor runs.
    # Prefer explicit user env, then :0 / Wayland, and only then inherited env.
    user_has_desktop = bool((base.get("DISPLAY") or "").strip()) or bool(
        (base.get("WAYLAND_DISPLAY") or "").strip()
    )
    if user_has_desktop:
        add(dict(base))

    if os.path.exists("/tmp/.X11-unix/X0"):
        e = dict(base)
        e["DISPLAY"] = ":0"
        add(e)
        xauth = os.path.expanduser("~/.Xauthority")
        if os.path.isfile(xauth):
            e2 = dict(e)
            e2["XAUTHORITY"] = xauth
            add(e2)
    rt = (base.get("XDG_RUNTIME_DIR") or "").strip() or (
        f"/run/user/{os.getuid()}" if os.path.isdir(f"/run/user/{os.getuid()}") else ""
    )
    if rt:
        for wn in ("wayland-0", "wayland-1"):
            sock = os.path.join(rt, wn)
            if os.path.exists(sock):
                e = dict(base)
                e["XDG_RUNTIME_DIR"] = rt
                e["WAYLAND_DISPLAY"] = wn
                add(e)

    if not user_has_desktop:
        add(dict(base))
    return candidates


def _paint_x11_root_black_once() -> None:
    """Best-effort solid-black X11 root so compositor gaps never flash the normal desktop wallpaper."""
    global _jarvis_x11_root_painted
    if _jarvis_x11_root_painted or not JARVIS_X11_SOLID_ROOT:
        return
    xsetroot = shutil.which("xsetroot")
    if not xsetroot:
        return
    for env in _mpv_desktop_env_candidates():
        disp = (env.get("DISPLAY") or "").strip()
        if not disp:
            continue
        try:
            p = subprocess.run(
                [xsetroot, "-solid", "#000000"],
                env=env,
                capture_output=True,
                timeout=4,
            )
            if p.returncode == 0:
                _jarvis_x11_root_painted = True
                logging.info(
                    "Monitor: X11 root set solid black (JARVIS_X11_SOLID_ROOT=1); brief gaps show black not wallpaper"
                )
                return
        except (OSError, subprocess.TimeoutExpired) as e:
            logging.debug("xsetroot DISPLAY=%s: %s", disp, e)


def _spawn_mpv_desktop(
    image_path: str,
    ipc_server: str | None = None,
    *,
    settle_s: float = 0.45,
    overlay_fast: bool = False,
    defer_ontop: bool = False,
) -> tuple[subprocess.Popen | None, str | None]:
    """Fullscreen mpv via GPU/X11/Wayland when a session already owns the display (DRM unavailable).

    Returns (proc, ipc_socket_path) where ipc_socket_path is set only if this mpv was started with
    --input-ipc-server and that attempt succeeded (retries without IPC return None for the socket).
    """
    mpv = shutil.which("mpv")
    if not mpv:
        return None, None
    abspath = os.path.abspath(image_path)
    envs = _mpv_desktop_env_candidates()
    if overlay_fast and envs:
        envs = [envs[0]]
    has_desktop_hint = any(
        (e.get("DISPLAY") or "").strip() or (e.get("WAYLAND_DISPLAY") or "").strip() for e in envs
    ) or os.path.exists("/tmp/.X11-unix/X0")
    if not has_desktop_hint:
        return None, None

    # vo=wayland is often absent in distro mpv builds; vo=gpu uses Wayland when WAYLAND_DISPLAY is set.
    last_err = b""
    for env in envs:
        has_disp = bool((env.get("DISPLAY") or "").strip())
        # With DISPLAY=:0 (typical Pi desktop + SSH), x11 often works more reliably than gpu first.
        vos_try = ("x11", "gpu") if has_disp else ("gpu", "x11")
        if overlay_fast:
            vos_try = (vos_try[0],)
        for vo in vos_try:
            if vo == "x11" and not (env.get("DISPLAY") or "").strip():
                continue
            top_and_render: list[str] = []
            if defer_ontop:
                # Decode under the still-image mpv; X11/Wayland VOs skip rendering when fully occluded.
                top_and_render.append("--force-render")
            else:
                top_and_render.append("--ontop")
            cmd = [
                mpv,
                "--no-terminal",
                "--no-audio",
                "--mute=yes",
                *_mpv_monitor_hwdec_cmd_args(),
                "--fullscreen",
                *top_and_render,
                "--background=color",
                "--background-color=#FF000000",
                "--keep-open=always",
                "--force-window=immediate",
                f"--vo={vo}",
                "--loop-file=yes",
                "--image-display-duration=inf",
            ]
            if not IMAGE_MPV_VERBOSE:
                cmd.insert(2, "--msg-level=all:error")
            else:
                cmd[2:2] = ["-v", "--msg-level=all=warn"]
            if ipc_server:
                _unlink_mpv_ipc_quiet(ipc_server)
                cmd.extend(["--input-ipc-server", ipc_server])
            cmd.append(abspath)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
            )
            label = f"mpv {vo} DISPLAY={env.get('DISPLAY', '')!r} WAYLAND={env.get('WAYLAND_DISPLAY', '')!r}"
            ok, err_b = _wait_display_proc_alive(proc, label=label, settle_s=settle_s)
            if ok:
                logging.info(
                    "Monitor image shown via mpv vo=%s pid=%s DISPLAY=%s WAYLAND=%s",
                    vo,
                    proc.pid,
                    env.get("DISPLAY", ""),
                    env.get("WAYLAND_DISPLAY", ""),
                )
                return proc, (ipc_server if ipc_server else None)
            last_err = err_b
    if last_err:
        logging.warning("mpv desktop (gpu/x11) failed; last err=%r", last_err[:2000])
    if ipc_server:
        logging.info("mpv desktop: retrying without --input-ipc-server (IPC can break vo=x11/gpu on Pi desktop/SSH)")
        return _spawn_mpv_desktop(
            image_path,
            ipc_server=None,
            settle_s=settle_s,
            overlay_fast=False,
            defer_ontop=defer_ontop,
        )
    return None, None


def _spawn_mpv_desktop_file_ipc_fast(
    media_path: str,
    *,
    defer_ontop: bool = False,
) -> tuple[subprocess.Popen, str | None] | None:
    """Fullscreen mpv playing media_path immediately (argv) + IPC — fast overlay vs idle+loadfile."""
    mpv = shutil.which("mpv")
    if not mpv or not _desktop_session_likely():
        return None
    settle = _mpv_overlay_spawn_settle_for_path(media_path)
    ipc_sock = _mpv_ipc_socket_path_runtime()
    proc, ipc_eff = _spawn_mpv_desktop(
        media_path,
        ipc_server=ipc_sock,
        settle_s=settle,
        overlay_fast=True,
        defer_ontop=defer_ontop,
    )
    if proc is None:
        _unlink_mpv_ipc_quiet(ipc_sock)
        ipc_sock2 = _mpv_ipc_socket_path_runtime()
        proc, ipc_eff = _spawn_mpv_desktop(
            media_path,
            ipc_server=ipc_sock2,
            settle_s=settle,
            overlay_fast=False,
            defer_ontop=defer_ontop,
        )
    if proc is None:
        return None
    if defer_ontop and not ipc_eff:
        logging.info(
            "Monitor: defer-ontop loading mpv has no IPC; respawning with ontop (may show brief black)"
        )
        _terminate_display_proc(proc)
        _unlink_mpv_ipc_quiet(ipc_sock)
        return _spawn_mpv_desktop_file_ipc_fast(media_path, defer_ontop=False)
    if ipc_eff and not _wait_mpv_ipc_socket_ready(ipc_eff, proc, timeout_s=1.5):
        logging.debug("Monitor: IPC socket slow after fast mpv spawn (continuing)")
    return proc, ipc_eff


def _spawn_fbi(image_path: str) -> subprocess.Popen | None:
    fbi = shutil.which("fbi")
    if not fbi:
        return None
    if IMAGE_CHVT and IMAGE_VT and IMAGE_VT > 0:
        try:
            p = subprocess.run(
                ["chvt", str(IMAGE_VT)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=2,
            )
            if p.returncode == 0:
                logging.info("Monitor image: chvt to %d OK", IMAGE_VT)
            else:
                logging.warning(
                    "Monitor image: chvt to %d rc=%s err=%r (SSH cannot switch VT; prefer mpv drm)",
                    IMAGE_VT,
                    p.returncode,
                    (p.stderr or b"")[:2000],
                )
        except Exception:
            pass

    cmd = [fbi, "-a", "-d", IMAGE_FBDEV, "-nointeractive"]
    if IMAGE_VT and IMAGE_VT > 0:
        cmd.extend(["-T", str(IMAGE_VT)])
    cmd.append(image_path)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    ok, err_b = _wait_display_proc_alive(proc, label="fbi")
    if not ok:
        logging.warning("fbi failed DISPLAY=%r err_tail=%r", os.environ.get("DISPLAY", ""), (err_b or b"")[:2000])
        return None
    logging.info("Monitor image shown via fbi pid=%s image=%s", proc.pid, image_path)
    return proc


def _spawn_feh(image_path: str) -> subprocess.Popen | None:
    feh = shutil.which("feh")
    if not feh:
        return None
    abspath = os.path.abspath(image_path)
    cmd = [
        feh,
        "--fullscreen",
        "--borderless",
        "--hide-pointer",
        "--no-menus",
        "--image-bg",
        "black",
        abspath,
    ]
    last_err = b""
    for env in _mpv_desktop_env_candidates():
        disp = (env.get("DISPLAY") or "").strip()
        if not disp:
            continue
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        ok, err_b = _wait_display_proc_alive(proc, label=f"feh DISPLAY={disp!r}")
        if ok:
            logging.info("Monitor image shown via feh pid=%s DISPLAY=%s", proc.pid, disp)
            return proc
        last_err = err_b
    if last_err:
        logging.warning("feh failed all DISPLAY candidates err=%r", (last_err or b"")[:2000])
    return None


def _terminate_display_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _spawn_monitor_viewer(
    image_path: str,
    ipc_server: str | None,
    *,
    skip_drm: bool = False,
) -> tuple[subprocess.Popen | None, str | None, str]:
    """Start a fullscreen viewer. Returns (proc, ipc_socket_or_None, backend) where backend is
    drm | desktop | feh | fbi | none."""
    mode = IMAGE_VIEWER
    proc: subprocess.Popen | None = None
    ipc_used: str | None = None
    backend = "none"
    if mode == "mpv-drm":
        proc = _spawn_mpv_drm(image_path)
        if proc is not None:
            backend = "drm"
        return proc, None, backend
    elif mode == "fbi":
        proc = _spawn_fbi(image_path)
        if proc is not None:
            backend = "fbi"
        return proc, None, backend
    elif mode == "feh":
        proc = _spawn_feh(image_path)
        if proc is not None:
            backend = "feh"
        return proc, None, backend

    if skip_drm:
        # Prefer desktop mpv (with optional IPC) so openscreen → loading → still can use loadfile without feh gaps.
        proc, ipc_used = _spawn_mpv_desktop(image_path, ipc_server=ipc_server)
        if proc is not None:
            backend = "desktop"
        if proc is None:
            proc, ipc_used = _spawn_mpv_desktop(image_path, ipc_server=None)
            if proc is not None:
                backend = "desktop"
        if proc is None and image_path.lower().endswith(".png") and shutil.which("feh"):
            proc = _spawn_feh(image_path)
            if proc is not None:
                backend = "feh"
                ipc_used = None
        if proc is None and not image_path.lower().endswith(".png"):
            proc = _spawn_feh(image_path)
            if proc is not None:
                backend = "feh"
                ipc_used = None
        if proc is None:
            proc = _spawn_fbi(image_path)
            if proc is not None:
                backend = "fbi"
        return proc, ipc_used, backend

    if _auto_try_drm_before_desktop():
        proc = _spawn_mpv_drm(image_path)
        if proc is not None:
            backend = "drm"
        if proc is None:
            proc, ipc_used = _spawn_mpv_desktop(image_path, ipc_server=ipc_server)
            if proc is not None:
                backend = "desktop"
    else:
        proc, ipc_used = _spawn_mpv_desktop(image_path, ipc_server=ipc_server)
        if proc is not None:
            backend = "desktop"
        if proc is None:
            proc = _spawn_mpv_drm(image_path)
            if proc is not None:
                backend = "drm"
    if proc is None:
        proc = _spawn_feh(image_path)
        if proc is not None:
            backend = "feh"
    if proc is None:
        proc = _spawn_fbi(image_path)
        if proc is not None:
            backend = "fbi"
    return proc, ipc_used, backend


def _show_image_on_monitor(image_path: str, turn_id: int) -> None:
    """Fullscreen-display image_path on the monitor (replaces previous image)."""
    global _image_display_proc, _image_display_path, _image_mpv_ipc_socket, _image_display_via_desktop

    # Only replace the image if this is still the newest requested turn.
    with _image_display_state_lock:
        if turn_id != _image_latest_turn_id:
            return
        old_proc = _image_display_proc
        old_path = _image_display_path
        ipc_sock = _image_mpv_ipc_socket
        skip_drm = (
            old_proc is not None
            and old_proc.poll() is None
            and _image_display_via_desktop
        )

    display_path = _monitor_raster_copy_for_playback(image_path)
    if display_path != image_path:
        logging.info("Monitor: raster playback path %s (from %s)", display_path, image_path)

    _log_image_file_magic(display_path, "Monitor image file")

    # Same-mpv IPC for loading video → final still avoids closing the fullscreen mpv (main fix for desktop gap).
    type_cross = _monitor_handoff_loadfile_risky(old_path, display_path)
    prefer_ipc_video_to_still = (
        bool(old_path)
        and _loading_media_is_video(old_path)
        and _monitor_path_is_still_image(display_path)
    )
    cross_risky = type_cross and not prefer_ipc_video_to_still
    if cross_risky:
        logging.debug(
            "Monitor: cross-type handoff %r → %r — skipping IPC loadfile, using on-top overlay",
            (old_path or "")[-80:],
            display_path[-80:],
        )
    elif prefer_ipc_video_to_still:
        logging.info(
            "Monitor: same-mpv IPC loadfile loading video → final still (avoids compositor gap from killing mpv)"
        )

    # Same mpv process: swap file via JSON IPC (no second window → no compositor/desktop gap).
    can_ipc = (
        not cross_risky
        and old_proc is not None
        and old_proc.poll() is None
        and ipc_sock is not None
        and os.path.exists(ipc_sock)
    )
    if can_ipc:
        for attempt in range(1, 10):
            if _mpv_ipc_loadfile(ipc_sock, display_path):
                settle = JARVIS_MPV_IPC_LOADFILE_SETTLE_S
                if prefer_ipc_video_to_still:
                    settle += JARVIS_MPV_IPC_VIDEO_TO_STILL_EXTRA_SETTLE_S
                if settle > 0:
                    time.sleep(settle)
                with _image_display_state_lock:
                    if turn_id != _image_latest_turn_id:
                        return
                    _image_display_path = display_path
                _remove_transient_monitor_file(old_path)
                logging.info(
                    "Monitor: seamless mpv IPC handoff → %s (attempt %d, settle=%.2fs video_to_still_extra=%.2fs)",
                    display_path,
                    attempt,
                    settle,
                    JARVIS_MPV_IPC_VIDEO_TO_STILL_EXTRA_SETTLE_S if prefer_ipc_video_to_still else 0.0,
                )
                return
            time.sleep(0.09)
        logging.info("Monitor: mpv IPC loadfile failed after retries; spawning new viewer process")

    # Start a new fullscreen mpv on top, then stop the old viewer — avoids killing first (desktop flash).
    if (
        skip_drm
        and old_proc is not None
        and old_proc.poll() is None
        and _desktop_session_likely()
        and shutil.which("mpv")
    ):
        got_overlay = _spawn_mpv_desktop_file_ipc_fast(display_path)
        if got_overlay:
            nproc, nsock = got_overlay
            backend_o = "desktop"
            with _image_display_state_lock:
                if turn_id != _image_latest_turn_id:
                    _terminate_display_proc(nproc)
                    _unlink_mpv_ipc_quiet(nsock)
                    return
            if JARVIS_MPV_OVERLAY_HANDOFF_SETTLE_S > 0:
                time.sleep(JARVIS_MPV_OVERLAY_HANDOFF_SETTLE_S)

            # Still on top of video: do not close loading until overlay mpv has had time to decode and paint.
            if _monitor_path_is_still_image(display_path) and JARVIS_MPV_PRE_KILL_STILL_SETTLE_S > 0:
                ok_pre, _ = _wait_display_proc_alive(
                    nproc,
                    label=f"mpv overlay decode before close-loading pid={nproc.pid}",
                    settle_s=JARVIS_MPV_PRE_KILL_STILL_SETTLE_S,
                )
                if not ok_pre:
                    rc = nproc.poll()
                    err_tail = b""
                    try:
                        _, err_tail = nproc.communicate(timeout=0.8)
                    except Exception:
                        pass
                    logging.warning(
                        "Monitor: overlay mpv died before closing loading (rc=%s); retrying viewer. err=%r",
                        rc,
                        (err_tail or b"")[:1200],
                    )
                    _terminate_display_proc(nproc)
                    _unlink_mpv_ipc_quiet(nsock)
                    new_ipc_b = _new_mpv_ipc_socket_path() if shutil.which("mpv") else None
                    nproc_b, nsock_b, backend_o = _spawn_monitor_viewer(
                        display_path, new_ipc_b, skip_drm=False
                    )
                    if nproc_b is None:
                        logging.error("Monitor: fallback viewer failed for final image %s", display_path)
                        with _image_display_state_lock:
                            if turn_id == _image_latest_turn_id:
                                _image_display_proc = None
                                _image_display_path = None
                                _image_mpv_ipc_socket = None
                                _image_display_via_desktop = False
                        return
                    nproc, nsock = nproc_b, nsock_b
                    settle_b = min(0.45, JARVIS_MPV_PRE_KILL_STILL_SETTLE_S)
                    if settle_b > 0:
                        _wait_display_proc_alive(
                            nproc,
                            label=f"mpv fallback before close-loading pid={nproc.pid}",
                            settle_s=settle_b,
                        )

            _terminate_display_proc(old_proc)
            _unlink_mpv_ipc_quiet(ipc_sock)
            _remove_transient_monitor_file(old_path)
            post_kill = 0.22 if _monitor_path_is_still_image(display_path) else 0.4
            ok_live, _ = _wait_display_proc_alive(
                nproc,
                label=f"mpv overlay final still pid={nproc.pid}",
                settle_s=post_kill,
            )
            if not ok_live:
                rc = nproc.poll()
                err_tail = b""
                try:
                    _, err_tail = nproc.communicate(timeout=0.8)
                except Exception:
                    pass
                logging.warning(
                    "Monitor: overlay mpv died after closing loading (rc=%s); retrying viewer for final image. err=%r",
                    rc,
                    (err_tail or b"")[:1200],
                )
                _terminate_display_proc(nproc)
                _unlink_mpv_ipc_quiet(nsock)
                new_ipc_r = _new_mpv_ipc_socket_path() if shutil.which("mpv") else None
                nproc2, nsock2, backend_o = _spawn_monitor_viewer(
                    display_path, new_ipc_r, skip_drm=False
                )
                if nproc2 is None:
                    logging.error("Monitor: fallback viewer failed for final image %s", display_path)
                    with _image_display_state_lock:
                        if turn_id == _image_latest_turn_id:
                            _image_display_proc = None
                            _image_display_path = None
                            _image_mpv_ipc_socket = None
                            _image_display_via_desktop = False
                    return
                nproc, nsock = nproc2, nsock2
            with _image_display_state_lock:
                if turn_id != _image_latest_turn_id:
                    _terminate_display_proc(nproc)
                    _unlink_mpv_ipc_quiet(nsock)
                    return
                _image_display_proc = nproc
                _image_display_path = display_path
                _image_mpv_ipc_socket = nsock
                _image_display_via_desktop = backend_o in ("desktop", "feh")
            logging.info("Monitor: desktop handoff via on-top mpv (final image on screen)")
            return

    # Overlay handoff unavailable: stop the old viewer, then show the new one (brief gap only if unavoidable).
    if skip_drm and old_proc is not None and old_proc.poll() is None:
        logging.info("Monitor: closing prior viewer before opening new (no overlay handoff)")
        _terminate_display_proc(old_proc)
        _unlink_mpv_ipc_quiet(ipc_sock)
        old_proc = None
        ipc_sock = None
        with _image_display_state_lock:
            if turn_id == _image_latest_turn_id:
                _image_display_proc = None
                _image_display_path = None
                _image_mpv_ipc_socket = None
        time.sleep(0.02)

    proc: subprocess.Popen | None = None
    ipc_got: str | None = None
    backend = "none"
    new_ipc: str | None = None
    if _desktop_session_likely() and shutil.which("mpv"):
        got_fast = _spawn_mpv_desktop_file_ipc_fast(display_path)
        if got_fast:
            proc, ipc_got = got_fast
            backend = "desktop"
        else:
            got_idle = _spawn_mpv_desktop_idle_ipc_then_load(display_path)
            if got_idle:
                proc, ipc_got = got_idle
                backend = "desktop"
    if proc is None:
        new_ipc = _new_mpv_ipc_socket_path() if shutil.which("mpv") else None
        proc, ipc_got, backend = _spawn_monitor_viewer(display_path, new_ipc, skip_drm=skip_drm)

    # If we were on X11 mpv for the loading clip, never fall through to vo=drm for the still —
    # DRM drives a different pipe than the desktop you are looking at.
    if proc is None and skip_drm and old_proc is not None and old_proc.poll() is None:
        logging.warning(
            "Monitor: desktop viewers failed for this file; stopping prior window and trying full fallback (DRM, etc.)"
        )
        _terminate_display_proc(old_proc)
        _unlink_mpv_ipc_quiet(ipc_sock)
        old_proc = None
        _unlink_mpv_ipc_quiet(new_ipc)
        new_ipc = _new_mpv_ipc_socket_path() if shutil.which("mpv") else None
        with _image_display_state_lock:
            if turn_id == _image_latest_turn_id:
                _image_display_proc = None
                _image_display_path = None
                _image_mpv_ipc_socket = None
                _image_display_via_desktop = False
        proc, ipc_got, backend = _spawn_monitor_viewer(display_path, new_ipc, skip_drm=False)

    # vo=drm often allows only one master; a second mpv may fail until the first exits.
    if proc is None and old_proc is not None and old_proc.poll() is None:
        logging.info("Monitor viewer handoff: retry after stopping previous (DRM exclusive mode).")
        _terminate_display_proc(old_proc)
        _unlink_mpv_ipc_quiet(ipc_sock)
        old_proc = None
        _unlink_mpv_ipc_quiet(new_ipc)
        new_ipc = _new_mpv_ipc_socket_path() if shutil.which("mpv") else None
        with _image_display_state_lock:
            if turn_id == _image_latest_turn_id:
                _image_display_proc = None
                _image_display_path = None
                _image_mpv_ipc_socket = None
        proc, ipc_got, backend = _spawn_monitor_viewer(display_path, new_ipc, skip_drm=False)

    if proc is None:
        logging.warning(
            "Could not show monitor image. With a desktop running, DRM is usually busy — "
            "auto mode prefers desktop mpv when X11/Wayland is detected. "
            "Try: export DISPLAY=:0  or  JARVIS_IMAGE_TRY_DRM_FIRST=0  or  python jarvis.py --test-monitor"
        )
        _unlink_mpv_ipc_quiet(new_ipc)
        return

    with _image_display_state_lock:
        if turn_id != _image_latest_turn_id:
            _terminate_display_proc(proc)
            _unlink_mpv_ipc_quiet(ipc_got)
            return

    # Drop previous viewer after the new one is ready (covers DRM → desktop and similar).
    _terminate_display_proc(old_proc)
    _unlink_mpv_ipc_quiet(ipc_sock)
    _remove_transient_monitor_file(old_path)

    with _image_display_state_lock:
        if turn_id != _image_latest_turn_id:
            _terminate_display_proc(proc)
            _unlink_mpv_ipc_quiet(ipc_got)
            return
        _image_display_proc = proc
        _image_display_path = display_path
        _image_mpv_ipc_socket = ipc_got
        _image_display_via_desktop = backend in ("desktop", "feh")


def _show_loading_placeholder_desktop_ipc_first(loading_path: str, turn_id: int) -> None:
    """Show the loading clip; prefer IPC loadfile or on-top idle mpv (no desktop flash)."""
    global _image_display_proc, _image_display_path, _image_mpv_ipc_socket, _image_display_via_desktop

    with _image_display_state_lock:
        if turn_id != _image_latest_turn_id:
            return
        old_proc = _image_display_proc
        old_path = _image_display_path
        ipc_sock = _image_mpv_ipc_socket
        skip_drm = (
            old_proc is not None
            and old_proc.poll() is None
            and _image_display_via_desktop
        )

    abspath_lp = os.path.abspath(loading_path)
    type_cross_load = _monitor_handoff_loadfile_risky(old_path, loading_path)
    prefer_ipc_still_to_video = (
        bool(old_path)
        and _monitor_path_is_still_image(old_path)
        and _loading_media_is_video(loading_path)
    )
    cross_risky = type_cross_load and not prefer_ipc_still_to_video
    if cross_risky:
        logging.debug(
            "Monitor: cross-type handoff %r → loading — skipping IPC loadfile, using on-top overlay",
            (old_path or "")[-80:],
        )
    elif prefer_ipc_still_to_video:
        logging.info(
            "Monitor: same-mpv IPC loadfile still → loading video (openscreen or prior final; no overlay kill)"
        )

    if (
        not cross_risky
        and ipc_sock
        and old_proc is not None
        and old_proc.poll() is None
        and os.path.exists(ipc_sock)
    ):
        for attempt in range(1, 16):
            ok_ld = False
            used_append_next = False
            if prefer_ipc_still_to_video and JARVIS_MPV_STILL_TO_VIDEO_APPEND:
                ok_ld = _mpv_ipc_still_to_video_seamless(ipc_sock, abspath_lp)
                used_append_next = ok_ld
            if not ok_ld:
                ok_ld = _mpv_ipc_loadfile(ipc_sock, abspath_lp)
                if not ok_ld and prefer_ipc_still_to_video:
                    time.sleep(0.04)
                    ok_ld = _mpv_ipc_loadfile(ipc_sock, abspath_lp)
            if ok_ld:
                if prefer_ipc_still_to_video and ipc_sock:
                    if not _mpv_ipc_wait_loading_video_painted(
                        ipc_sock, timeout_s=JARVIS_MPV_LOADING_FIRST_FRAME_WAIT_S
                    ):
                        logging.debug("Monitor: IPC still→video first-frame wait timed out")
                settle_ld = JARVIS_MPV_IPC_LOADFILE_SETTLE_S
                if prefer_ipc_still_to_video:
                    settle_ld += JARVIS_MPV_IPC_STILL_TO_VIDEO_EXTRA_SETTLE_S
                if settle_ld > 0:
                    time.sleep(settle_ld)
                with _image_display_state_lock:
                    if turn_id != _image_latest_turn_id:
                        return
                    _image_display_path = loading_path
                _remove_transient_monitor_file(old_path)
                logging.info(
                    "Monitor: IPC → loading clip (%s, attempt %d, settle=%.2fs)",
                    "append+playlist-next" if used_append_next else "loadfile replace",
                    attempt,
                    settle_ld,
                )
                return
            time.sleep(0.08)
        if prefer_ipc_still_to_video:
            logging.info(
                "Monitor: IPC still→loading video failed after retries; using overlay (underlay hides wallpaper if enabled)"
            )

    if (
        skip_drm
        and old_proc is not None
        and old_proc.poll() is None
        and _desktop_session_likely()
        and shutil.which("mpv")
    ):
        got_overlay = _spawn_mpv_desktop_file_ipc_fast(
            loading_path, defer_ontop=prefer_ipc_still_to_video
        )
        if got_overlay:
            nproc, nsock = got_overlay
            with _image_display_state_lock:
                if turn_id != _image_latest_turn_id:
                    _terminate_display_proc(nproc)
                    _unlink_mpv_ipc_quiet(nsock)
                    return
            if JARVIS_MPV_OVERLAY_HANDOFF_SETTLE_S > 0:
                time.sleep(JARVIS_MPV_OVERLAY_HANDOFF_SETTLE_S)
            # If we raise loading with ontop while the still viewer is still alive, we briefly have two
            # ontop fullscreen windows; killing the still then makes the compositor flash the black underlay.
            # Remove the still first (loading is already decoded underneath), then pin loading above underlay.
            killed_still_before_ontop = False
            if prefer_ipc_still_to_video and nsock:
                if not _mpv_ipc_wait_loading_video_painted(
                    nsock, timeout_s=JARVIS_MPV_LOADING_FIRST_FRAME_WAIT_S
                ):
                    logging.debug(
                        "Monitor: defer-ontop loading first-frame wait timed out before still removal"
                    )
                _terminate_display_proc(old_proc)
                _unlink_mpv_ipc_quiet(ipc_sock)
                killed_still_before_ontop = True
                if not _mpv_ipc_request(nsock, ["set_property", "ontop", True]):
                    logging.debug(
                        "Monitor: IPC set_property ontop failed after still removed (stacking may be wrong)"
                    )
                time.sleep(0.05)
            elif _loading_media_is_video(loading_path) and JARVIS_MPV_PRE_KILL_LOADING_VIDEO_SETTLE_S > 0:
                ok_lv, _ = _wait_display_proc_alive(
                    nproc,
                    label=f"mpv loading overlay before close-still pid={nproc.pid}",
                    settle_s=JARVIS_MPV_PRE_KILL_LOADING_VIDEO_SETTLE_S,
                )
                if not ok_lv:
                    logging.warning(
                        "Monitor: loading overlay mpv died before closing still viewer (rc=%s)",
                        nproc.poll(),
                    )
            if not killed_still_before_ontop:
                _terminate_display_proc(old_proc)
                _unlink_mpv_ipc_quiet(ipc_sock)
            with _image_display_state_lock:
                if turn_id != _image_latest_turn_id:
                    _terminate_display_proc(nproc)
                    _unlink_mpv_ipc_quiet(nsock)
                    return
                _image_display_proc = nproc
                _image_display_path = loading_path
                _image_mpv_ipc_socket = nsock
                _image_display_via_desktop = True
            if nsock is None and _loading_media_is_video(loading_path):
                logging.info(
                    "Monitor: loading mpv had no IPC socket; replacing with idle+IPC for video→still loadfile"
                )
                _terminate_display_proc(nproc)
                _unlink_mpv_ipc_quiet(nsock)
                idle_reload = _spawn_mpv_desktop_idle_ipc_then_load(loading_path)
                if idle_reload:
                    nproc, nsock = idle_reload
                    with _image_display_state_lock:
                        if turn_id != _image_latest_turn_id:
                            _terminate_display_proc(nproc)
                            _unlink_mpv_ipc_quiet(nsock)
                            return
                        _image_display_proc = nproc
                        _image_display_path = loading_path
                        _image_mpv_ipc_socket = nsock
                        _image_display_via_desktop = True
                else:
                    fg = _spawn_mpv_desktop_file_ipc_fast(loading_path)
                    if fg:
                        nproc, nsock = fg
                        with _image_display_state_lock:
                            if turn_id != _image_latest_turn_id:
                                _terminate_display_proc(nproc)
                                _unlink_mpv_ipc_quiet(nsock)
                                return
                            _image_display_proc = nproc
                            _image_display_path = loading_path
                            _image_mpv_ipc_socket = nsock
                            _image_display_via_desktop = True
            _remove_transient_monitor_file(old_path)
            logging.info("Monitor: loading clip via on-top mpv (handoff from prior viewer)")
            return

    if skip_drm and old_proc is not None and old_proc.poll() is None:
        logging.info("Monitor: replacing prior desktop viewer before loading (fallback)")
        _terminate_display_proc(old_proc)
        _unlink_mpv_ipc_quiet(ipc_sock)
        with _image_display_state_lock:
            if turn_id == _image_latest_turn_id:
                _image_display_proc = None
                _image_display_path = None
                _image_mpv_ipc_socket = None
        time.sleep(0.02)

    # Cold spawn after prior viewer died: fast mpv first (quick fullscreen), then idle+IPC if needed for socket.
    got = _spawn_mpv_desktop_file_ipc_fast(loading_path)
    if not got:
        got = _spawn_mpv_desktop_idle_ipc_then_load(loading_path)
    if not got:
        _show_image_on_monitor(loading_path, turn_id)
        return

    proc, sock = got
    if sock is None and _loading_media_is_video(loading_path):
        logging.info(
            "Monitor: loading mpv without IPC; respawning idle+IPC so final uses same-process loadfile"
        )
        _terminate_display_proc(proc)
        _unlink_mpv_ipc_quiet(sock)
        got = _spawn_mpv_desktop_idle_ipc_then_load(loading_path)
        if not got:
            got = _spawn_mpv_desktop_file_ipc_fast(loading_path)
        if not got:
            _show_image_on_monitor(loading_path, turn_id)
            return
        proc, sock = got

    with _image_display_state_lock:
        if turn_id != _image_latest_turn_id:
            _terminate_display_proc(proc)
            _unlink_mpv_ipc_quiet(sock)
            return
        _image_display_proc = proc
        _image_display_path = loading_path
        _image_mpv_ipc_socket = sock
        _image_display_via_desktop = True
    _remove_transient_monitor_file(old_path)
    logging.info("Monitor: loading clip on mpv (fast or idle+IPC) → handoff to final image")
    return


def _show_openscreen_via_desktop_idle_ipc(path: str, turn_id: int) -> bool:
    """Use idle+IPC mpv for openscreen so loading/final can loadfile in the same process (no feh gap)."""
    global _image_display_proc, _image_display_path, _image_mpv_ipc_socket, _image_display_via_desktop

    if not _desktop_session_likely() or not shutil.which("mpv"):
        return False
    display_path = _monitor_raster_copy_for_playback(path)
    if display_path != path:
        logging.info("Monitor openscreen raster path %s (from %s)", display_path, path)
    # Idle+IPC first so --input-ipc-server is reliable for PNG → loading MP4 loadfile (no overlay gap).
    got = _spawn_mpv_desktop_idle_ipc_then_load(display_path)
    if not got:
        got = _spawn_mpv_desktop_file_ipc_fast(display_path)
    if not got:
        return False
    proc, sock = got
    if sock is None:
        logging.info("Monitor: openscreen mpv without IPC; respawning idle+IPC for still→loading loadfile")
        _terminate_display_proc(proc)
        _unlink_mpv_ipc_quiet(sock)
        got = _spawn_mpv_desktop_idle_ipc_then_load(display_path)
        if not got:
            got = _spawn_mpv_desktop_file_ipc_fast(display_path)
        if not got:
            return False
        proc, sock = got
    with _image_display_state_lock:
        if turn_id != _image_latest_turn_id:
            _terminate_display_proc(proc)
            _unlink_mpv_ipc_quiet(sock)
            return False
        _image_display_proc = proc
        _image_display_path = display_path
        _image_mpv_ipc_socket = sock
        _image_display_via_desktop = True
    logging.info("Monitor: openscreen on mpv IPC session (seamless → loading → still)")
    return True


def _show_openscreen_at_startup() -> None:
    """Show wj_openscreen.png on the monitor until the first voice turn (turn >= 1)."""
    global _image_latest_turn_id
    if not IMAGE_ENABLED:
        return
    path = os.environ.get("JARVIS_OPENSCREEN_PATH", "").strip() or _BUNDLED_OPENSCREEN_PNG
    if not os.path.isfile(path):
        logging.warning(
            "Openscreen PNG not found (%s); monitor stays on desktop until the first question. "
            "Override: JARVIS_OPENSCREEN_PATH=/path/to.png",
            path,
        )
        return
    with _image_display_state_lock:
        _image_latest_turn_id = IMAGE_OPENSCREEN_TURN_ID
    logging.info(
        "Monitor idle openscreen (until first question): %s",
        path,
    )
    try:
        if not _show_openscreen_via_desktop_idle_ipc(path, IMAGE_OPENSCREEN_TURN_ID):
            _show_image_on_monitor(path, IMAGE_OPENSCREEN_TURN_ID)
    except Exception as e:
        logging.warning("Openscreen display failed: %s", e)


def _monitor_begin_turn_after_stt(turn_id: int) -> None:
    """Call immediately after STT: claim the monitor for this turn and show the loading clip during Venice chat."""
    global _image_latest_turn_id
    if not IMAGE_ENABLED or not VENICE_API_KEY:
        return
    with _image_display_state_lock:
        _image_latest_turn_id = turn_id
    if not IMAGE_PLACEHOLDER_ENABLED:
        return
    try:
        loading_path = _ensure_loading_media()
        _show_loading_placeholder_desktop_ipc_first(loading_path, turn_id)
    except Exception as e:
        logging.warning("Monitor loading at question time failed for turn %d: %s", turn_id, e)


def _start_image_thread(turn_id: int, question: str, reply: str) -> None:
    """Start image generation in parallel with TTS playback."""
    global _image_latest_turn_id
    if not IMAGE_ENABLED:
        return
    if not VENICE_API_KEY:
        return

    with _image_display_state_lock:
        _image_latest_turn_id = turn_id

    def _worker() -> None:
        try:
            logging.info("Monitor image worker starting for turn %d", turn_id)
            t_prompt = time.monotonic()
            use_prompt_llm = IMAGE_PROMPT_AGENT_ENABLED and not IMAGE_FAST_IMAGE_PROMPT
            if use_prompt_llm:
                prompt = _get_image_prompt(question, reply)
            else:
                prompt = _image_prompt_without_llm(question, reply)
                logging.info(
                    "Monitor image prompt from reply in %.2fs (fast=%s prompt_agent=%s → Venice /image/generate next)",
                    time.monotonic() - t_prompt,
                    IMAGE_FAST_IMAGE_PROMPT,
                    IMAGE_PROMPT_AGENT_ENABLED,
                )
            if use_prompt_llm:
                logging.info(
                    "Monitor image prompt LLM step took %.2fs (web_search=%s model=%s)",
                    time.monotonic() - t_prompt,
                    IMAGE_PROMPT_WEB_SEARCH,
                    IMAGE_PROMPT_MODEL or CHAT_MODEL,
                )
            ext = "jpeg" if (IMAGE_FORMAT == "jpg" or IMAGE_FORMAT == "jpeg") else IMAGE_FORMAT
            out_path = os.path.join(IMAGE_TMP_DIR, f"turn_{turn_id}.{ext}")
            os.makedirs(IMAGE_TMP_DIR, exist_ok=True)
            t_img = time.monotonic()
            _generate_monitor_image(prompt, out_path)
            logging.info("Monitor image Venice /image/generate wall time %.2fs", time.monotonic() - t_img)
            try:
                sz = os.path.getsize(out_path)
            except OSError:
                sz = -1
            logging.info("Monitor image generated turn=%d path=%s bytes=%d", turn_id, out_path, sz)
            _show_image_on_monitor(out_path, turn_id)
        except Exception as e:
            logging.warning("Image generation/show failed for turn %d: %s", turn_id, e)

    t = threading.Thread(target=_worker, daemon=True, name=f"jarvis-image-gen-{turn_id}")
    t.start()

# Optional: download all TTS chunks in parallel, merge them into a single WAV,
# and play once. This removes the chunk boundary pause completely, but it
# increases time-to-first-audio because playback waits for all downloads.
TTS_MERGE_ONCE = bool(_env_int("JARVIS_TTS_MERGE_ONCE", 0))
TTS_MERGE_WORKERS = max(1, _env_int("JARVIS_TTS_MERGE_WORKERS", 4))

# Best-effort logging of the produced audio duration (helps confirm truncation
# vs playback issues). Set 0 to disable.
TTS_PROBE_DURATION = _env_int("JARVIS_TTS_PROBE_DURATION", 1)

# === OUTPUT AUDIO (ffplay/SDL + aplay fallback) ===
# Default: HDMI / monitor speakers (Pi 5 micro-HDMI). We auto-pick a card from `aplay -l` whose
# name matches SDL_ALSA_OUTPUT_KEYWORDS (vc4-hdmi, hdmi, …).
#
# REVERT to USB speakers (pick one — no need to do both):
#   A) Environment (good for quick tests): before starting Jarvis,
#          export JARVIS_SDL_ALSA_OUTPUT_DEVICE=hw:3,0
#      Use the card/device numbers from `aplay -l` for your USB adapter (often card 2 or 3).
#   B) In this file: set _DEFAULT_SDL_ALSA_OUTPUT_DEVICE below back to "hw:3,0" (or your USB hw:N,M).
#
# Optional override to any ALSA device (HDMI or USB), e.g. hw:3,0 or plughw:0,0:
#   export JARVIS_SDL_ALSA_OUTPUT_DEVICE=hw:X,Y
_DEFAULT_SDL_ALSA_OUTPUT_DEVICE = ""
SDL_ALSA_OUTPUT_DEVICE = os.environ.get(
    "JARVIS_SDL_ALSA_OUTPUT_DEVICE", _DEFAULT_SDL_ALSA_OUTPUT_DEVICE
).strip()
# Keywords used to auto-detect the right ALSA card from `aplay -l`.
SDL_ALSA_OUTPUT_KEYWORDS = ["hdmi", "vc4-hdmi", "vc4hdmi", "monitor", "display"]
# Pi 5 micro-HDMI near the USB-C power port is usually ALSA "vc4-hdmi-0" (HDMI0). The other port is vc4-hdmi-1.
# Set JARVIS_HDMI_PORT=1 if your cable is on the second micro-HDMI.
# (Do not use _env_int here — it clamps to min 1, so default 0 would wrongly become 1.)
def _hdmi_port_from_env() -> int:
    raw = os.environ.get("JARVIS_HDMI_PORT", "").strip()
    if not raw:
        return 0
    try:
        return max(0, min(1, int(raw)))
    except ValueError:
        return 0


JARVIS_HDMI_PORT = _hdmi_port_from_env()

_CACHED_SDL_AUDIO_ENV: dict[str, str] | None = None


def _alsa_plug_device(spec: str) -> str:
    """Use plughw: for SDL/ffplay — raw hw: often fails with 'Unsupported audio format' on vc4 HDMI."""
    s = (spec or "").strip()
    if s.lower().startswith("hw:"):
        return "plughw:" + s[3:].lstrip(":")
    return s


def _hdmi_card_sort_key(port_want: int, card_name: str) -> tuple[int, int]:
    """Lower tuple sorts first. Prefer vc4-hdmi-0 vs vc4-hdmi-1 to match Pi HDMI0 / HDMI1."""
    n = card_name.lower()
    if port_want == 0:
        if "vc4-hdmi-0" in n or n == "vc4-hdmi-0":
            return (0, 0)
        if "vc4-hdmi-1" in n or n == "vc4-hdmi-1":
            return (0, 2)
    else:
        if "vc4-hdmi-1" in n or n == "vc4-hdmi-1":
            return (0, 0)
        if "vc4-hdmi-0" in n or n == "vc4-hdmi-0":
            return (0, 2)
    if "hdmi" in n:
        return (0, 1)
    return (1, 0)


def _resolve_sdl_audio_env() -> dict[str, str]:
    """
    Returns env vars that force ffplay/SDL to use the intended ALSA output device.

    Uses ``AUDIODEV=plughw:X,Y`` (not raw hw:) with ``SDL_AUDIODRIVER=alsa`` so SDL can open
    HDMI on Pi (see SDL_OpenAudio … Unsupported audio format with hw:).
    """
    global _CACHED_SDL_AUDIO_ENV
    if _CACHED_SDL_AUDIO_ENV is not None:
        return _CACHED_SDL_AUDIO_ENV

    if SDL_ALSA_OUTPUT_DEVICE:
        plug = _alsa_plug_device(SDL_ALSA_OUTPUT_DEVICE)
        _CACHED_SDL_AUDIO_ENV = {"SDL_AUDIODRIVER": "alsa", "AUDIODEV": plug}
        logging.info(
            "Audio output: forced ALSA device for ffplay/SDL (AUDIODEV=%s). "
            "Unset JARVIS_SDL_ALSA_OUTPUT_DEVICE or clear _DEFAULT_SDL_ALSA_OUTPUT_DEVICE for HDMI auto-detect.",
            plug,
        )
        return _CACHED_SDL_AUDIO_ENV

    aplay = shutil.which("aplay")
    if not aplay:
        _CACHED_SDL_AUDIO_ENV = {}
        return _CACHED_SDL_AUDIO_ENV

    try:
        r = subprocess.run([aplay, "-l"], capture_output=True, text=True, timeout=5)
        out = r.stdout or ""
    except Exception:
        _CACHED_SDL_AUDIO_ENV = {}
        return _CACHED_SDL_AUDIO_ENV

    # Example line:
    # card 0: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
    card_re = re.compile(r"card\s+(\d+):[^\[]*\[([^\]]+)\],\s+device\s+(\d+):", re.IGNORECASE)
    candidates: list[tuple[int, int, str]] = []
    for m in card_re.finditer(out):
        card_id = int(m.group(1))
        card_name = m.group(2)
        dev_id = int(m.group(3))
        hay = card_name.lower()
        if any(k in hay for k in SDL_ALSA_OUTPUT_KEYWORDS):
            candidates.append((card_id, dev_id, card_name))

    if not candidates:
        _CACHED_SDL_AUDIO_ENV = {}
        return _CACHED_SDL_AUDIO_ENV

    candidates.sort(key=lambda t: _hdmi_card_sort_key(JARVIS_HDMI_PORT, t[2]))
    chosen = candidates[0]
    plug = f"plughw:{chosen[0]},{chosen[1]}"
    _CACHED_SDL_AUDIO_ENV = {
        "SDL_AUDIODRIVER": "alsa",
        "AUDIODEV": plug,
    }
    logging.info(
        "Audio output: auto-detected HDMI from aplay -l card=%d dev=%d label=%r AUDIODEV=%s (JARVIS_HDMI_PORT=%d)",
        chosen[0],
        chosen[1],
        chosen[2],
        plug,
        JARVIS_HDMI_PORT,
    )
    return _CACHED_SDL_AUDIO_ENV


_SCRIPT_DIR = Path(__file__).resolve().parent


def setup_logging() -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    h_err = logging.StreamHandler()
    h_err.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(h_err)
    log_path = os.environ.get("JARVIS_LOGFILE", "").strip()
    if not log_path:
        log_path = str(_SCRIPT_DIR / "jarvis.log")
    try:
        h_file = logging.FileHandler(log_path, encoding="utf-8")
        h_file.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(h_file)
        logging.info("Logging to %s", log_path)
    except OSError as e:
        logging.warning("Could not open log file %s: %s", log_path, e)


def log_audio_devices() -> None:
    try:
        d_in, d_out = sd.default.device
        devices = sd.query_devices()
        di = devices[d_in] if isinstance(d_in, int) else {}
        do = devices[d_out] if isinstance(d_out, int) else {}
        logging.info(
            "Default audio — input: %s | output: %s",
            di.get("name", d_in),
            do.get("name", d_out),
        )
    except Exception as e:
        logging.warning("Could not query audio devices: %s", e)


def stats_summary() -> str:
    with _stats_lock:
        c = _stats.copy()
    return (
        f'turns={c["turns"]} | Venice chat OK={c["chat_ok"]} err={c["chat_err"]} | '
        f'TTS OK={c["tts_ok"]} err={c["tts_err"]}'
    )


class Phase(Enum):
    LISTEN_WAKE = auto()
    RECORD_CMD = auto()


def jarvis_score(preds: dict) -> tuple[str, float]:
    """Return (label, score) for any loaded model whose key looks like the Jarvis wake model."""
    best_label, best = "", 0.0
    for label, score in preds.items():
        if "jarvis" in label.lower():
            if score > best:
                best_label, best = label, float(score)
    return best_label, best


def text_for_tts(text: str) -> str:
    """Strip markdown / line noise so the speech API gets speakable text."""
    if not text:
        return ""
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    t = t.replace("`", "")
    t = re.sub(r"\^\d+\^", "", t)
    t = re.sub(r"#([\w]+)", r"\1", t)
    t = re.sub(r"\[\[(\d+)\]\]\([^)]+\)", r"footnote \1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clip_text_for_tts(text: str, max_chars: int) -> tuple[str, bool]:
    """Shorten very long replies for TTS only (avoids slow synth / timeouts)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    chunk = text[:max_chars]
    for sep in (". ", ".\n", "? ", "! ", "\n"):
        i = chunk.rfind(sep)
        if i > max_chars // 2:
            return text[: i + len(sep)].strip() + " […]", True
    sp = chunk.rfind(" ")
    if sp > max_chars // 2:
        return text[:sp].strip() + " […]", True
    return chunk.strip() + " […]", True


def split_for_tts(text: str, max_chunk_chars: int) -> list[str]:
    """Split text into chunks <= max_chunk_chars, preferring sentence boundaries."""
    if max_chunk_chars <= 0 or len(text) <= max_chunk_chars:
        return [text]

    # Split into sentences but keep punctuation with the sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur = ""

    def push_cur() -> None:
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not cur:
            if len(s) <= max_chunk_chars:
                cur = s
            else:
                # Sentence is still too long; split on spaces.
                start = 0
                while start < len(s):
                    end = min(len(s), start + max_chunk_chars)
                    if end < len(s):
                        # Prefer breaking at whitespace before end.
                        ws = s.rfind(" ", start, end)
                        if ws > start + max_chunk_chars // 2:
                            end = ws
                    part = s[start:end].strip()
                    if part:
                        chunks.append(part)
                    start = end
                cur = ""
        else:
            if len(cur) + 1 + len(s) <= max_chunk_chars:
                cur = cur + " " + s
            else:
                push_cur()
                if len(s) <= max_chunk_chars:
                    cur = s
                else:
                    # s is too long: split and append pieces.
                    start = 0
                    while start < len(s):
                        end = min(len(s), start + max_chunk_chars)
                        if end < len(s):
                            ws = s.rfind(" ", start, end)
                            if ws > start + max_chunk_chars // 2:
                                end = ws
                        part = s[start:end].strip()
                        if part:
                            chunks.append(part)
                        start = end
                    cur = ""

    push_cur()
    return chunks or [text]


def strip_citations_and_links(text: str) -> str:
    """Remove citation markup Venice/models still emit (footnotes, links, source blocks)."""
    if not text:
        return text
    t = text
    t = re.sub(r"\[\[\d+\]\]\([^\)]*\)", "", t)
    t = re.sub(r"\[([^\]]*)\]\([^\)]+\)", "", t)
    t = re.sub(r"(?i)\bSources?:[\s\S]*$", "", t)
    t = re.sub(r"(?i)\bReferences?:[\s\S]*$", "", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\^\d+\^", "", t)
    t = re.sub(r"\s+\[\s*\]", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def strip_wake_from_transcript(text: str) -> str:
    if not text:
        return ""
    t = re.sub(
        r"^\s*(hey\s*,?\s*)?jarvis\s*[,:]?\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return t


def load_models():
    wake = WakeWordModel(wakeword_model_paths=WAKE_MODEL_PATHS)
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    vad = webrtcvad.Vad(3)
    return wake, whisper, vad


def transcribe_audio(model: WhisperModel, pcm: np.ndarray) -> str | None:
    if pcm.size == 0:
        return None
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        import wave

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.astype(np.int16).tobytes())
        segments, _ = model.transcribe(path, language="en", beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text or None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def get_venice_response(user_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.7,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
        "venice_parameters": {
            "enable_web_search": "on",
            "enable_x_search": False,
            "enable_web_scraping": True,
            "enable_web_citations": False,
            "include_venice_system_prompt": False,
        },
        "parallel_tool_calls": False,
        "tools": [],
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(CHAT_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        out = strip_citations_and_links(raw)
        if len(out) < len(raw) * 0.85 and len(raw) > 80:
            logging.debug("Stripped citation noise: raw_chars=%d clean_chars=%d", len(raw), len(out))
        finish = choice.get("finish_reason")
        usage = body.get("usage") or {}
        comp = usage.get("completion_tokens")
        dt = time.monotonic() - t0
        with _stats_lock:
            _stats["chat_ok"] += 1
            n = _stats["chat_ok"]
        logging.info(
            "Venice chat #%d OK model=%s %.2fs prompt_chars=%d reply_chars=%d finish=%s usage=%s",
            n,
            CHAT_MODEL,
            dt,
            len(user_text),
            len(out),
            finish,
            usage,
        )
        if finish == "length":
            logging.warning(
                "Reply truncated at max_tokens=%s (raise JARVIS_MAX_TOKENS). completion_tokens=%s",
                MAX_COMPLETION_TOKENS,
                comp,
            )
        return out
    except Exception as e:
        dt = time.monotonic() - t0
        with _stats_lock:
            _stats["chat_err"] += 1
            n = _stats["chat_err"]
        logging.error("Venice chat err #%d after %.2fs: %s", n, dt, e)
        return "Sorry, I could not reach Venice right now."


def _play_audio_file(path: str) -> bool:
    """Try several players; ffplay/SDL often fails headless on Pi while aplay works."""
    errs: list[str] = []
    t_play = time.monotonic()

    ffplay = shutil.which("ffplay")
    if ffplay:
        sdl_env = _resolve_sdl_audio_env()
        drivers: list[str | None] = []
        if os.environ.get("SDL_AUDIODRIVER"):
            drivers.append(os.environ.get("SDL_AUDIODRIVER"))
        drivers.extend(["alsa", "pulse", None])
        seen = set()
        for driver in drivers:
            if driver in seen:
                continue
            seen.add(driver)
            env = os.environ.copy()
            env.update(sdl_env)
            if driver:
                env["SDL_AUDIODRIVER"] = driver
            else:
                env.pop("SDL_AUDIODRIVER", None)
            p = subprocess.run(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "warning",
                    "-i",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
            if p.returncode == 0:
                logging.info(
                    "Playback: ffplay OK (SDL_AUDIODRIVER=%s) %.2fs",
                    driver or "default",
                    time.monotonic() - t_play,
                )
                return True
            errs.append(f"ffplay/{driver or 'default'} rc={p.returncode}: {p.stderr.strip()[:200]}")

    mpg = shutil.which("mpg123")
    if mpg and path.lower().endswith(".mp3"):
        sdl_m = _resolve_sdl_audio_env()
        mpg_cmd = [mpg, "-q"]
        adev_m = (sdl_m.get("AUDIODEV") or "").strip()
        if adev_m:
            mpg_cmd.extend(["-a", adev_m])
        mpg_cmd.append(path)
        p = subprocess.run(mpg_cmd, capture_output=True, text=True, timeout=600)
        if p.returncode == 0:
            logging.info("Playback: mpg123 OK %.2fs", time.monotonic() - t_play)
            return True
        errs.append(f"mpg123 rc={p.returncode}: {p.stderr.strip()[:200]}")

    mpv = shutil.which("mpv")
    if mpv:
        sdl_v = _resolve_sdl_audio_env()
        mpv_cmd = [mpv, "--no-video", "--really-quiet"]
        adev_v = (sdl_v.get("AUDIODEV") or "").strip()
        if adev_v:
            mpv_cmd.append(f"--audio-device=alsa/{adev_v}")
        mpv_cmd.append(path)
        p = subprocess.run(
            mpv_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if p.returncode == 0:
            logging.info("Playback: mpv OK %.2fs", time.monotonic() - t_play)
            return True
        errs.append(f"mpv rc={p.returncode}")

    ffmpeg = shutil.which("ffmpeg")
    aplay = shutil.which("aplay")
    if ffmpeg and aplay:
        wav = path + ".jarvis_play.wav"
        try:
            r = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    wav,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                errs.append(f"ffmpeg decode rc={r.returncode}: {r.stderr.strip()[:200]}")
            else:
                sdl = _resolve_sdl_audio_env()
                aplay_cmd = [aplay, "-q"]
                adev = (sdl.get("AUDIODEV") or "").strip()
                if adev:
                    aplay_cmd.extend(["-D", adev])
                aplay_cmd.append(wav)
                p = subprocess.run(aplay_cmd, capture_output=True, text=True, timeout=600)
                if p.returncode == 0:
                    logging.info("Playback: ffmpeg+aplay OK %.2fs", time.monotonic() - t_play)
                    return True
                errs.append(f"aplay rc={p.returncode}: {p.stderr.strip()[:200]}")
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass

    pw = shutil.which("pw-play")
    if pw and path.lower().endswith(".wav"):
        p = subprocess.run([pw, path], capture_output=True, text=True, timeout=600)
        if p.returncode == 0:
            logging.info("Playback: pw-play OK %.2fs", time.monotonic() - t_play)
            return True
        errs.append(f"pw-play rc={p.returncode}")

    logging.error(
        "Playback failed (no working player or device). Tried: %s",
        " | ".join(errs) if errs else "(no ffplay/ffmpeg+aplay)",
    )
    return False


def _buffer_venice_stream_to_path(resp: requests.Response, path: str) -> int:
    """Write a streaming Venice TTS HTTP body to ``path``; return total bytes."""
    written = 0
    first_bytes_logged = False
    with open(path, "wb") as f:
        for piece in resp.iter_content(chunk_size=TTS_STREAM_CHUNK_BYTES):
            if not piece:
                continue
            if not first_bytes_logged:
                preview = piece[:16]
                try:
                    printable = preview.decode("utf-8", errors="replace")
                except Exception:
                    printable = ""
                logging.info(
                    "TTS stream first bytes: hex=%s ascii=%r",
                    preview.hex(),
                    printable,
                )
                first_bytes_logged = True
            f.write(piece)
            written += len(piece)
    return written


def _tts_download_chunk_to_path(
    chunk_text: str,
    path: str,
    *,
    base_payload: dict,
    headers: dict,
    chunk_idx: int,
    n_chunks: int,
    phase: str,
) -> int:
    """POST one TTS chunk and write audio bytes to ``path``. Returns byte count."""
    payload = dict(base_payload)
    payload["input"] = chunk_text
    logging.info(
        "Venice TTS chunk %d/%d: %s speech (format=%s, chars=%d)…",
        chunk_idx,
        n_chunks,
        phase,
        base_payload["response_format"],
        len(chunk_text),
    )
    if TTS_STREAMING:
        payload["streaming"] = True
        resp = requests.post(
            TTS_URL,
            json=payload,
            headers=headers,
            timeout=(15, TTS_HTTP_TIMEOUT),
            stream=True,
        )
        resp.raise_for_status()
        return _buffer_venice_stream_to_path(resp, path)
    resp = requests.post(
        TTS_URL,
        json=payload,
        headers=headers,
        timeout=(15, TTS_HTTP_TIMEOUT),
    )
    resp.raise_for_status()
    data = resp.content
    if len(data) < 80:
        raise ValueError("TTS response not audio")
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def speak(text: str) -> None:
    to_speak = text_for_tts(text)
    if not to_speak:
        logging.warning("TTS skipped — empty text after sanitizing")
        return
    to_speak, clipped = clip_text_for_tts(to_speak, TTS_MAX_INPUT_CHARS)
    if clipped:
        logging.warning(
            "TTS input clipped to %d chars (JARVIS_TTS_MAX_CHARS); say less in chat or raise cap.",
            TTS_MAX_INPUT_CHARS,
        )

    chunks: list[str]
    if TTS_CHUNK_CHARS > 0 and len(to_speak) > TTS_CHUNK_CHARS:
        chunks = split_for_tts(to_speak, TTS_CHUNK_CHARS)
        logging.warning(
            "TTS chunking enabled: input_chars=%d chunk_chars=%d -> %d chunks",
            len(to_speak),
            TTS_CHUNK_CHARS,
            len(chunks),
        )
    else:
        chunks = [to_speak]

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    fmt = TTS_RESPONSE_FORMAT if TTS_RESPONSE_FORMAT in ("mp3", "wav", "opus", "aac", "flac") else "mp3"
    ext = {"wav": "wav", "opus": "opus", "aac": "aac", "flac": "flac"}.get(fmt, "mp3")
    base_payload = {
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
        "response_format": fmt,
        "speed": TTS_SPEED,
        "temperature": 0.9,
        "top_p": 1.0,
    }
    if TTS_STREAMING and base_payload["response_format"] != "mp3":
        logging.warning(
            "TTS streaming enabled, but response_format=%s is not safe for FIFO playback; forcing mp3.",
            base_payload["response_format"],
        )
        base_payload["response_format"] = "mp3"
        fmt = "mp3"
        ext = "mp3"
    ffprobe = shutil.which("ffprobe")

    def probe_duration(path: str) -> str | None:
        if not TTS_PROBE_DURATION or not ffprobe:
            return None
        try:
            # duration in seconds
            r = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                s = r.stdout.strip()
                return s if s else None
        except Exception:
            return None
        return None

    if TTS_MERGE_ONCE and len(chunks) > 1:
        # Download all chunks concurrently, then merge to a single WAV and play once.
        # This avoids the audible pause between separate ffplay invocations.
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logging.warning("TTS merge-once requested, but ffmpeg not found; skipping.")
        else:
            tmpdir = tempfile.mkdtemp(prefix="jarvis_tts_merge_")
            try:
                audio_paths: list[str] = []
                with ThreadPoolExecutor(max_workers=TTS_MERGE_WORKERS) as ex:
                    futures = []
                    for i, chunk_text in enumerate(chunks, start=1):
                        out_path = os.path.join(tmpdir, f"chunk_{i}.{ext}")
                        audio_paths.append(out_path)
                        futures.append(
                            ex.submit(
                                _tts_download_chunk_to_path,
                                chunk_text,
                                out_path,
                                base_payload=base_payload,
                                headers=headers,
                                chunk_idx=i,
                                n_chunks=len(chunks),
                                phase="merge",
                            )
                        )
                    # Raise the first error if any chunk fails.
                    for fut in as_completed(futures):
                        fut.result()

                list_path = os.path.join(tmpdir, "list.txt")
                with open(list_path, "w", encoding="utf-8") as lf:
                    for p in audio_paths:
                        # ffmpeg concat demuxer: file 'path'
                        p_esc = p.replace("'", "'\\''")
                        lf.write(f"file '{p_esc}'\n")

                merged_path = os.path.join(tmpdir, "merged.wav")
                p = subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        list_path,
                        "-ac",
                        "1",
                        "-ar",
                        "48000",
                        merged_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=max(60, int(len(chunks) * 25)),
                )
                if p.returncode != 0 or not os.path.exists(merged_path):
                    raise RuntimeError(f"ffmpeg merge failed rc={p.returncode}: {p.stderr.strip()[:200]}")

                _play_audio_file(merged_path)
                return
            except Exception as e:
                logging.warning("TTS merge-once failed (%s); falling back to prefetch playback.", e)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # Run TTS chunks in order. Optional prefetch overlaps download of chunk N+1 with playback of N.
    n = len(chunks)
    prefetch_on = TTS_PREFETCH_NEXT and n > 1
    prefetch_exc: list[BaseException | None] = [None]
    prefetch_thread: threading.Thread | None = None
    future_path: str | None = None
    path_ready = ""
    written = 0

    def _start_prefetch(chunk_idx: int, chunk_text: str, out_path: str) -> threading.Thread:
        prefetch_exc[0] = None

        def _run() -> None:
            try:
                _tts_download_chunk_to_path(
                    chunk_text,
                    out_path,
                    base_payload=base_payload,
                    headers=headers,
                    chunk_idx=chunk_idx,
                    n_chunks=n,
                    phase="prefetch",
                )
            except BaseException as e:
                prefetch_exc[0] = e

        t = threading.Thread(
            target=_run,
            daemon=True,
            name=f"jarvis-tts-prefetch-{chunk_idx}",
        )
        t.start()
        return t

    if prefetch_on:
        fd0, path_ready = tempfile.mkstemp(suffix=f".{ext}")
        os.close(fd0)
        try:
            written = _tts_download_chunk_to_path(
                chunks[0],
                path_ready,
                base_payload=base_payload,
                headers=headers,
                chunk_idx=1,
                n_chunks=n,
                phase="requesting",
            )
        except BaseException:
            try:
                os.remove(path_ready)
            except OSError:
                pass
            raise

    for i in range(n):
        idx = i + 1
        chunk = chunks[i]
        if prefetch_on:
            if i > 0:
                if prefetch_thread is not None and future_path is not None:
                    prefetch_thread.join(timeout=float(TTS_HTTP_TIMEOUT) + 120.0)
                    if prefetch_thread.is_alive():
                        logging.error(
                            "TTS prefetch chunk %d/%d still running after timeout — will sync retry.",
                            idx,
                            n,
                        )
                        prefetch_exc[0] = TimeoutError("prefetch join timeout")
                    err = prefetch_exc[0]
                    prefetch_exc[0] = None
                    prefetch_thread = None
                    path_ready = future_path
                    future_path = None
                    if err is not None:
                        logging.warning(
                            "TTS prefetch failed for chunk %d/%d: %s — requesting synchronously.",
                            idx,
                            n,
                            err,
                        )
                        written = _tts_download_chunk_to_path(
                            chunk,
                            path_ready,
                            base_payload=base_payload,
                            headers=headers,
                            chunk_idx=idx,
                            n_chunks=n,
                            phase="requesting",
                        )
                    else:
                        written = os.path.getsize(path_ready)
                else:
                    fd_r, path_ready = tempfile.mkstemp(suffix=f".{ext}")
                    os.close(fd_r)
                    written = _tts_download_chunk_to_path(
                        chunk,
                        path_ready,
                        base_payload=base_payload,
                        headers=headers,
                        chunk_idx=idx,
                        n_chunks=n,
                        phase="requesting",
                    )
            path = path_ready
        else:
            fd, path = tempfile.mkstemp(suffix=f".{ext}")
            os.close(fd)
            written = _tts_download_chunk_to_path(
                chunk,
                path,
                base_payload=base_payload,
                headers=headers,
                chunk_idx=idx,
                n_chunks=n,
                phase="requesting",
            )

        t0 = time.monotonic()
        try:
            chunk_ok = False
            dur: str | None = None
            stream_play_ok = False
            if written < 80:
                logging.error(
                    "Venice TTS chunk %d: response too small (%d bytes), skipping playback.",
                    idx,
                    written,
                )
            else:
                if not TTS_STREAMING:
                    dur = probe_duration(path)
                stream_play_ok = _play_audio_file(path)
                chunk_ok = stream_play_ok
                if TTS_STREAMING and TTS_STREAMING_FALLBACK and written > 0 and (
                    not stream_play_ok
                    or (time.monotonic() - t0) <= TTS_STREAMING_FALLBACK_IF_DT_S
                ):
                    logging.warning(
                        "TTS streaming fallback for chunk %d: stream_play_ok=%s dt=%.2fs (written=%d). Retrying streaming=false.",
                        idx,
                        stream_play_ok,
                        time.monotonic() - t0,
                        written,
                    )
                    payload2 = dict(base_payload)
                    payload2["input"] = chunk
                    payload2["streaming"] = False
                    resp2 = requests.post(
                        TTS_URL,
                        json=payload2,
                        headers=headers,
                        timeout=(15, TTS_HTTP_TIMEOUT),
                    )
                    resp2.raise_for_status()
                    data2 = resp2.content
                    if len(data2) >= 80:
                        with open(path, "wb") as f:
                            f.write(data2)
                        logging.info(
                            "TTS fallback: non-streaming audio bytes=%d for chunk %d; playing…",
                            len(data2),
                            idx,
                        )
                        chunk_ok = _play_audio_file(path)
                        if not chunk_ok:
                            logging.error(
                                "TTS fallback playback failed for chunk %d after non-streaming fetch.",
                                idx,
                            )
                    else:
                        chunk_ok = False
                        logging.warning(
                            "TTS fallback: response too small (%d bytes) for chunk %d; not playing.",
                            len(data2),
                            idx,
                        )

            dt = time.monotonic() - t0
            with _stats_lock:
                if chunk_ok:
                    _stats["tts_ok"] += 1
                    stat_n = _stats["tts_ok"]
                else:
                    _stats["tts_err"] += 1
                    stat_n = _stats["tts_err"]
            if chunk_ok:
                if TTS_STREAMING:
                    logging.info(
                        "Venice TTS #%d chunk %d/%d OK voice=%s %.2fs bytes=%d (streaming path)…",
                        stat_n,
                        idx,
                        n,
                        TTS_VOICE,
                        dt,
                        written,
                    )
                else:
                    logging.info(
                        "Venice TTS #%d chunk %d/%d OK voice=%s %.2fs bytes=%d dur=%s (playback)…",
                        stat_n,
                        idx,
                        n,
                        TTS_VOICE,
                        dt,
                        written,
                        dur or "unknown",
                    )
            else:
                logging.error(
                    "Venice TTS err #%d chunk %d/%d voice=%s after %.2fs bytes=%d…",
                    stat_n,
                    idx,
                    n,
                    TTS_VOICE,
                    dt,
                    written,
                )

            if prefetch_on and i + 1 < n:
                fd_n, future_path = tempfile.mkstemp(suffix=f".{ext}")
                os.close(fd_n)
                prefetch_thread = _start_prefetch(i + 2, chunks[i + 1], future_path)
        except Exception as e:
            dt = time.monotonic() - t0
            with _stats_lock:
                _stats["tts_err"] += 1
                stat_n = _stats["tts_err"]
            logging.error(
                "Venice TTS err #%d chunk %d/%d after %.2fs: %s",
                stat_n,
                idx,
                n,
                dt,
                e,
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def finalize_cmd_audio(carry: np.ndarray, frames: list[np.ndarray]) -> np.ndarray:
    """Merge fixed-size VAD frames and any final partial buffer for Whisper."""
    parts = list(frames)
    if carry.size:
        parts.append(carry)
    if not parts:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(parts)


def run_command_pipeline(whisper: WhisperModel, pcm: np.ndarray) -> None:
    with _stats_lock:
        _stats["turns"] += 1
        turn = _stats["turns"]
    logging.info("Turn #%d start (pcm_samples=%d)", turn, pcm.size)
    text = transcribe_audio(whisper, pcm)
    if not text:
        logging.info("Turn #%d — no speech from Whisper", turn)
        return
    text = strip_wake_from_transcript(text)
    if not text:
        logging.info("Turn #%d — wake phrase only", turn)
        return
    logging.info('Turn #%d STT (%d chars): "%s"', turn, len(text), text)
    _monitor_begin_turn_after_stt(turn)
    reply = get_venice_response(text)
    # Log the full reply (what TTS speaks). Truncating here looked like model cutoffs.
    logging.info("Turn #%d reply (%d chars):", turn, len(reply))
    logging.info("%s", reply)
    # Start image generation in parallel with spoken playback so it never blocks
    # the voice response.
    _start_image_thread(turn, text, reply)
    speak(reply)
    logging.info("Turn #%d done | %s", turn, stats_summary())


def run_test_monitor() -> None:
    """Exercise DRM and desktop mpv outputs; writes per-attempt logs under /tmp. No Venice API key needed."""
    setup_logging()
    ffmpeg = shutil.which("ffmpeg")
    mpv = shutil.which("mpv")
    if not ffmpeg or not mpv:
        logging.error("Need ffmpeg and mpv on PATH (ffmpeg=%r mpv=%r).", ffmpeg, mpv)
        return

    try:
        ver = subprocess.run(
            [mpv, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        line0 = ((ver.stdout or "") + (ver.stderr or "")).split("\n")[0].strip()
        if line0:
            logging.info("mpv: %s", line0)
    except Exception as e:
        logging.info("mpv --version: %s", e)

    dri_dir = "/dev/dri"
    if os.path.isdir(dri_dir):
        for n in sorted(os.listdir(dri_dir)):
            if not n.startswith("card"):
                continue
            p = os.path.join(dri_dir, n)
            logging.info(
                "DRM node %s read=%s write=%s",
                p,
                os.access(p, os.R_OK),
                os.access(p, os.W_OK),
            )

    test_path = "/tmp/jarvis_drm_test_pattern.png"
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=#ff00ff:s=1920x1080",
                "-frames:v",
                "1",
                test_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.error("Could not create test pattern: %s", e)
        return

    discovered = _drm_connected_connectors()
    logging.info("DRM sysfs connected: %s", discovered or "(none)")
    specs = _mpv_drm_attempt_specs()
    logging.info(
        "Phase A: %d mpv --vo=drm attempt(s), ~5s each. Full mpv log per attempt in /tmp/jarvis_mpv_drm_*.log",
        len(specs),
    )
    abspath = os.path.abspath(test_path)
    for i, (dev, conn) in enumerate(specs, start=1):
        # Do not pass --no-terminal: mpv then emits no usable stderr/stdout for diagnosis.
        cmd = [
            mpv,
            "--no-audio",
            "--mute=yes",
            "--fullscreen",
            "--vo=drm",
            "--keep-open=no",
            "--image-display-duration=5",
            "-v",
            "--msg-level=all=v",
        ]
        if dev:
            cmd.append(f"--drm-device={dev}")
        if conn:
            cmd.append(f"--drm-connector={conn}")
        cmd.append(abspath)
        log_path = f"/tmp/jarvis_mpv_drm_{i}.log"
        logging.info(
            "DRM test %d/%d: device=%r connector=%r → log %s",
            i,
            len(specs),
            dev or "default",
            conn or "(unset)",
            log_path,
        )
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                    env=os.environ.copy(),
                )
        except subprocess.TimeoutExpired:
            logging.warning("mpv timed out for DRM combo %d (see %s)", i, log_path)
            continue
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                blob = f.read()
        except OSError:
            blob = ""
        logging.info("--- tail %s ---\n%s", log_path, (blob[-3500:] if blob else "(empty)"))
        if proc.returncode != 0:
            logging.warning(
                "mpv rc=%s for DRM device=%r connector=%r (full log: %s)",
                proc.returncode,
                dev,
                conn,
                log_path,
            )
        else:
            logging.info("mpv DRM finished OK device=%r connector=%r", dev, conn)

    has_desktop = any(
        (e.get("DISPLAY") or "").strip() or (e.get("WAYLAND_DISPLAY") or "").strip()
        for e in _mpv_desktop_env_candidates()
    ) or os.path.exists("/tmp/.X11-unix/X0")
    if has_desktop:
        logging.info(
            "Phase B: desktop compositor (~5s each). When a GUI owns DRM, vo=drm fails; "
            "mpv vo=gpu (X11 or Wayland) or vo=x11 may still paint HDMI."
        )
        envs = _mpv_desktop_env_candidates()
        di = 0
        for env in envs:
            for vo in ("gpu", "x11"):
                if vo == "x11" and not (env.get("DISPLAY") or "").strip():
                    continue
                di += 1
                cmd = [
                    mpv,
                    "--no-audio",
                    "--mute=yes",
                    "--fullscreen",
                    f"--vo={vo}",
                    "--keep-open=no",
                    "--image-display-duration=5",
                    "-v",
                    "--msg-level=all=v",
                    abspath,
                ]
                log_path = f"/tmp/jarvis_mpv_desktop_{di}.log"
                logging.info(
                    "Desktop test %d: vo=%s DISPLAY=%r WAYLAND=%r XAUTHORITY=%r → %s",
                    di,
                    vo,
                    env.get("DISPLAY"),
                    env.get("WAYLAND_DISPLAY"),
                    env.get("XAUTHORITY"),
                    log_path,
                )
                try:
                    with open(log_path, "w", encoding="utf-8") as logf:
                        proc = subprocess.run(
                            cmd,
                            stdout=logf,
                            stderr=subprocess.STDOUT,
                            timeout=30,
                            env=env,
                        )
                except subprocess.TimeoutExpired:
                    logging.warning("mpv desktop test %d timed out (%s)", di, log_path)
                    continue
                try:
                    with open(log_path, encoding="utf-8", errors="replace") as f:
                        blob = f.read()
                except OSError:
                    blob = ""
                logging.info("--- tail %s ---\n%s", log_path, (blob[-3500:] if blob else "(empty)"))
                if proc.returncode != 0:
                    logging.warning("mpv desktop rc=%s vo=%s (log %s)", proc.returncode, vo, log_path)
                else:
                    logging.info("mpv desktop OK vo=%s", vo)
    else:
        logging.info(
            "Skipping Phase B (no X11 socket, no DISPLAY/WAYLAND in environment). "
            "Install Raspberry Pi OS Desktop or export DISPLAY=:0 from a logged-in GUI session."
        )

    u = getpass.getuser()
    logging.info(
        "Next steps: open the /tmp/jarvis_mpv_*.log files above for the real mpv error line. "
        "Common fixes: (1) add user to groups video,render and re-login; "
        "(2) if a desktop is running, rely on Jarvis auto mode trying mpv gpu/x11 after drm; "
        "(3) from the Pi’s local console: xhost +SI:localuser:%s so SSH can use DISPLAY=:0; "
        "(4) if DRM worked, set JARVIS_IMAGE_DRM_DEVICE and JARVIS_IMAGE_DRM_CONNECTOR to the winning pair.",
        u,
    )


def main() -> None:
    setup_logging()
    if not VENICE_API_KEY:
        logging.error("Set VENICE_API_KEY (export VENICE_API_KEY=...).")
        return

    log_audio_devices()
    wake_model, whisper, vad = load_models()
    logging.info("Models loaded (Whisper %s). %s", WHISPER_MODEL_SIZE, stats_summary())
    logging.info(
        "Tips: pause ~%.1fs after wake before your question; max_tokens=%d; TTS model=%s speed=%.2f timeout=%ds "
        "(JARVIS_MAX_TOKENS / JARVIS_TTS_SPEED / JARVIS_TTS_TIMEOUT / JARVIS_TTS_MODEL).",
        POST_WAKE_HINT_S,
        MAX_COMPLETION_TOKENS,
        TTS_MODEL,
        TTS_SPEED,
        TTS_HTTP_TIMEOUT,
    )
    if IMAGE_ENABLED:
        auto_order = "mpv drm first" if _auto_try_drm_before_desktop() else "desktop mpv (gpu/x11) first"
        logging.info(
            "Monitor images: viewer=%s; auto tries %s, then fbi, feh. "
            "desktop_session_likely=%s JARVIS_IMAGE_TRY_DRM_FIRST=%d. "
            "No picture? python jarvis.py --test-monitor. Verbose: JARVIS_IMAGE_MPV_VERBOSE=1.",
            IMAGE_VIEWER,
            auto_order,
            _desktop_session_likely(),
            IMAGE_TRY_DRM_FIRST,
        )
        if IMAGE_PLACEHOLDER_ENABLED:
            logging.info(
                "Monitor loading: %s — starts right after STT (before Venice chat), loops until the image is ready. "
                "Override: JARVIS_IMAGE_PLACEHOLDER_PATH",
                IMAGE_PLACEHOLDER_PATH,
            )
        logging.info(
            "Image pipeline: prompt_agent=%s fast_image_prompt=%s web_search_on_prompt=%s "
            "(JARVIS_IMAGE_PROMPT_AGENT / JARVIS_IMAGE_FAST_IMAGE_PROMPT / JARVIS_IMAGE_PROMPT_WEB_SEARCH). "
            "Slow path (LLM before image): FAST=0 and AGENT=1. Logs: prompt s + /image/generate s.",
            IMAGE_PROMPT_AGENT_ENABLED,
            IMAGE_FAST_IMAGE_PROMPT,
            IMAGE_PROMPT_WEB_SEARCH,
        )
        if JARVIS_MONITOR_BLACK_UNDERLAY:
            logging.info(
                "Monitor: black fullscreen underlay mpv starts under Jarvis (hides wallpaper during mpv handoffs). "
                "Disable: JARVIS_MONITOR_BLACK_UNDERLAY=0"
            )
    # Resolve patience keys to whatever names ONNX basename produced
    patience, thresh = {}, {}
    for name in wake_model.models.keys():
        patience[name] = WAKE_PATIENCE.get(name, WAKE_PATIENCE.get("hey_jarvis_v0.1", 3))
        thresh[name] = WAKE_THRESHOLD_PATIENCE.get(
            name, WAKE_THRESHOLD_PATIENCE.get("hey_jarvis_v0.1", WAKE_THRESHOLD)
        )

    phase = Phase.LISTEN_WAKE
    pending = np.zeros(0, dtype=np.int16)
    cmd_carry = np.zeros(0, dtype=np.int16)
    cmd_frames: list[np.ndarray] = []
    skip_left = 0
    silence_start: float | None = None
    cmd_t0: float | None = None
    cooldown_until = 0.0
    interaction_lock = threading.Lock()

    if IMAGE_ENABLED:
        _start_monitor_black_underlay_mpv()
        _paint_x11_root_black_once()
    _show_openscreen_at_startup()

    logging.info(
        "Listening for wake word — say “Hey Jarvis”, then your prompt. "
        "One response at a time. Ctrl+C to stop."
    )

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=VAD_FRAME,
        ) as stream:
            while True:
                block, _ = stream.read(VAD_FRAME)
                block = np.asarray(block, dtype=np.int16).flatten()
                now = time.monotonic()

                if interaction_lock.locked():
                    continue

                if phase == Phase.LISTEN_WAKE:
                    if now < cooldown_until:
                        continue

                    pending = np.concatenate([pending, block])
                    while len(pending) >= OWW_CHUNK:
                        frame = pending[:OWW_CHUNK]
                        pending = pending[OWW_CHUNK:]
                        preds = wake_model.predict(
                            frame,
                            patience=patience,
                            threshold=thresh,
                        )
                        label, score = jarvis_score(preds)
                        if score >= WAKE_THRESHOLD:
                            logging.info(
                                "Wake detected (%s ≈ %.2f) — recording command…",
                                label,
                                score,
                            )
                            wake_model.reset()
                            phase = Phase.RECORD_CMD
                            pending = np.zeros(0, dtype=np.int16)
                            cmd_carry = np.zeros(0, dtype=np.int16)
                            cmd_frames.clear()
                            skip_left = int(POST_WAKE_SKIP_S * SAMPLE_RATE)
                            silence_start = None
                            cmd_t0 = time.monotonic()
                            break

                elif phase == Phase.RECORD_CMD:

                    def finish_turn():
                        nonlocal phase, cmd_carry, cmd_frames, cooldown_until
                        pcm = finalize_cmd_audio(cmd_carry, cmd_frames)
                        cmd_carry = np.zeros(0, dtype=np.int16)
                        cmd_frames = []
                        wake_model.reset()
                        phase = Phase.LISTEN_WAKE
                        cooldown_until = time.monotonic() + COOLDOWN_S

                        def work():
                            with interaction_lock:
                                run_command_pipeline(whisper, pcm)

                        threading.Thread(target=work, daemon=True).start()

                    b = block
                    if skip_left > 0:
                        drop = min(skip_left, len(b))
                        b = b[drop:]
                        skip_left -= drop

                    timed_out = cmd_t0 and (time.monotonic() - cmd_t0) > MAX_COMMAND_S

                    if b.size == 0:
                        if timed_out:
                            finish_turn()
                        continue

                    cmd_carry = np.concatenate([cmd_carry, b])
                    ended = False
                    while len(cmd_carry) >= VAD_FRAME:
                        frame = cmd_carry[:VAD_FRAME]
                        cmd_carry = cmd_carry[VAD_FRAME:]
                        cmd_frames.append(frame)
                        speech = vad.is_speech(
                            np.ascontiguousarray(frame).tobytes(),
                            SAMPLE_RATE,
                        )
                        if speech:
                            silence_start = None
                        else:
                            if silence_start is None:
                                silence_start = time.monotonic()
                            elif time.monotonic() - silence_start > SILENCE_END_S:
                                finish_turn()
                                ended = True
                                break
                    if ended:
                        continue

                    if cmd_t0 and (time.monotonic() - cmd_t0) > MAX_COMMAND_S:
                        finish_turn()

    except KeyboardInterrupt:
        if IMAGE_ENABLED:
            _stop_monitor_underlay()
        logging.info("Shutdown requested. %s", stats_summary())


if __name__ == "__main__":
    if "--test-monitor" in sys.argv:
        run_test_monitor()
    else:
        main()
