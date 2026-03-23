# Whisper Jarvis

**A privacy-minded, self-hosted voice assistant for your desk, kitchen, or workshop.**  
Speak naturally, hear answers out loud, and see a matching **illustration on the monitor**—without tying your home to a single vendor’s closed ecosystem.

Whisper Jarvis is built for a **Raspberry Pi 5 (8 GB RAM)** and a typical “kiosk” setup: **microphone**, **speakers**, and **HDMI display**. The same stack can run on other Linux desktops with the right audio and display tooling.

---

## Why this exists

Consumer smart speakers are convenient, but they often mean:

- Opaque data handling and long retention  
- A fixed assistant personality and capability set  
- Hardware that exists mainly to lock you into one brand’s cloud  

**Whisper Jarvis** is an alternative: **wake word and speech-to-text stay on your machine**, while **reasoning, speech synthesis, and optional rich features** go through **[Venice AI](https://venice.ai)**—so you can pick from many models, tune behavior, and use Venice’s options (such as web context or X-related workflows when you enable them) **on your terms**, with an API key you control.

This project does not replace reading Venice’s own privacy policy and terms; it simply **reduces how much of the pipeline has to live inside someone else’s opaque appliance**.

---

## What it does

| Stage | Where it runs | What happens |
|--------|----------------|--------------|
| **Wake** | **Local** (on-device) | Listens for **“Hey Jarvis”** via [openWakeWord](https://github.com/dscripka/openWakeWord). |
| **Speech → text** | **Local** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base` by default) turns your question into text—audio is not sent to the cloud for transcription. |
| **Answer** | **Venice AI** | Your text is sent to Venice’s chat API; the assistant replies in plain, speakable language. |
| **Voice reply** | **Venice AI** | Text is synthesized with Venice **TTS** (e.g. Kokoro voices) and played through your speakers. |
| **Visual** | **Venice AI + local display** | An **image / infographic-style** frame is generated from the conversation and shown on the **monitor** (with a loading state while it renders). |

So: **hot mic handling and transcription are local**; **LLM, TTS, and image generation use Venice** with your key. You decide which Venice models and options to use.

---

## Hardware & software expectations

- **Device**: Tested around a **Raspberry Pi 5 with 8 GB RAM**; needs enough CPU/RAM for Whisper and wake-word models.  
- **Audio**: USB or built-in **microphone** + **speakers** (this tree is oriented toward **HDMI audio**; you can adjust for other outputs in code/env).  
- **Display**: **HDMI monitor** for the image pipeline (mpv, DRM, or X11/Wayland paths depending on your OS session).  
- **System tools**: `ffmpeg` / `ffplay` and often **mpv** are used for audio/video and monitor output; `aplay` or PipeWire players may be used depending on configuration.

---

## Quick start

### 1. Clone and enter the project

```bash
git clone https://github.com/hyper186/whisper-jarvis.git
cd whisper-jarvis
```

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install numpy requests sounddevice webrtcvad faster-whisper openwakeword
```

Install any system packages your distro needs for **PortAudio** (sounddevice), **ffmpeg**, and **mpv**.

### 3. API key

Copy the example env file and add your key:

```bash
cp .env.example .env
# Edit .env: set VENICE_API_KEY=...
```

Or export in the shell: `export VENICE_API_KEY='...'`

Never commit `.env` or real keys (they are listed in `.gitignore`).

### 4. Run

**Foreground** (logs to terminal + default `jarvis.log`):

```bash
./venv/bin/python -u jarvis.py
```

**Background** (uses `jarvis.sh`; loads `.env` automatically):

```bash
chmod +x jarvis.sh
./jarvis.sh start    # requires VENICE_API_KEY
./jarvis.sh status
./jarvis.sh stop
```

**Tip**: After the wake phrase, pause briefly (~0.7 s) before your question so the start of your sentence is not clipped.

---

## Customization

- **Chat model**: Default is set in `jarvis.py` as `CHAT_MODEL` (e.g. Venice’s catalog). Change it there to try other Venice models.  
- **TTS**: `JARVIS_TTS_MODEL`, `JARVIS_TTS_VOICE`, `JARVIS_TTS_SPEED`, `JARVIS_TTS_FORMAT`, etc.  
- **Images / monitor**: `JARVIS_IMAGE_*` variables control model, aspect ratio, viewer (`mpv`, DRM, `fbi`, …), and placeholder media.  
- **Venice “advanced” behavior**: The chat path can request Venice features such as **web context** (see `venice_parameters` in `get_venice_response` in `jarvis.py`). Tune these when you want richer answers—or stricter, lower-latency behavior.

Run **`python jarvis.py --test-monitor`** to debug HDMI/display routing without a full voice turn.

---

## Repository layout (high level)

| Path | Role |
|------|------|
| `jarvis.py` | Main assistant: audio loop, Whisper, Venice chat/TTS/image, display orchestration. |
| `jarvis.sh` | Start/stop/status wrapper, optional `.env` loading. |
| `.env.example` | Documents required `VENICE_API_KEY` (and optional `HF_TOKEN` for Hugging Face). |
| `Venice Loading Video.mp4` / `wj_openscreen.png` | Bundled loading / idle visuals for the monitor. |

---

## Privacy & security (practical summary)

- **Local**: Wake-word detection and **Whisper** transcription.  
- **Cloud (Venice)**: Chat completions, TTS audio, and image generation—only what you send after local STT.  
- **Secrets**: Use **`.env`** or the environment; do not commit API keys or logs (`jarvis.log` is gitignored).

For Venice-specific assurances (retention, regions, enterprise options), see **[Venice AI](https://venice.ai)** documentation.

---

## Name

**Whisper Jarvis** — local whisper, Jarvis-style voice interaction, Venice-powered brain and voice.

---

*Built for makers who want an AI-forward home or office without surrendering the whole stack to a single black box.*
