# Pocket Clipper 📱✂️

A 100% free, offline-capable YouTube AI Clipper that runs entirely on your Android phone via **Termux**.

## Features
- **Zero Cost:** No OpenAI, OpusClip, or cloud GPU fees.
- **On-Device AI:** Uses `whisper.cpp` for transcription and `llama.cpp` (Qwen 2.5 1.5B) for viral clip selection.
- **Auto Face-Tracking:** Uses OpenCV to keep the 9:16 vertical crop centered on the speaker.
- **1-Tap Share:** Share any YouTube link directly to Termux to auto-clip.

## Installation
Open Termux (from F-Droid) and run:

```bash
pkg update -y && pkg upgrade -y && pkg install git -y
git clone [https://github.com/YOUR_USERNAME/pocket-clipper.git](https://github.com/YOUR_USERNAME/pocket-clipper.git)
cd pocket-clipper
chmod +x setup.sh
./setup.sh
