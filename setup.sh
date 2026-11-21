#!/data/data/com.termux/files/usr/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 1. Granting Storage Access ==="
termux-setup-storage

echo "=== 2. Installing Termux Packages & OpenCV ==="
pkg update -y && pkg upgrade -y
pkg install -y python ffmpeg git cmake clang make curl python-numpy opencv

echo "=== 3. Installing yt-dlp ==="
pip install -U "yt-dlp[default]"

echo "=== 4. Cloning & Building whisper.cpp ==="
if [ ! -d "$HOME/whisper.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git "$HOME/whisper.cpp"
fi

cd "$HOME/whisper.cpp"
cmake -S . -B build -DGGML_NO_OPENMP=ON
cmake --build build -j"$(nproc)"

echo "=== 5. Downloading Whisper Model (base.en ~142MB) ==="
bash ./models/download-ggml-model.sh base.en

echo "=== 6. Cloning & Building llama.cpp ==="
if [ ! -d "$HOME/llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$HOME/llama.cpp"
fi

cd "$HOME/llama.cpp"
cmake -S . -B build -DGGML_NO_OPENMP=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-cli -j"$(nproc)"

echo "=== 7. Downloading Qwen 2.5 1.5B Model (~1.1GB) ==="
mkdir -p "$HOME/llama.cpp/models"
curl -L -o "$HOME/llama.cpp/models/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

echo "=== 8. Downloading Face Detection Model ==="
curl -L -o "$REPO_DIR/haarcascade_frontalface_default.xml" \
  "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

echo "=== 9. Installing Termux Share-Menu Hook ==="
mkdir -p "$HOME/bin"
cp "$REPO_DIR/termux-url-opener" "$HOME/bin/termux-url-opener"
chmod +x "$HOME/bin/termux-url-opener"

echo "=== Setup Complete! ==="
echo "Share any YouTube link to Termux, or run: python clipper.py <YOUTUBE_URL>"