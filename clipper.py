import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import cv2

WHISPER_DIR = Path.home() / "whisper.cpp"
WHISPER_BIN = WHISPER_DIR / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = WHISPER_DIR / "models" / "ggml-base.en.bin"

LLAMA_DIR = Path.home() / "llama.cpp"
LLAMA_BIN = LLAMA_DIR / "build" / "bin" / "llama-cli"
LLAMA_MODEL = LLAMA_DIR / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

CASCADE_PATH = Path(__file__).parent / "haarcascade_frontalface_default.xml"
WORK_DIR = Path("workspace")
OUTPUT_DIR = Path("/storage/emulated/0/Download/PocketClipper")

HOOK_WORDS = {
    "secret", "mistake", "never", "always", "truth", "actually", "crazy",
    "insane", "money", "stop", "why", "how", "best", "worst", "first",
    "million", "hack", "trick", "important", "listen", "wait", "problem"
}


def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout


def download_video(url: str) -> Path:
    print("[1/5] Downloading YouTube video...")
    video_path = WORK_DIR / "source.mp4"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(video_path),
        url,
    ]
    run_cmd(cmd)
    return video_path


def extract_audio(video_path: Path) -> Path:
    print("[2/5] Extracting 16kHz mono audio for Whisper...")
    audio_path = WORK_DIR / "audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(audio_path)
    ]
    run_cmd(cmd)
    return audio_path


def transcribe_audio(audio_path: Path) -> Path:
    print("[3/5] Transcribing locally with whisper.cpp (this may take a few minutes)...")
    srt_prefix = WORK_DIR / "transcript"
    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        "-f", str(audio_path),
        "-osrt",
        "-of", str(srt_prefix),
        "-t", "4"
    ]
    run_cmd(cmd)
    return Path(f"{srt_prefix}.srt")


def parse_timestamp(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def format_timestamp(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def parse_srt(srt_path: Path):
    content = srt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            start_str, end_str = lines[1].split(" --> ")
            text = " ".join(lines[2:]).strip()
            segments.append({
                "start": parse_timestamp(start_str.strip()),
                "end": parse_timestamp(end_str.strip()),
                "text": text
            })
    return segments


def find_viral_clips_llm(segments, num_clips: int, target_duration: int):
    print("[4/5] Analyzing transcript with local LLM (Qwen 2.5 1.5B)...")

    transcript_lines = [f"[{int(s['start'])}s] {s['text']}" for s in segments]
    transcript_text = "\n".join(transcript_lines)[:10000]

    prompt = (
        f"You are an expert short-form video editor. Read this timestamped YouTube transcript "
        f"and pick the {num_clips} most engaging, self-contained moments for YouTube Shorts/Reels. "
        f"Each clip should be roughly {target_duration} seconds long.\n\n"
        f"Return ONLY a valid JSON array with no extra text, using this exact format:\n"
        f'[{{"start": 15, "end": 60, "reason": "Strong hook"}}]\n\n'
        f"Transcript:\n{transcript_text}\n\nJSON:"
    )

    cmd = [
        str(LLAMA_BIN),
        "-m", str(LLAMA_MODEL),
        "-p", prompt,
        "-n", "512",
        "-c", "4096",
        "-t", "4",
        "--temp", "0.2",
        "--no-display-prompt",
        "-no-cnv"
    ]

    try:
        output = run_cmd(cmd)
        match = re.search(r"\[.*\]", output, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in LLM output.")

        llm_clips = json.loads(match.group(0))
        selected = []

        for item in llm_clips[:num_clips]:
            start_t = float(item["start"])
            end_t = float(item["end"])
            if end_t <= start_t:
                continue

            clip_segs = [
                s for s in segments
                if s["end"] > start_t and s["start"] < end_t
            ]
            if not clip_segs:
                continue

            print(f"  -> Picked {int(start_t)}s-{int(end_t)}s: {item.get('reason', 'High engagement')}")
            selected.append({
                "start": start_t,
                "end": end_t,
                "segments": clip_segs
            })

        if selected:
            return selected

    except Exception as e:
        print(f"[WARN] LLM selection failed ({e}). Falling back to keyword scoring...")

    return find_viral_clips(segments, num_clips, target_duration)


def find_viral_clips(segments, num_clips: int, target_duration: int):
    print("[4/5] Scoring transcript for high-engagement moments (fallback)...")
    candidates = []

    for i in range(len(segments)):
        start_time = segments[i]["start"]
        window_text = []
        end_time = start_time

        for j in range(i, len(segments)):
            end_time = segments[j]["end"]
            window_text.append(segments[j]["text"])
            if end_time - start_time >= target_duration:
                break

        duration = end_time - start_time
        if duration < target_duration * 0.6:
            continue

        full_text = " ".join(window_text)
        words = re.findall(r"\b\w+\b", full_text.lower())
        if not words:
            continue

        pace_score = len(words) / max(duration, 1)
        hook_score = sum(2.5 for w in words if w in HOOK_WORDS)
        punct_score = full_text.count("?") * 3.0 + full_text.count("!") * 2.0
        total_score = pace_score + hook_score + punct_score

        candidates.append({
            "start": start_time,
            "end": end_time,
            "score": total_score,
            "segments": segments[i:j + 1]
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for cand in candidates:
        if len(selected) >= num_clips:
            break
        overlaps = any(
            not (cand["end"] <= s["start"] or cand["start"] >= s["end"])
            for s in selected
        )
        if not overlaps:
            selected.append(cand)

    return selected


def write_clip_srt(clip_segments, clip_start: float, out_srt: Path):
    lines = []
    for idx, seg in enumerate(clip_segments, start=1):
        rel_start = max(0.0, seg["start"] - clip_start)
        rel_end = max(rel_start + 0.2, seg["end"] - clip_start)
        lines.append(f"{idx}")
        lines.append(f"{format_timestamp(rel_start)} --> {format_timestamp(rel_end)}")
        lines.append(seg["text"])
        lines.append("")
    out_srt.write_text("\n".join(lines), encoding="utf-8")


def get_speaker_crop_x(video_path: Path, start_sec: float, duration: float) -> str:
    if not CASCADE_PATH.exists():
        return "(iw-ih*9/16)/2"

    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "(iw-ih*9/16)/2"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = int(height * 9 / 16)

    face_centers_x = []
    sample_count = 6

    for i in range(sample_count):
        ts_ms = (start_sec + (duration * (i + 0.5) / sample_count)) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_centers_x.append(x + w / 2)

    cap.release()

    if not face_centers_x:
        return "(iw-ih*9/16)/2"

    median_x = statistics.median(face_centers_x)
    crop_x = int(median_x - crop_w / 2)
    crop_x = max(0, min(crop_x, width - crop_w))
    return str(crop_x)


def render_clips(video_path: Path, clips):
    print(f"[5/5] Rendering {len(clips)} vertical 9:16 clips with face-tracking & captions...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, clip in enumerate(clips, start=1):
        clip_srt = WORK_DIR / f"clip_{idx}.srt"
        write_clip_srt(clip["segments"], clip["start"], clip_srt)

        out_file = OUTPUT_DIR / f"viral_clip_{idx}.mp4"
        duration = clip["end"] - clip["start"]
        crop_x = get_speaker_crop_x(video_path, clip["start"], duration)

        vf_filter = (
            f"crop=ih*9/16:ih:{crop_x}:0,scale=720:1280,"
            f"subtitles={clip_srt}:force_style="
            f"'FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,BorderStyle=1,Outline=2,Alignment=2,MarginV=40'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip["start"]),
            "-t", str(duration),
            "-i", str(video_path),
            "-vf", vf_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(out_file)
        ]
        run_cmd(cmd)
        print(f"  -> Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Free On-Device YouTube AI Clipper")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--clips", type=int, default=3, help="Number of clips to generate")
    parser.add_argument("--duration", type=int, default=45, help="Target clip length in seconds")
    args = parser.parse_args()

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    video_path = download_video(args.url)
    audio_path = extract_audio(video_path)
    srt_path = transcribe_audio(audio_path)
    segments = parse_srt(srt_path)

    if not segments:
        print("[ERROR] No speech detected in video.")
        sys.exit(1)

    clips = find_viral_clips_llm(segments, args.clips, args.duration)
    render_clips(video_path, clips)
    print("\nDone! Check your phone's Downloads/PocketClipper folder.")


if __name__ == "__main__":
    main()