#!/usr/bin/env python3
"""YouTube link -> vertical Shorts clips. Gemini picks the moments, ffmpeg cuts them.

Usage:
    export GEMINI_API_KEY=your_key
    python shorts_clipper.py "https://youtu.be/VIDEO_ID" -n 5 --captions

Only use on videos you own or have permission to reuse.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yt_dlp
from google import genai
from google.genai import types
from pydantic import BaseModel

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class Clip(BaseModel):
    start: str          # "MM:SS" or "H:MM:SS"
    end: str
    title: str
    hook: str           # on-screen text for the first 2 seconds
    reason: str
    hashtags: list[str]
    score: int          # 1-10 viral potential


PROMPT = """You are a short-form video editor. Watch this video and pick the {n} best \
moments to cut as YouTube Shorts.

Rules:
- Each clip is {lo}-{hi} seconds long.
- Start at the beginning of a sentence with a strong hook in the first 2 seconds. \
End on a completed thought, never mid-sentence.
- Each clip must make sense with zero context from the rest of the video.
- Clips must not overlap.
- Timestamps must be exact positions in the video, formatted MM:SS (or H:MM:SS).
- hook: punchy on-screen text, under 8 words.
- title: under 60 characters.
- hashtags: 3-5, no spaces.
- score: 1-10 for viral potential, be harsh.
- If fewer than {n} moments are genuinely good, return fewer. Do not pad.
Sort by score, highest first."""


def video_id(url: str) -> str:
    m = re.search(r"(?:youtu\.be/|v=|shorts/|embed/)([\w-]{11})", url)
    if not m:
        sys.exit(f"Could not find a YouTube video ID in: {url}")
    return m.group(1)


def parse_ts(ts: str) -> float:
    parts = [float(p) for p in str(ts).strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def analyze(url, api_key, model, n, lo, hi) -> list[Clip]:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )
    clean_url = f"https://www.youtube.com/watch?v={video_id(url)}"
    resp = client.models.generate_content(
        model=model,
        contents=types.Content(parts=[
            types.Part(file_data=types.FileData(file_uri=clean_url)),
            types.Part(text=PROMPT.format(n=n, lo=lo, hi=hi)),
        ]),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Clip],
            temperature=0.3,
        ),
    )
    clips = resp.parsed
    if clips is None:  # fallback if SDK didn't parse
        clips = [Clip(**c) for c in json.loads(resp.text)]
    return clips


def download(url, outdir: Path):
    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(outdir / "source.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "js_runtimes": {"node": {}},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return outdir / "source.mp4", info.get("duration")


def ffmpeg(*args, cwd=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True, cwd=cwd)


def reframe(src: Path, start: float, dur: float, dst: Path, fit: str):
    enc = ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]
    if fit == "blur":
        fc = ("[0:v]split=2[a][b];"
              "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
              "[b]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
              "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
        ffmpeg("-ss", f"{start:.2f}", "-i", str(src), "-t", f"{dur:.2f}",
               "-filter_complex", fc, "-map", "[v]", "-map", "0:a?", *enc, str(dst))
    else:  # center crop
        vf = "crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)',scale=1080:1920,setsar=1"
        ffmpeg("-ss", f"{start:.2f}", "-i", str(src), "-t", f"{dur:.2f}",
               "-vf", vf, *enc, str(dst))


_whisper = None


def make_srt(clip: Path, srt: Path, size: str):
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(size, device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(str(clip), word_timestamps=True, vad_filter=True)
    words = [w for s in segments for w in s.words]

    def fmt(t):
        ms = int(t * 1000)
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i in range(0, len(words), 3):  # 3 words per caption = punchy Shorts style
        chunk = words[i:i + 3]
        text = " ".join(w.word.strip() for w in chunk).upper()
        lines.append(f"{len(lines) + 1}\n{fmt(chunk[0].start)} --> {fmt(chunk[-1].end)}\n{text}\n")
    srt.write_text("\n".join(lines), encoding="utf-8")
    return bool(lines)


def burn_captions(raw: Path, srt: Path, dst: Path):
    style = "FontName=Arial,FontSize=18,Bold=1,Outline=2,Alignment=2,MarginV=70"
    ffmpeg("-i", raw.name, "-vf", f"subtitles={srt.name}:force_style='{style}'",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", dst.name, cwd=raw.parent)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"


def run(url, api_key, out="shorts_out", model=DEFAULT_MODEL, n=5, lo=20, hi=59,
        fit="crop", captions=False, whisper_size="base", keep_source=False, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    log(f"Asking {model} to pick moments (this watches the whole video)...")
    clips = analyze(url, api_key, model, n, lo, hi)
    log(f"Gemini suggested {len(clips)} clips. Downloading source...")
    src, total = download(url, out)

    results = []
    for i, c in enumerate(clips, 1):
        start, end = parse_ts(c.start), parse_ts(c.end)
        if total:
            end = min(end, total)
        if end - start > hi + 5:
            log(f"Clip {i}: too long ({end - start:.0f}s), trimming to {hi}s")
            end = start + hi
        if end - start < 5 or (total and start >= total):
            log(f"Clip {i}: bad timestamps {c.start}-{c.end}, skipped")
            continue

        name = f"{i:02d}_{slug(c.title)}"
        raw, final = out / f"{name}_raw.mp4", out / f"{name}.mp4"
        log(f"Clip {i}: {c.start}-{c.end} '{c.title}'")
        reframe(src, start, end - start, raw, fit)

        if captions:
            srt = out / f"{name}.srt"
            if make_srt(raw, srt, whisper_size):
                burn_captions(raw, srt, final)
                raw.unlink()
            else:
                raw.rename(final)
            srt.unlink(missing_ok=True)
        else:
            raw.rename(final)

        results.append({**c.model_dump(), "file": str(final),
                        "start_s": round(start, 1), "end_s": round(end, 1)})

    (out / "clips.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if not keep_source:
        src.unlink(missing_ok=True)
    log(f"Done. {len(results)} clips in {out}/")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-n", type=int, default=5, help="max clips")
    ap.add_argument("--min", type=int, default=20, dest="lo")
    ap.add_argument("--max", type=int, default=59, dest="hi")
    ap.add_argument("--out", default="shorts_out")
    ap.add_argument("--fit", choices=["crop", "blur"], default="crop",
                    help="crop = center crop to 9:16, blur = full frame over blurred background")
    ap.add_argument("--captions", action="store_true", help="burn in Whisper captions")
    ap.add_argument("--whisper", default="base", help="whisper model size")
    ap.add_argument("--keep-source", action="store_true")
    a = ap.parse_args()
    if not a.api_key:
        sys.exit("Set GEMINI_API_KEY or pass --api-key")
    run(a.url, a.api_key, a.out, a.model, a.n, a.lo, a.hi, a.fit,
        a.captions, a.whisper, a.keep_source)


if __name__ == "__main__":
    main()
