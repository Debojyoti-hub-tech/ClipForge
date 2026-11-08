#!/usr/bin/env python3
"""Pro edit mode: YouTube link -> edited, viral-style Shorts.

Per clip: reframe to 9:16 -> cut silence/fillers -> Gemini edit plan (5 hooks, keywords,
zoom targets) -> hook line + animated captions + punch-in zooms + SFX + music ducking
+ loudness normalisation.

Usage (PowerShell):
    $env:GEMINI_API_KEY = "your_key"
    python reel_editor.py "https://youtu.be/VIDEO_ID" -n 3 --music "C:\\path\\to\\music_folder"

Needs shorts_clipper.py in the same folder. Only use on videos you own or may reuse.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from collections import namedtuple
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

from shorts_clipper import DEFAULT_MODEL, analyze, download, ffmpeg, parse_ts, reframe, slug

Word = namedtuple("Word", "text start end")
FPS = 30
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "hmm", "mm", "mhm"}

# ASS colours are &HBBGGRR&
YELLOW, RED = "&H00E6FF&", "&H3B3BFF&"


# ----------------------------------------------------------------------------- models
class Zoom(BaseModel):
    start: float
    end: float
    x: float          # 0-1, centre of the thing to focus on
    y: float
    scale: float      # 1.2-1.5
    reason: str


class EditPlan(BaseModel):
    hooks: list[str]
    best_hook: int
    keywords: list[str]
    zooms: list[Zoom]


PLAN_PROMPT = """You are a top-tier viral short-form editor. This clip is {dur:.1f} seconds long, \
already cut tight. Word-timed transcript (seconds:word):
{transcript}

Return an edit plan.

HOOKS: write exactly 5 hooks for the first {hs:.0f} seconds, shown as bold text at the top of the screen.
- Max 8 words each. Shocking, emotional, urgent, controversial or surprising. Pattern-interrupting.
- Create a curiosity gap that THIS CLIP actually pays off. The hook must be true to what is said \
or shown. Viewers who feel baited swipe at second 4 and kill retention.
- Never generic. Banned: "wait for it", "you won't believe", "this changes everything", \
"watch till the end", "secret", "hack".
- Use concrete specifics from the clip (numbers, names, stakes, a bold claim).
- best_hook = index (0-4) of the one most likely to stop a scroll.

KEYWORDS: 6-15 single words, lowercase exactly as spoken, that carry the most meaning, \
emotion, numbers or surprise. They get colour emphasis in the captions.

ZOOMS: 3-7 punch-in zooms (roughly one per 8 seconds, fewer for short clips). Each lands on a \
moment that matters: a key claim, an emotional beat, a surprising detail, on-screen text.
- start/end in seconds of this clip, 1.2-3.0s long, no overlap, at least 1s apart, none before 1.0s.
- scale 1.2-1.5.
- x,y = normalised (0-1) centre of the exact thing viewers should look at (speaker's face, an \
object, on-screen text). LOOK at the video and place it precisely.
- reason: a few words."""


# ----------------------------------------------------------------------------- helpers
def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9']", "", t.lower())


def make_client(api_key):
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))


def probe_duration(path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "csv=p=0", str(path)])
    return float(out.decode().strip())


_whisper = None


def transcribe(path, size) -> list[Word]:
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(size, device="cpu", compute_type="int8")
    # the prompt nudges Whisper into transcribing "um/uh" instead of silently dropping them
    segs, _ = _whisper.transcribe(
        str(path), word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
        initial_prompt="Umm, uh, so, like, you know, I mean, hmm. Okay, so, uh, here's the thing.")
    return [Word(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]


# ----------------------------------------------------------------------------- silence / filler removal
def keep_segments(words, dur, gap=0.30, pad_in=0.05, pad_out=0.09, min_len=0.3):
    segs, cur, cut = [], None, False
    for w in words:
        if norm(w.text) in FILLERS:
            cut = True
            continue
        if cur is None:
            cur = [w.start, w.end]
        elif cut or w.start - cur[1] > gap:
            segs.append(cur)
            cur = [w.start, w.end]
        else:
            cur[1] = w.end
        cut = False
    if cur:
        segs.append(cur)
    if not segs:
        return [(0.0, dur)]
    out = []
    for i, (s, e) in enumerate(segs):
        lo = out[-1][1] if out else 0.0
        s2, e2 = max(lo, s - pad_in), min(dur, e + pad_out)
        if i + 1 < len(segs):
            e2 = min(e2, (e + segs[i + 1][0]) / 2)
        if e2 - s2 >= min_len:
            out.append((s2, e2))
    return out or [(0.0, dur)]


def tighten(src: Path, segs, dst: Path):
    """Concatenate kept segments. Every other segment gets a 7% punch-in, which hides the
    jump cuts and gives a visual change at every cut."""
    parts = []
    for i, (s, e) in enumerate(segs):
        punch = ""
        if i % 2 == 1:
            punch = (",crop=w=trunc(iw/1.07/2)*2:h=trunc(ih/1.07/2)*2:x=(iw-ow)/2:y=(ih-oh)*0.3,"
                     "scale=1080:1920")
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS{punch},setsar=1,fps={FPS}[v{i}]")
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.012,afade=t=out:st={max(e - s - 0.02, 0):.3f}:d=0.02[a{i}]")
    concat = "".join(f"[v{i}][a{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=1[v][a]"
    ffmpeg("-i", str(src), "-filter_complex", ";".join(parts + [concat]), "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", str(dst))


# ----------------------------------------------------------------------------- Gemini edit plan
def plan_edit(client, model, clip: Path, words, dur, hook_seconds, log):
    f = None
    try:
        f = client.files.upload(file=str(clip))
        while f.state is not None and f.state.name == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        transcript = " ".join(f"{w.start:.1f}:{w.text}" for w in words)
        resp = client.models.generate_content(
            model=model,
            contents=[f, PLAN_PROMPT.format(dur=dur, transcript=transcript, hs=hook_seconds)],
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=EditPlan, temperature=0.8))
        return resp.parsed or EditPlan(**json.loads(resp.text))
    except Exception as e:  # noqa: BLE001
        log(f"  Gemini edit plan failed ({e}); using fallback hook and no zooms")
        return None
    finally:
        if f is not None:
            try:
                client.files.delete(name=f.name)
            except Exception:  # noqa: BLE001
                pass


def clean_plan(plan, dur, words, fallback_hook):
    hooks = [h.strip() for h in (plan.hooks if plan else []) if h and h.strip()]
    if plan and 0 <= plan.best_hook < len(plan.hooks) and plan.hooks[plan.best_hook].strip():
        hook = plan.hooks[plan.best_hook].strip()
    else:
        hook = hooks[0] if hooks else (fallback_hook or "")
    if hook and hook not in hooks:
        hooks.insert(0, hook)

    kws = {norm(k) for k in (plan.keywords if plan else []) if norm(k)}
    if not kws:  # fallback: numbers and long words
        kws = {norm(w.text) for w in words if re.search(r"\d", w.text) or len(norm(w.text)) >= 8}

    zooms, last_end = [], 0.0
    for z in sorted(plan.zooms if plan else [], key=lambda z: z.start):
        s, e = max(z.start, 1.0), min(z.end, dur - 0.2)
        if e - s < 0.8 or s < last_end + 0.8:
            continue
        e = min(e, s + 3.0)
        zooms.append(Zoom(start=s, end=e, x=min(max(z.x, 0), 1), y=min(max(z.y, 0), 1),
                          scale=min(max(z.scale, 1.15), 1.5), reason=z.reason))
        last_end = e
    return hooks, hook, kws, zooms


# ----------------------------------------------------------------------------- captions + hook (ASS)
ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,8,4,2,60,60,560,1
Style: Hook,Arial Black,70,&H00000000,&H00000000,&H0000E6FF,&H00000000,0,0,0,0,100,100,0,0,3,22,0,8,70,70,230,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ts(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def esc(t: str) -> str:
    return re.sub(r"[{}\\]", "", t)


def build_ass(words, keywords, hook, hook_end, dur) -> str:
    lines = [ASS_HEAD]
    if hook:
        text = "\\N".join(textwrap.wrap(esc(hook).upper(), width=17))
        lines.append(f"Dialogue: 1,{ts(0)},{ts(hook_end)},Hook,,0,0,0,,"
                     f"{{\\fad(0,250)\\fscx55\\fscy55\\t(0,170,\\fscx106\\fscy106)"
                     f"\\t(170,300,\\fscx100\\fscy100)}}{text}")

    groups, g = [], []
    for w in words:
        t = esc(w.text)
        if not norm(t):
            continue
        if g and (len(g) >= 3 or len(" ".join(x.text for x in g) + " " + t) > 15
                  or w.start - g[-1].end > 0.45 or g[-1].text[-1:] in ".?!,"):
            groups.append(g)
            g = []
        g.append(Word(t, w.start, w.end))
    if g:
        groups.append(g)

    for gi, g in enumerate(groups):
        nxt = groups[gi + 1][0].start if gi + 1 < len(groups) else dur
        for j, w in enumerate(g):
            end = g[j + 1].start if j + 1 < len(g) else min(w.end + 0.08, nxt)
            end = max(end, w.start + 0.06)
            toks = []
            for k, x in enumerate(g):
                T, kw = x.text.upper(), norm(x.text) in keywords
                if k == j and kw:
                    toks.append(f"{{\\c{RED}\\fscx128\\fscy128\\t(0,100,\\fscx112\\fscy112)}}{T}{{\\r}}")
                elif k == j:
                    toks.append(f"{{\\c{YELLOW}\\fscx118\\fscy118\\t(0,100,\\fscx100\\fscy100)}}{T}{{\\r}}")
                elif kw:
                    toks.append(f"{{\\c{RED}\\fscx112\\fscy112}}{T}{{\\r}}")
                else:
                    toks.append(T)
            lines.append(f"Dialogue: 0,{ts(w.start)},{ts(end)},Cap,,0,0,0,,{' '.join(toks)}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- zoom + render
def smooth(a, b, ramp=0.25):
    u = f"clip(min((on/{FPS}-{a:.3f})/{ramp:.3f},({b:.3f}-on/{FPS})/{ramp:.3f}),0,1)"
    return f"({u}*{u}*(3-2*{u}))"


def zoom_filter(zooms):
    ws = [smooth(z.start, z.end) for z in zooms]
    z_expr = "1+" + "+".join(f"({z.scale - 1:.3f})*{w}" for z, w in zip(zooms, ws))
    cx = "0.5+" + "+".join(f"({z.x - 0.5:.3f})*{w}" for z, w in zip(zooms, ws))
    cy = "0.5+" + "+".join(f"({z.y - 0.5:.3f})*{w}" for z, w in zip(zooms, ws))
    x = f"clip(({cx})*iw-iw/zoom/2,0,iw-iw/zoom)"
    y = f"clip(({cy})*ih-ih/zoom/2,0,ih-ih/zoom)"
    return f"zoompan=z='{z_expr}':x='{x}':y='{y}':d=1:s=1080x1920:fps={FPS}"


def make_sfx(workdir: Path):
    ffmpeg("-f", "lavfi", "-i", "anoisesrc=color=pink:duration=0.5:amplitude=0.5", "-af",
           "highpass=f=400,lowpass=f=8000,afade=t=in:d=0.22,afade=t=out:st=0.22:d=0.28,volume=3",
           "whoosh.wav", cwd=workdir)
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=50:duration=0.7", "-af",
           "afade=t=out:st=0:d=0.7,volume=4", "hit.wav", cwd=workdir)


def pick_music(path):
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        files = [f for f in p.iterdir() if f.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}]
        return random.choice(files) if files else None
    return p if p.exists() else None


def render(clip: Path, dst: Path, ass_name: str, zooms, music, dur):
    work = dst.parent
    make_sfx(work)
    inputs, idx = ["-i", clip.name], 1
    music_i = whoosh_i = None
    if music:
        inputs += ["-stream_loop", "-1", "-i", str(Path(music).resolve())]
        music_i, idx = idx, idx + 1
    if zooms:
        inputs += ["-i", "whoosh.wav"]
        whoosh_i, idx = idx, idx + 1
    inputs += ["-i", "hit.wav"]
    hit_i = idx

    v = (f"[0:v]fps={FPS},eq=contrast=1.05:saturation=1.12:gamma=0.98,unsharp=5:5:0.5:3:3:0.0")
    if zooms:
        v += ",scale=1620:2880:flags=lanczos," + zoom_filter(zooms)
    v += f",format=yuv420p,ass={ass_name}[v]"

    st = "aresample=44100,aformat=channel_layouts=stereo"
    a, mix = [], ["[vo]"]
    if music:
        a.append(f"[0:a]{st},asplit=2[vo][vsc]")
        a.append(f"[{music_i}:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS,{st},loudnorm=I=-27:TP=-3:LRA=7,"
                 f"afade=t=in:d=0.6,afade=t=out:st={max(dur - 1.2, 0):.2f}:d=1.2[mus0]")
        a.append("[mus0][vsc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=350[mus]")
        mix.append("[mus]")
    else:
        a.append(f"[0:a]{st}[vo]")
    if zooms:
        outs = "".join(f"[w{i}]" for i in range(len(zooms)))
        a.append(f"[{whoosh_i}:a]{st},volume=0.55,asplit={len(zooms)}{outs}")
        for i, z in enumerate(zooms):
            ms = int(max(z.start - 0.12, 0) * 1000)
            a.append(f"[w{i}]adelay={ms}|{ms}[wd{i}]")
            mix.append(f"[wd{i}]")
    a.append(f"[{hit_i}:a]{st},volume=0.9,adelay=40|40[hd]")
    mix.append("[hd]")
    a.append("".join(mix) + f"amix=inputs={len(mix)}:duration=first:normalize=0:dropout_transition=0,"
             "loudnorm=I=-14:TP=-1.5:LRA=11[a]")

    ffmpeg(*inputs, "-filter_complex", ";".join([v] + a), "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.2f}", "-movflags", "+faststart",
           dst.name, cwd=work)


# ----------------------------------------------------------------------------- one clip, end to end
def edit_clip(raw: Path, final: Path, client, model, whisper_size, music, hook_seconds,
              fallback_hook, log=print):
    work = final.parent
    dur0 = probe_duration(raw)
    log("  transcribing (pass 1) ...")
    w0 = transcribe(raw, whisper_size)
    gap = 0.30
    segs = keep_segments(w0, dur0, gap=gap)
    while len(segs) > 70:  # keep the ffmpeg command under Windows' length limit
        gap += 0.1
        segs = keep_segments(w0, dur0, gap=gap)

    tight = work / f"{final.stem}_tight.mp4"
    tighten(raw, segs, tight)
    dur = probe_duration(tight)
    log(f"  tightened {dur0:.1f}s -> {dur:.1f}s ({len(segs)} cuts)")

    log("  transcribing (pass 2) ...")
    words = transcribe(tight, whisper_size)
    log("  asking Gemini for hooks, keywords and zooms ...")
    plan = plan_edit(client, model, tight, words, dur, hook_seconds, log)
    hooks, hook, keywords, zooms = clean_plan(plan, dur, words, fallback_hook)
    log(f"  hook: {hook}")

    ass = work / f"{final.stem}.ass"
    ass.write_text(build_ass(words, keywords, hook, min(hook_seconds, dur * 0.6), dur), encoding="utf-8")
    log(f"  rendering ({len(zooms)} zooms) ...")
    render(tight, final, ass.name, zooms, pick_music(music), dur)

    for tmp in (tight, ass, work / "whoosh.wav", work / "hit.wav"):
        tmp.unlink(missing_ok=True)
    return {"hook": hook, "hook_options": hooks, "keywords": sorted(keywords),
            "zooms": [z.model_dump() for z in zooms], "duration": round(dur, 1)}


def pipeline(url, api_key, out="reels_out", model=DEFAULT_MODEL, n=3, lo=25, hi=60, fit="crop",
             music=None, whisper_size="base", hook_seconds=4.0, keep_source=False, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Asking {model} to pick moments ...")
    clips = analyze(url, api_key, model, n, lo, hi)
    log(f"Got {len(clips)} candidate clips. Downloading ...")
    src, total = download(url, out)
    client = make_client(api_key)

    results = []
    for i, c in enumerate(clips, 1):
        start, end = parse_ts(c.start), parse_ts(c.end)
        if total:
            end = min(end, total)
        if end - start > hi + 5:
            end = start + hi
        if end - start < 5:
            log(f"Clip {i}: bad timestamps, skipped")
            continue
        name = f"{i:02d}_{slug(c.title)}"
        raw, final = out / f"{name}_raw.mp4", out / f"{name}.mp4"
        log(f"Clip {i}: {c.start}-{c.end} '{c.title}'")
        try:
            reframe(src, start, end - start, raw, fit)
            info = edit_clip(raw, final, client, model, whisper_size, music, hook_seconds, c.hook, log)
        except Exception as e:  # noqa: BLE001
            log(f"  Clip {i} failed: {e}")
            continue
        finally:
            raw.unlink(missing_ok=True)
        (out / f"{name}_hooks.txt").write_text("\n".join(f"{k + 1}. {h}" for k, h in
                                                          enumerate(info["hook_options"])), encoding="utf-8")
        results.append({"title": c.title, "hashtags": c.hashtags, "file": str(final),
                        "source_range": f"{c.start}-{c.end}", **info})

    (out / "reels.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if not keep_source:
        src.unlink(missing_ok=True)
    log(f"Done. {len(results)} reels in {out}/")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--min", type=int, default=25, dest="lo")
    ap.add_argument("--max", type=int, default=60, dest="hi")
    ap.add_argument("--out", default="reels_out")
    ap.add_argument("--fit", choices=["crop", "blur"], default="crop")
    ap.add_argument("--music", help="music file or folder (royalty-free only)")
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--hook-seconds", type=float, default=4.0)
    ap.add_argument("--keep-source", action="store_true")
    a = ap.parse_args()
    if not a.api_key:
        sys.exit("Set GEMINI_API_KEY or pass --api-key")
    pipeline(a.url, a.api_key, a.out, a.model, a.n, a.lo, a.hi, a.fit, a.music, a.whisper,
             a.hook_seconds, a.keep_source)


if __name__ == "__main__":
    main()
