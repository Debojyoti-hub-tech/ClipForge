#!/usr/bin/env python3
"""reel_pro: premium edit mode (builds on reel_editor.py and shorts_clipper.py).

New vs reel_editor.py:
  * Full-width hook banner at the top for the WHOLE reel, colours picked from the video's palette
  * Intro (white flash + blur-to-sharp + punch-out + hit) and outro (blur + push-in + fade + CTA)
  * Hinglish transcription (Roman script) with a Gemini listening pass that fixes mis-heard words
  * Bigger creator-style captions, slow continuous push-in, subtle vignette and grade

Usage (PowerShell):
    $env:GEMINI_API_KEY = "your_key"
    python reel_pro.py "https://youtu.be/VIDEO_ID" -n 2 --music "C:\\path\\to\\music"

Only use on videos you own or may reuse.
"""
import argparse
import colorsys
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from reel_editor import (FPS, Word, Zoom, clean_plan, esc, keep_segments, make_client, make_sfx,
                         norm, pick_music, probe_duration, smooth, tighten, ts)
from shorts_clipper import DEFAULT_MODEL, analyze, download, ffmpeg, parse_ts, reframe, slug

INTRO_S, OUTRO_S = 0.6, 0.7
BANNER_TOP = 110          # keeps the banner under the phone's status bar / Shorts header
YELLOW, RED = "&H00E6FF&", "&H3B3BFF&"


# ----------------------------------------------------------------------------- Hinglish transcription
PROMPT_HI = ("Haan toh basically yeh jo cheez hai na, matlab bahut important hai. Umm, uh, aur phir "
             "hum log kya karte hain, you know, bilkul simple hai, samjhe?")
PROMPT_EN = "Umm, uh, so, like, you know, I mean, hmm. Okay, so, uh, here's the thing."
_models: dict = {}


def transcribe_pro(path, size, lang) -> list[Word]:
    from faster_whisper import WhisperModel
    if size not in _models:
        _models[size] = WhisperModel(size, device="cpu", compute_type="int8")
    segs, _ = _models[size].transcribe(
        str(path), word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
        language=None if lang == "auto" else lang, task="transcribe",
        initial_prompt=PROMPT_HI if lang == "hi" else PROMPT_EN)
    return [Word(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]


# ----------------------------------------------------------------------------- Gemini edit plan (+ transcript fixes)
class Fix(BaseModel):
    i: int
    text: str


class PlanPro(BaseModel):
    hooks: list[str]
    best_hook: int
    keywords: list[str]
    zooms: list[Zoom]
    fixes: list[Fix]


PLAN_PRO = """You are a top-tier viral short-form editor. This clip is {dur:.1f} seconds long, \
already cut tight. {lang_note}
Word-timed transcript from Whisper, as index:seconds:word (it contains errors):
{transcript}

Return an edit plan.

FIXES: Listen to the audio. Whisper mis-hears words, especially Hindi, Hinglish and Indian accents. \
For every word index that is wrong, give the corrected word. Write Hindi words in Roman script the \
way creators type Hinglish ("kya", "nahi", "matlab", "bhai"), keep English words in English. \
Any word in Devanagari script MUST be fixed to Roman script. One word replaces one word: never \
merge or split. Only list words that need fixing.

HOOKS: exactly 5 hooks, max 8 words each, shown as bold text in a banner at the top for the whole reel.
- Shocking, emotional, urgent, controversial or surprising. Pattern-interrupting, never generic.
- Create a curiosity gap that THIS CLIP actually pays off. The hook must be true to what is said or \
shown; baited viewers swipe and kill retention.
- Banned: "wait for it", "you won't believe", "this changes everything", "watch till the end".
- Write the hooks in the speaker's own style: Hinglish in Roman script if they speak Hinglish.
- Use concrete specifics (numbers, names, stakes, a bold claim).
- best_hook = index (0-4) of the one most likely to stop a scroll.

KEYWORDS: 6-15 single words, lowercase, spelled exactly as they will appear AFTER your fixes, that \
carry the most meaning, emotion, numbers or surprise. They get colour emphasis in the captions.

ZOOMS: 4-8 punch-in zooms, roughly one every 5-6 seconds. Each lands on something that matters: a \
key claim, an emotional beat, a surprising detail, a face, an object, an action, on-screen text.
- start/end in seconds of this clip, 1.2-3.0s long, no overlap, at least 1s apart, none before 1.0s.
- scale 1.2-1.5.
- x,y = normalised (0-1) centre of the exact thing viewers should look at. LOOK at the video and \
place it precisely.
- reason: a few words."""


def plan_pro(client, model, clip: Path, words, dur, lang, log):
    f = None
    try:
        f = client.files.upload(file=str(clip))
        while f.state is not None and f.state.name == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        note = ("The speaker mixes Hindi and English (Hinglish)." if lang == "hi"
                else "The speaker speaks English." if lang == "en" else "")
        transcript = " ".join(f"{i}:{w.start:.1f}:{w.text}" for i, w in enumerate(words))
        resp = client.models.generate_content(
            model=model,
            contents=[f, PLAN_PRO.format(dur=dur, transcript=transcript, lang_note=note)],
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=PlanPro, temperature=0.8))
        return resp.parsed or PlanPro(**json.loads(resp.text))
    except Exception as e:  # noqa: BLE001
        log(f"  Gemini edit plan failed ({e}); using fallback hook, Whisper text, no zooms")
        return None
    finally:
        if f is not None:
            try:
                client.files.delete(name=f.name)
            except Exception:  # noqa: BLE001
                pass


def apply_fixes(words, fixes):
    out = list(words)
    for fx in fixes or []:
        t = esc(fx.text).strip()
        if 0 <= fx.i < len(out) and t and len(t.split()) == 1:
            out[fx.i] = Word(t, out[fx.i].start, out[fx.i].end)
    return out


# ----------------------------------------------------------------------------- theme palette
def lum(rgb):
    def f(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def hsv(h, s, v):
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(h % 1, s, v))


def ass_col(rgb):
    return f"&H{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}&"


def sample_pixels(clip, n=12):
    dur = probe_duration(clip)
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
                                   f"fps={n / dur:.4f},scale=24:42", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-"])
    return [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw) - 2, 3)]


def theme_palette(clip):
    """Banner / text / accent colours derived from the video: dominant hue (saturation-weighted)
    and overall brightness. Text colour is whichever of white / near-black reads best."""
    try:
        px = sample_pixels(clip)
    except Exception:  # noqa: BLE001
        px = []
    if not px:
        return {"bg": (11, 11, 15), "text": (255, 255, 255), "accent": (255, 199, 0), "mode": "default"}
    mean_l = sum(0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in px) / len(px) / 255
    sx = sy = wt = 0.0
    for r, g, b in px:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        w = s * v
        sx += w * math.cos(2 * math.pi * h)
        sy += w * math.sin(2 * math.pi * h)
        wt += w
    sat = wt / len(px)
    hue = (math.atan2(sy, sx) / (2 * math.pi)) % 1 if wt > 1e-6 else 0.12

    if sat < 0.08:                      # near-greyscale footage: black banner, gold accent
        bg, accent, mode = (11, 11, 15), (255, 199, 0), "neutral"
    elif mean_l < 0.38:                 # dark video: vivid banner that pops
        bg, accent, mode = hsv(hue, 0.88, 0.95), hsv(hue + 0.5, 0.85, 1.0), "dark-video/vivid"
    else:                               # bright video: dark tinted banner, bright analogous accent
        bg, accent, mode = hsv(hue, 0.70, 0.14), hsv(hue, 0.85, 1.0), "bright-video/dark"
    white, ink = (255, 255, 255), (17, 17, 17)
    text = white if contrast(white, bg) >= contrast(ink, bg) else ink
    if contrast(accent, bg) < 3:
        accent = text
    return {"bg": bg, "text": text, "accent": accent, "mode": mode}


# ----------------------------------------------------------------------------- ASS: banner, captions, CTA
ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,90,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,1,9,5,2,50,50,540,1
Style: Hook,Arial Black,66,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def banner_events(hook, pal, dur, font=66, width=18):
    lines = textwrap.wrap(esc(hook).upper(), width=width)[:3] or [""]
    h = int(len(lines) * font * 1.2) + 56
    bg, txt, acc = ass_col(pal["bg"]), ass_col(pal["text"]), ass_col(pal["accent"])
    a, b = ts(0), ts(dur)
    rect = lambda hh: f"m 0 0 l 1080 0 1080 {hh} 0 {hh}"  # noqa: E731
    body = "\\N".join(lines[:-1] + [f"{{\\1c{acc}}}{lines[-1]}{{\\1c{txt}}}"] if len(lines) > 1 else lines)
    return [
        f"Dialogue: 0,{a},{b},Hook,,0,0,0,,{{\\an7\\move(0,{BANNER_TOP - h},0,{BANNER_TOP},0,260)"
        f"\\bord0\\shad0\\1c{bg}\\p1}}{rect(h)}{{\\p0}}",
        f"Dialogue: 0,{a},{b},Hook,,0,0,0,,{{\\an7\\move(0,{BANNER_TOP},0,{BANNER_TOP + h},0,260)"
        f"\\bord0\\shad0\\1c{acc}\\p1}}{rect(10)}{{\\p0}}",
        f"Dialogue: 3,{a},{b},Hook,,0,0,0,,{{\\an8\\move(540,{BANNER_TOP - h + 26},540,{BANNER_TOP + 26},0,260)"
        f"\\fad(120,0)\\1c{txt}}}{body}",
    ]


def build_ass_pro(words, keywords, hook, pal, dur, cta):
    lines = [ASS_HEAD]
    if hook:
        lines += banner_events(hook, pal, dur)
    if cta:
        lines.append(f"Dialogue: 4,{ts(max(dur - 1.5, 0))},{ts(dur - 0.12)},Cap,,0,0,0,,"
                     f"{{\\an5\\move(540,1230,540,1180,0,250)\\fad(200,250)\\fs72\\1c{ass_col(pal['accent'])}}}"
                     f"{esc(cta).upper()}")

    groups, g = [], []
    for w in words:
        t = esc(w.text)
        if not norm(t):
            continue
        if g and (len(g) >= 3 or len(" ".join(x.text for x in g) + " " + t) > 14
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
            lines.append(f"Dialogue: 2,{ts(w.start)},{ts(end)},Cap,,0,0,0,,{' '.join(toks)}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- render
def zoom_filter_pro(zooms, dur):
    ws = [smooth(z.start, z.end, 0.2) for z in zooms]
    intro_w = f"pow(max(1-on/{FPS}/{INTRO_S},0),2)"
    z_terms = [f"0.04*on/({FPS}*{dur:.2f})",                                  # slow push-in all reel
               f"0.32*{intro_w}",                                             # intro punch-out
               f"0.10*pow(clip((on/{FPS}-({dur:.2f}-{OUTRO_S}))/{OUTRO_S},0,1),2)"]  # outro push-in
    z_terms += [f"({z.scale - 1:.3f})*{w}" for z, w in zip(zooms, ws)]
    cx = "0.5" + "".join(f"+({z.x - 0.5:.3f})*{w}" for z, w in zip(zooms, ws))
    cy = "0.5+(-0.08)*" + intro_w + "".join(f"+({z.y - 0.5:.3f})*{w}" for z, w in zip(zooms, ws))
    x = f"clip(({cx})*iw-iw/zoom/2,0,iw-iw/zoom)"
    y = f"clip(({cy})*ih-ih/zoom/2,0,ih-ih/zoom)"
    return f"zoompan=z='1+{'+'.join(z_terms)}':x='{x}':y='{y}':d=1:s=1080x1920:fps={FPS}"


def sfx_group(a, mix, idx, events, tag, st):
    n = len(events)
    outs = "".join(f"[{tag}{i}]" for i in range(n))
    a.append(f"[{idx}:a]{st},asplit={n}{outs}" if n > 1 else f"[{idx}:a]{st},anull[{tag}0]")
    for i, (ms, vol) in enumerate(events):
        a.append(f"[{tag}{i}]volume={vol},adelay={ms}|{ms}[{tag}d{i}]")
        mix.append(f"[{tag}d{i}]")


def render_pro(clip: Path, dst: Path, ass_name: str, zooms, music, dur):
    work = dst.parent
    make_sfx(work)
    inputs, idx, music_i = ["-i", clip.name], 1, None
    if music:
        inputs += ["-stream_loop", "-1", "-i", str(Path(music).resolve())]
        music_i, idx = idx, idx + 1
    inputs += ["-i", "whoosh.wav", "-i", "hit.wav"]
    whoosh_i, hit_i = idx, idx + 1

    blur_p = (f"min(1,max(1-T/{INTRO_S},0)+max((T-({dur:.2f}-{OUTRO_S}))/{OUTRO_S},0))")
    v = (f"[0:v]fps={FPS},eq=contrast=1.05:saturation=1.12:gamma=0.98,unsharp=5:5:0.5:3:3:0.0,"
         f"scale=1620:2880:flags=lanczos,{zoom_filter_pro(zooms, dur)},format=yuv420p,"
         f"split=2[s][b0];"
         f"[b0]scale=270:480,gblur=sigma=5,scale=1080:1920,format=yuv420p[b];"
         f"[s][b]blend=all_expr='A*(1-{blur_p})+B*{blur_p}':"
         f"enable='lt(t,{INTRO_S})+gt(t,{dur - OUTRO_S:.2f})',"
         f"vignette=angle=PI/7,ass={ass_name},"
         f"fade=t=in:st=0:d=0.28:color=white,fade=t=out:st={max(dur - 0.5, 0):.2f}:d=0.5[v]")

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
    whooshes = [(0, 0.6)] + [(int(max(z.start - 0.12, 0) * 1000), 0.55) for z in zooms]
    sfx_group(a, mix, whoosh_i, whooshes, "w", st)
    sfx_group(a, mix, hit_i, [(40, 0.9), (int(max(dur - 0.45, 0) * 1000), 0.5)], "h", st)
    a.append("".join(mix) + f"amix=inputs={len(mix)}:duration=first:normalize=0:dropout_transition=0,"
             f"loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=out:st={max(dur - 0.5, 0):.2f}:d=0.5[a]")

    ffmpeg(*inputs, "-filter_complex", ";".join([v] + a), "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.2f}", "-movflags", "+faststart",
           dst.name, cwd=work)


# ----------------------------------------------------------------------------- one clip, end to end
def edit_clip_pro(raw: Path, final: Path, client, model, whisper_size, lang, music, cta,
                  fallback_hook, log=print):
    work = final.parent
    dur0 = probe_duration(raw)
    log("  transcribing (pass 1) ...")
    w0 = transcribe_pro(raw, whisper_size, lang)
    gap = 0.30
    segs = keep_segments(w0, dur0, gap=gap)
    while len(segs) > 70:
        gap += 0.1
        segs = keep_segments(w0, dur0, gap=gap)

    tight = work / f"{final.stem}_tight.mp4"
    tighten(raw, segs, tight)
    dur = probe_duration(tight)
    log(f"  tightened {dur0:.1f}s -> {dur:.1f}s ({len(segs)} cuts)")

    log("  transcribing (pass 2) ...")
    words = transcribe_pro(tight, whisper_size, lang)
    log("  asking Gemini: fixes, hooks, keywords, zooms ...")
    plan = plan_pro(client, model, tight, words, dur, lang, log)
    if plan:
        words = apply_fixes(words, plan.fixes)
        log(f"  fixed {len(plan.fixes)} words")
    hooks, hook, keywords, zooms = clean_plan(plan, dur, words, fallback_hook)
    pal = theme_palette(tight)
    log(f"  hook: {hook}  (banner theme: {pal['mode']})")

    ass = work / f"{final.stem}.ass"
    ass.write_text(build_ass_pro(words, keywords, hook, pal, dur, cta), encoding="utf-8")
    log(f"  rendering ({len(zooms)} zooms) ...")
    render_pro(tight, final, ass.name, zooms, pick_music(music), dur)

    for tmp in (tight, ass, work / "whoosh.wav", work / "hit.wav"):
        tmp.unlink(missing_ok=True)
    return {"hook": hook, "hook_options": hooks, "keywords": sorted(keywords),
            "zooms": [z.model_dump() for z in zooms], "duration": round(dur, 1),
            "banner_theme": pal["mode"]}


def pipeline(url, api_key, out="reels_out", model=DEFAULT_MODEL, n=3, lo=25, hi=60, fit="crop",
             music=None, whisper_size="small", lang="hi", cta="Follow for more", keep_source=False,
             log=print):
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
            info = edit_clip_pro(raw, final, client, model, whisper_size, lang, music, cta, c.hook, log)
        except Exception as e:  # noqa: BLE001
            log(f"  Clip {i} failed: {e}")
            continue
        finally:
            raw.unlink(missing_ok=True)
        (out / f"{name}_hooks.txt").write_text(
            "\n".join(f"{k + 1}. {h}" for k, h in enumerate(info["hook_options"])), encoding="utf-8")
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
    ap.add_argument("--whisper", default="small", help="tiny/base/small/medium/large-v3 (bigger = more accurate, slower)")
    ap.add_argument("--lang", default="hi", choices=["hi", "en", "auto"],
                    help="hi = Hinglish (Roman script), en = English, auto = let Whisper detect")
    ap.add_argument("--cta", default="Follow for more", help="end-card text, '' to disable")
    ap.add_argument("--keep-source", action="store_true")
    a = ap.parse_args()
    if not a.api_key:
        sys.exit("Set GEMINI_API_KEY or pass --api-key")
    pipeline(a.url, a.api_key, a.out, a.model, a.n, a.lo, a.hi, a.fit, a.music, a.whisper,
             a.lang, a.cta, a.keep_source)


if __name__ == "__main__":
    main()
