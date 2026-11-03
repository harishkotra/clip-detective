#!/usr/bin/env python3
"""Clip Detective - sample frames from a video so a 2B model can answer questions about it.

Strategy: scene-change detection first (captures real events), then uniform fill
if we got too few frames. Every frame keeps its real timestamp.
"""
import json
import os
import re
import shutil
import subprocess
import sys


def _tool(name, env):
    """ffmpeg lives in different places on different machines."""
    if os.environ.get(env):
        return os.environ[env]
    found = shutil.which(name)
    if found:
        return found
    for cand in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}", f"/usr/bin/{name}"):
        if os.path.exists(cand):
            return cand
    sys.exit(f"{name} not found. Install ffmpeg or set {env}=/path/to/{name}")


FFMPEG = _tool("ffmpeg", "FFMPEG")
FFPROBE = _tool("ffprobe", "FFPROBE")

MAX_FRAMES = 12
FRAME_WIDTH = 896

def duration(path: str) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def scene_frames(path: str, outdir: str, threshold: float):
    """Extract scene-change frames, return [(timestamp, filename)]."""
    cmd = [FFMPEG, "-v", "info", "-i", path, "-an",
           "-vf", f"select='gt(scene,{threshold})',showinfo,scale={FRAME_WIDTH}:-2",
           "-vsync", "vfr", "-q:v", "3", os.path.join(outdir, "scene_%04d.jpg")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stamps = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
    files = sorted(f for f in os.listdir(outdir) if f.startswith("scene_"))
    return list(zip(stamps, files))


def uniform_frames(path: str, outdir: str, dur: float, need: int, start_idx: int):
    """Grab `need` evenly spaced frames, avoiding the very start/end."""
    out = []
    for i in range(need):
        t = dur * (i + 0.5) / need
        name = f"uni_{start_idx + i:04d}.jpg"
        subprocess.run(
            [FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
             "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "3", os.path.join(outdir, name)],
            capture_output=True)
        if os.path.exists(os.path.join(outdir, name)):
            out.append((t, name))
    return out


def pick(frames, max_n):
    """Keep frames spread across the timeline, always keeping first and last.
    Dedupes by whole-second so the strip never shows the same moment twice."""
    seen_sec, uniq = set(), []
    for t, f in sorted(frames, key=lambda x: x[0]):
        if int(t) in seen_sec:
            continue
        seen_sec.add(int(t))
        uniq.append((t, f))
    if len(uniq) <= max_n:
        return uniq
    step = len(uniq) / max_n
    out = []
    for i in range(max_n):
        out.append(uniq[int(i * step)])
    return out


def extract(path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    dur = duration(path)

    for thresh in (0.30, 0.18, 0.10):
        frames = scene_frames(path, outdir, thresh)
        if len(frames) >= 4:
            break

    if len(frames) < 4:
        frames += uniform_frames(path, outdir, dur, MAX_FRAMES, 0)

    frames = pick(frames, MAX_FRAMES)
    return dur, [{"t": round(t, 2), "f": f} for t, f in frames]


if __name__ == "__main__":
    print(json.dumps(extract(sys.argv[1], sys.argv[2])))
