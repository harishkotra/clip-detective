#!/usr/bin/env python3
"""Clip Detective - local video Q&A on LFM3-VL-2B. No cloud, no API key.

Serves the UI, samples frames with ffmpeg, and talks to llama-server
(OpenAI-compatible) with the sampled frames attached as images.
"""
import base64
import json
import os
import re
import shutil
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sampler

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
STATIC = os.path.join(ROOT, "static")
LLAMA = os.environ.get("LLAMA_URL", "http://127.0.0.1:9911")

SYSTEM = (
    "You are Clip Detective. You are given frames sampled from ONE video, each labeled "
    "with its timestamp. The user asks a question about the video. Answer only from "
    "what is visible in the frames. Be specific and short: state the answer, then cite "
    "the frame timestamp in the form [mm:ss]. If the frames do not show it, say what "
    "you would need instead of guessing."
)


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def llama_health() -> bool:
    try:
        with urllib.request.urlopen(LLAMA + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ask(question: str, frames, meta) -> dict:
    content = [{
        "type": "text",
        "text": (f"Video: {meta['filename']}, {mmss(meta['duration'])} long, "
                 f"{len(frames)} frames sampled.\n\nQuestion: {question}")
    }]
    for fr in frames:
        with open(os.path.join(DATA, meta["id"], fr["f"]), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        content.append({"type": "text", "text": f"Frame at {mmss(fr['t'])}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": content}],
        "temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0,
        "repeat_penalty": 1.1, "max_tokens": 700, "stream": False,
    }).encode()

    t0 = time.time()
    req = urllib.request.Request(LLAMA + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = json.loads(r.read())
    dt = time.time() - t0

    msg = payload["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if not text:
        # reasoning-model fallback: take the last paragraph of the think stream
        think = (msg.get("reasoning_content") or "").strip()
        text = think.split("\n\n")[-1] if think else ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return {"answer": text, "seconds": round(dt, 1),
            "tokens": payload.get("usage", {}).get("completion_tokens", 0),
            "frames": len(frames)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            return self._file(os.path.join(STATIC, "index.html"), "text/html")
        if p == "/api/health":
            return self._send(200, {"model": llama_health()})
        m = re.match(r"^/frames/([a-f0-9]+)/([^/]+)$", p)
        if m:
            return self._file(os.path.join(DATA, m.group(1), m.group(2)), "image/jpeg")
        if p.startswith("/static/"):
            return self._file(os.path.join(STATIC, p[8:]),
                              "text/css" if p.endswith(".css") else "application/javascript")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if self.path == "/api/upload":
            vid = uuid.uuid4().hex[:12]
            d = os.path.join(DATA, vid)
            os.makedirs(d, exist_ok=True)
            name = urllib.request.unquote(self.headers.get("X-Filename", "clip.mp4"))
            ext = os.path.splitext(name)[1] or ".mp4"
            src = os.path.join(d, "source" + ext)
            with open(src, "wb") as fh:
                remaining = n
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            try:
                dur, frames = sampler.extract(src, d)
            except Exception as e:
                return self._send(500, {"error": f"could not read video: {e}"})
            meta = {"id": vid, "filename": name, "duration": dur,
                    "frames": frames, "src": os.path.basename(src)}
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump(meta, fh)
            return self._send(200, meta)

        if self.path == "/api/ask":
            req = json.loads(self.rfile.read(n) or b"{}")
            meta_path = os.path.join(DATA, req.get("id", ""), "meta.json")
            if not os.path.isfile(meta_path):
                return self._send(404, {"error": "unknown clip"})
            with open(meta_path) as fh:
                meta = json.load(fh)
            try:
                return self._send(200, ask(req["question"], meta["frames"], meta))
            except Exception as e:
                return self._send(500, {"error": str(e)})

        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    port = int(os.environ.get("PORT", "8787"))
    print(f"Clip Detective on http://127.0.0.1:{port}  (model: {LLAMA})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
