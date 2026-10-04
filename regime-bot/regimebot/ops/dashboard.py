"""Read-only live dashboard. Standard library only, bound to localhost.

Reach it from your laptop through an SSH tunnel: ssh -L 8765:127.0.0.1:8765 <server>
"""

from __future__ import annotations

import ipaddress
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from regimebot.decide.playbook import Playbook
from regimebot.hmm.fit import RegimeModel, expected_durations


def _last_candle(state_dir: Path) -> dict[str, Any]:
    p = state_dir / "journal.jsonl"
    if not p.exists():
        return {}
    last: dict[str, Any] = {}
    with p.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "candle":
                last = rec
    return last


def snapshot(state_dir: Path, model: RegimeModel, books: Mapping[str, Playbook]) -> dict[str, Any]:
    rec = _last_candle(state_dir)
    res = rec.get("result", {})
    d = res.get("decision", {})
    active = d.get("active")
    dur = None
    try:
        alpha = json.loads((state_dir / "engine.json").read_text()).get("alpha")
        if alpha and active:
            idx = [i for i, lab in enumerate(model.labels) if lab == active]
            w = np.array([alpha[i] for i in idx])
            ds = expected_durations(model.transmat)[idx]
            dur = float((w * ds).sum() / w.sum()) if w.sum() > 0 else None
    except (OSError, ValueError, KeyError):
        pass
    pb = books.get(active) if active else None
    return {
        "ts": rec.get("ts"),
        "probs": d.get("probs", {}),
        "next_probs": d.get("next_probs", {}),
        "current_state": active,
        "uncertain": d.get("uncertain"),
        "expected_duration": dur,
        "playbook": asdict(pb) if pb else None,
        "action": {"reason": d.get("reason"), "orders": res.get("orders", []),
                   "vetoes": res.get("vetoes", []), "target": d.get("target")},
        "result": {"equity": rec.get("equity"), "position": rec.get("position_after"),
                   "fills": rec.get("fills", [])},
    }


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Regime Bot</title>
<style>
:root{--bg:#fafafa;--fg:#1a1a1a;--muted:#666;--bar:#3b6ea5;--card:#fff;--line:#e3e3e3}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e8e8e8;--muted:#9aa0a6;
--bar:#6fa3d8;--card:#1b1f24;--line:#2c3138}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
h1{font-size:20px}h2{font-size:13px;color:var(--muted);margin:0 0 8px;text-transform:uppercase}
.row{display:flex;align-items:center;gap:8px;margin:4px 0}.row span{width:80px}
.track{flex:1;background:var(--line);height:10px;border-radius:5px}
.fill{background:var(--bar);height:10px;border-radius:5px}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:0}
.big{font-size:24px;font-weight:600}
</style></head><body><main><h1>Regime bot (paper)</h1><div class="grid">
<div class="card"><h2>Current state</h2><div class="big" id="state">-</div>
<div id="dur"></div><div id="ts" style="color:var(--muted)"></div></div>
<div class="card"><h2>State probabilities</h2><div id="probs"></div></div>
<div class="card"><h2>Active playbook</h2><pre id="pb"></pre></div>
<div class="card"><h2>Action taken</h2><pre id="act"></pre></div>
<div class="card"><h2>Result</h2><pre id="res"></pre></div></div></main>
<script>
function bars(p){return Object.entries(p||{}).map(([k,v])=>
 `<div class="row"><span>${k}</span><div class="track"><div class="fill"
 style="width:${(v*100).toFixed(1)}%"></div></div><b>${(v*100).toFixed(1)}%</b></div>`).join("")}
function show(s){
 const st=(s.current_state||"none")+(s.uncertain?" (uncertain)":"");
 document.getElementById("state").textContent=st;
 document.getElementById("dur").textContent=s.expected_duration?
   `expected duration ${s.expected_duration.toFixed(1)} bars`:"";
 document.getElementById("ts").textContent=s.ts||"";
 document.getElementById("probs").innerHTML=bars(s.probs);
 document.getElementById("pb").textContent=JSON.stringify(s.playbook,null,1);
 document.getElementById("act").textContent=JSON.stringify(s.action,null,1);
 document.getElementById("res").textContent=JSON.stringify(s.result,null,1);}
fetch("/api/state").then(r=>r.json()).then(show);
new EventSource("/events").onmessage=e=>show(JSON.parse(e.data));
</script></body></html>"""


def serve(state_dir: Path, model: RegimeModel, books: Mapping[str, Playbook],
          host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError("dashboard binds to loopback only; use an SSH tunnel")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                body = json.dumps(snapshot(state_dir, model, books), default=str).encode()
                self._send(200, body, "application/json")
            elif self.path.startswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                once = "once=1" in self.path
                journal = state_dir / "journal.jsonl"
                seen = -1.0
                try:
                    while True:
                        m = journal.stat().st_mtime if journal.exists() else 0.0
                        if m != seen:
                            seen = m
                            data = json.dumps(snapshot(state_dir, model, books), default=str)
                            self.wfile.write(f"data: {data}\n\n".encode())
                            self.wfile.flush()
                            if once:
                                return
                        time.sleep(2)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._send(404, b"not found", "text/plain")

        def _deny(self) -> None:
            self._send(405, b"read-only", "text/plain")

        do_POST = do_PUT = do_DELETE = do_PATCH = _deny  # noqa: N815

    return ThreadingHTTPServer((host, port), Handler)
