#!/usr/bin/env python3
"""Irodori-TTS 再生アプリ — テキスト → Databricks Model Serving で音声合成 → その場で再生。

依存なし（Python 標準ライブラリのみ）。認証は既存の Databricks CLI プロファイルを流用。

特長:
  - 長文は「文ごとに分割 → 各文を合成 → 連結」して品質を確保（1回の合成は ~256トークン/~30秒が上限のため）。
  - モデルがサポートする全パラメータ（num_steps, cfg_scale_*, caption/VoiceDesign, duration_scale,
    sway, truncation, seed, 参照音声によるボイスクローン ...）を `params_json` 経由で指定可能。
  - 「追加パラメータ(JSON)」欄に任意の params_json を書けば、いかなるチューニングにも対応。

起動:
    python tts_app.py                       # http://127.0.0.1:8765 を開く
    python tts_app.py --port 9000 --profile DEFAULT --endpoint irodori-tts-express
"""
from __future__ import annotations

import argparse
import base64
import configparser
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CFG = {"profile": "DEFAULT", "endpoint": "irodori-tts-express"}

PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Irodori-TTS 再生</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, "Hiragino Kaku Gothic ProN", Meiryo, system-ui, sans-serif;
         max-width: 780px; margin: 0 auto; padding: 24px; line-height: 1.6; }
  h1 { font-size: 1.3rem; margin: 0 0 4px; }
  .sub { color: #888; font-size: .85rem; margin-bottom: 18px; }
  textarea { width: 100%; box-sizing: border-box; padding: 12px; font-size: 1rem;
             border: 1px solid #8888; border-radius: 10px; resize: vertical;
             background: #0000000a; font-family: inherit; }
  textarea#text { min-height: 140px; } textarea#extra { min-height: 70px; font-size: .85rem; font-family: ui-monospace, monospace; }
  .row { display: flex; gap: 14px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
  button { padding: 10px 22px; font-size: 1rem; border: 0; border-radius: 10px;
           background: #ff5f46; color: #fff; cursor: pointer; font-weight: 600; }
  button:disabled { opacity: .5; cursor: default; }
  label { font-size: .82rem; color: #888; display: inline-flex; gap: 6px; align-items: center; }
  input[type=number], input[type=text] { padding: 6px 8px; border-radius: 8px; border: 1px solid #8888;
                       background: #0000000a; font: inherit; }
  input[type=number] { width: 80px; } input#caption { width: 260px; }
  details { border: 1px solid #8884; border-radius: 10px; padding: 8px 12px; margin: 12px 0; }
  summary { cursor: pointer; font-size: .9rem; color: #888; }
  #status { font-size: .9rem; min-height: 1.4em; margin-top: 8px; }
  audio { width: 100%; margin-top: 12px; }
  .meta { font-size: .8rem; color: #888; margin-top: 6px; white-space: pre-wrap; }
  a { color: #ff5f46; }
</style></head>
<body>
  <h1>🗣️ Irodori-TTS 再生</h1>
  <div class="sub">日本語テキストを入力して「生成して再生」。エンドポイント <code id="ep"></code>（Databricks Model Serving）が合成します。
    長文は文ごとに分割・連結して品質を確保します。</div>

  <textarea id="text">こんにちは。今日はデータブリックスについてご紹介します。データブリックスは、データとAIを一つに統合した、データインテリジェンスプラットフォームです。レイクハウスアーキテクチャを基盤に、データエンジニアリング、分析、機械学習、そして生成AIまでを、単一の基盤で扱うことができます。ぜひ、あなたのデータ活用にお役立てください。</textarea>

  <div class="row">
    <button id="go">生成して再生</button>
    <label><input type="checkbox" id="split" checked> 長文を分割して連結</label>
    <label>分割方式
      <select id="chunk_mode">
        <option value="semantic" selected>セマンティック (LLM)</option>
        <option value="rule">文字数ベース (高速)</option>
      </select></label>
    <label>1チャンク最大文字数 <input type="number" id="max_chars" value="170" min="20" max="260"></label>
  </div>
  <div class="row">
    <label><input type="checkbox" id="fix_voice" checked> 声を固定（全文を同一の声に揃える）</label>
    <label>voice_seed <input type="number" id="voice_seed" value="42"></label>
  </div>

  <div class="row">
    <label>num_steps <input type="number" id="num_steps" value="40" min="1" max="200"></label>
    <label>cfg_scale_text <input type="number" id="cfg_scale_text" value="3.0" step="0.5"></label>
    <label>duration_scale <input type="number" id="duration_scale" value="1.0" step="0.1"></label>
    <label>seed <input type="number" id="seed" placeholder="ランダム"></label>
  </div>
  <div class="row">
    <label>caption（VoiceDesign / 話し方）<input type="text" id="caption" placeholder="例: 落ち着いた女性、ゆっくり丁寧に"></label>
    <label>cfg_scale_caption <input type="number" id="cfg_scale_caption" value="3.0" step="0.5"></label>
  </div>

  <details>
    <summary>追加設定 — セマンティック分割用LLM / 任意の合成パラメータ</summary>
    <p style="font-size:.82rem;color:#888;margin:8px 0">
      分割用LLM (FM API): <input type="text" id="fm_model" value="databricks-claude-haiku-4-5" style="width:260px"></p>
    <p style="font-size:.8rem;color:#888;margin:8px 0">合成パラメータ上書き (JSON) 例:
      <code>{"t_schedule_mode":"sway","sway_coeff":-1.0,"num_steps":8}</code>。
      <a href="#" id="show_params">サポートされているパラメータ一覧を表示</a></p>
    <textarea id="extra" placeholder='{}'></textarea>
  </details>

  <div id="status"></div>
  <audio id="player" controls hidden></audio>
  <div class="meta" id="meta"></div>
  <div><a id="dl" hidden download="tts.wav">⬇︎ WAVをダウンロード</a></div>

<script>
document.getElementById('ep').textContent = "%ENDPOINT%";
const $ = id => document.getElementById(id);
const go=$('go'), status=$('status'), player=$('player'), meta=$('meta'), dl=$('dl');

function collectParams() {
  const p = {};
  const ns=$('num_steps').value; if(ns) p.num_steps=Number(ns);
  const ct=$('cfg_scale_text').value; if(ct) p.cfg_scale_text=Number(ct);
  const ds=$('duration_scale').value; if(ds && Number(ds)!==1) p.duration_scale=Number(ds);
  const sd=$('seed').value; if(sd!=='') p.seed=Number(sd);
  const cap=$('caption').value.trim(); if(cap){ p.caption=cap; const cc=$('cfg_scale_caption').value; if(cc) p.cfg_scale_caption=Number(cc); }
  const extra=$('extra').value.trim();
  if(extra){ try { Object.assign(p, JSON.parse(extra)); } catch(e){ throw new Error('追加パラメータのJSONが不正: '+e.message); } }
  return p;
}

go.onclick = async () => {
  const text = $('text').value.trim();
  if(!text){ status.textContent='テキストを入力してください。'; return; }
  let params;
  try { params = collectParams(); } catch(e){ status.textContent='❌ '+e.message; return; }
  go.disabled=true; player.hidden=true; dl.hidden=true; meta.textContent='';
  status.textContent='⏳ 合成中… (初回はエンドポイント起動で1〜2分かかる場合があります)';
  const t0=performance.now();
  try {
    const res = await fetch('/api/tts', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, params, split: $('split').checked, max_chars: Number($('max_chars').value||170),
                            chunk_mode: $('chunk_mode').value, fm_model: $('fm_model').value.trim(),
                            fix_voice: $('fix_voice').checked, voice_seed: Number($('voice_seed').value||42)})});
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || ('HTTP '+res.status));
    const src='data:audio/wav;base64,'+data.audio_wav_b64;
    player.src=src; player.hidden=false; player.play().catch(()=>{});
    dl.href=src; dl.hidden=false;
    const wall=((performance.now()-t0)/1000).toFixed(1);
    meta.textContent = `音声 ${data.total_audio_seconds}s / サーバ合成 ${data.total_synth_seconds}s / 往復 ${wall}s / ${data.segments}文\n`
      + (data.notes||'');
    status.textContent='✅ 完了';
  } catch(e){ status.textContent='❌ エラー: '+e.message; }
  finally { go.disabled=false; }
};

$('show_params').onclick = async (ev) => {
  ev.preventDefault(); status.textContent='パラメータ一覧を取得中…';
  try {
    const res=await fetch('/api/params'); const d=await res.json();
    if(!res.ok) throw new Error(d.error||('HTTP '+res.status));
    $('extra').placeholder = 'サポート: ' + (d.supported_params||[]).join(', ');
    meta.textContent = 'サポートされているパラメータ:\n' + (d.supported_params||[]).join(', ')
      + '\n+ ref_wav_b64 (base64 WAV でボイスクローン)';
    status.textContent='✅ 取得しました（追加パラメータ欄のプレースホルダにも表示）';
  } catch(e){ status.textContent='❌ '+e.message; }
};
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
def _host_token() -> tuple[str, str]:
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    section = cfg[CFG["profile"]] if (cfg.has_section(CFG["profile"]) or CFG["profile"] == "DEFAULT") else {}
    host = section.get("host") if section else None
    if not host:
        raise RuntimeError(f"host not found for profile '{CFG['profile']}'")
    host = host.rstrip("/")
    token = None
    try:
        out = subprocess.run(["databricks", "auth", "token", "-p", CFG["profile"]],
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            token = json.loads(out.stdout).get("access_token")
    except Exception:
        token = None
    if not token:
        token = section.get("token") if section else None
    if not token:
        raise RuntimeError("could not obtain an access token")
    return host, token


def _invoke(records: list[dict]) -> list[dict]:
    """POST dataframe_records to the endpoint (long timeout survives cold start)."""
    host, token = _host_token()
    url = f"{host}/serving-endpoints/{CFG['endpoint']}/invocations"
    body = json.dumps({"dataframe_records": records}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:400]}") from e
    preds = data.get("predictions") or data.get("outputs") or data
    return preds if isinstance(preds, list) else [preds]


def _split_sentences(text: str) -> list[str]:
    """Split on Japanese/ASCII sentence enders and newlines, keeping the ender."""
    parts = re.split(r"(?<=[。！？!?\n])", text)
    return [s.strip() for s in parts if s.strip()]


def _atomic_units(text: str, max_chars: int) -> list[str]:
    """Split into the smallest safe units: sentences, and any over-long sentence further
    split on Japanese commas. Every unit is <= max_chars, so grouping never truncates."""
    parts: list[str] = []
    for s in _split_sentences(text):
        if len(s) <= max_chars:
            parts.append(s)
        else:
            buf = ""
            for x in re.split(r"(?<=[、,])", s):
                if buf and len(buf) + len(x) > max_chars:
                    parts.append(buf); buf = x
                else:
                    buf += x
            if buf:
                parts.append(buf)
    return parts or [text]


def _greedy_group(units: list[str], max_chars: int) -> list[str]:
    """Pack consecutive units into chunks up to ~max_chars (rule-based fallback chunker)."""
    chunks: list[str] = []
    cur = ""
    for p in units:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur); cur = p
        else:
            cur += p
    if cur:
        chunks.append(cur)
    return chunks or [""]


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Rule-based chunking: atomic units greedily packed up to ~max_chars (~24s @170)."""
    return _greedy_group(_atomic_units(text, max_chars), max_chars)


def _fm_call(messages: list[dict], model: str, max_tokens: int = 4000) -> str:
    """Call a Databricks FM API chat endpoint (OpenAI-compatible) and return the content."""
    host, token = _host_token()
    url = f"{host}/serving-endpoints/{model}/invocations"
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["message"]["content"]


def _fm_group(text: str, max_chars: int, model: str) -> list[str]:
    """Semantic chunking via LLM. We split into atomic units ourselves (reliable) and let the
    LLM only decide how to GROUP consecutive units at natural boundaries — it returns index
    groups, never text, so the spoken content is guaranteed verbatim. Raises on any anomaly
    (caller falls back to rule-based)."""
    units = _atomic_units(text, max_chars)
    if len(units) <= 1:
        return units
    listing = "\n".join(f"{i}: ({len(u)}字) {u}" for i, u in enumerate(units))
    system = ("あなたは日本語ナレーション原稿を音声合成用に区切る編集者です。"
              "文の断片(units)を、意味のまとまり（話題・節の切れ目）で連続グループにまとめます。")
    user = (
        f"次の断片を連続するグループにまとめてください。\n制約:\n"
        f"- 各グループの合計文字数は {max_chars} 字以下。できるだけ {max_chars} 字に近づけ、グループ数を最小化する。\n"
        f"- 番号 0..{len(units) - 1} を順序どおり、重複なく、過不足なく全て使う。\n"
        f"- 断片を分割・改変しない（グループ化のみ）。話題が変わる位置で区切るのが望ましい。\n"
        f'- 出力はJSONのみ: {{"groups": [[0,1],[2,3,4], ...]}}\n\n断片:\n{listing}'
    )
    content = _fm_call([{"role": "system", "content": system}, {"role": "user", "content": user}], model)
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError("LLM returned no JSON")
    groups = json.loads(m.group(0)).get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("LLM returned no groups")
    flat = [i for g in groups for i in g]
    if flat != list(range(len(units))):  # must be an exact, ordered partition → integrity
        raise ValueError(f"invalid partition: {flat} vs 0..{len(units) - 1}")
    chunks: list[str] = []
    for g in groups:
        c = "".join(units[i] for i in g)
        chunks.extend(_greedy_group(_atomic_units(c, max_chars), max_chars) if len(c) > max_chars else [c])
    return chunks


def _concat_wavs(b64_list: list[str], gap_s: float = 0.14, fade_ms: float = 12) -> tuple[str, float]:
    """Concatenate mono 16-bit WAVs with a short silence gap. A few-ms linear fade on each
    clip edge removes the click caused by cutting at a non-zero sample."""
    import array
    clips, rate, width, ch = [], None, None, None
    for b64 in b64_list:
        if not b64:
            continue
        w = wave.open(io.BytesIO(base64.b64decode(b64)))
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        a = array.array("h"); a.frombytes(w.readframes(w.getnframes()))
        clips.append(a)
    if not clips:
        raise RuntimeError("no audio produced")

    fade = int(fade_ms / 1000.0 * rate) * ch  # samples to ramp at each edge
    for a in clips:
        n = len(a)
        f = min(fade, n // 2)
        for i in range(f):
            g = i / f
            a[i] = int(a[i] * g)
            a[n - 1 - i] = int(a[n - 1 - i] * g)

    gap = array.array("h", [0] * (int(gap_s * rate) * ch))
    out = array.array("h")
    for i, a in enumerate(clips):
        if i:
            out.extend(gap)
        out.extend(a)

    buf = io.BytesIO()
    ww = wave.open(buf, "wb")
    ww.setnchannels(ch); ww.setsampwidth(width); ww.setframerate(rate)
    ww.writeframes(out.tobytes()); ww.close()
    return base64.b64encode(buf.getvalue()).decode("ascii"), round(len(out) / ch / rate, 2)


def handle_tts(req: dict) -> dict:
    text = str(req.get("text", "")).strip()
    params = dict(req.get("params") or {})
    max_chars = int(req.get("max_chars", 170))
    chunk_note = ""
    if not req.get("split"):
        segs = [text]
    elif req.get("chunk_mode", "semantic") == "semantic" and len(text) > max_chars:
        model = req.get("fm_model") or "databricks-claude-haiku-4-5"
        try:
            segs = _fm_group(text, max_chars, model)
            chunk_note = f"分割: セマンティック(LLM {model}) → {len(segs)}チャンク"
        except Exception as e:  # any LLM/parse/validation failure → safe rule-based fallback
            segs = _chunk_text(text, max_chars)
            chunk_note = f"分割: LLM失敗のため文字数ベースにフォールバック ({type(e).__name__}: {e}) → {len(segs)}チャンク"
    else:
        segs = _chunk_text(text, max_chars)
        chunk_note = f"分割: 文字数ベース → {len(segs)}チャンク"

    # Voice consistency: with --no-ref, each sentence samples a NEW random voice, so a
    # concatenated multi-sentence clip drifts in timbre. Fix = anchor every sentence to ONE
    # reference (the model's speaker/voice-cloning path). If the caller didn't supply
    # ref_wav_b64, we synthesize a reference from the first sentence (no_ref + voice_seed),
    # then condition every sentence on it with a fixed seed.
    fix_voice = bool(req.get("fix_voice", True)) and len(segs) > 1
    ref_note = ""
    if fix_voice:
        voice_seed = int(req.get("voice_seed", 42))
        ref_b64 = params.get("ref_wav_b64")
        if not ref_b64:
            ref_params = {k: v for k, v in params.items() if k != "ref_wav_b64"}
            ref_params.update({"no_ref": True, "seed": voice_seed})
            ref_pred = _invoke([{"text": segs[0], "params_json": json.dumps(ref_params, ensure_ascii=False)}])[0]
            if ref_pred.get("error"):
                raise RuntimeError("reference synth failed: " + ref_pred["error"])
            ref_b64 = ref_pred["audio_wav_b64"]
            ref_note = f"声を固定: 第1文から参照音声を生成 (voice_seed={voice_seed})"
        else:
            ref_note = "声を固定: 指定の参照音声(ref_wav_b64)を全文に適用"
        params.pop("no_ref", None)
        params["ref_wav_b64"] = ref_b64
        params.setdefault("seed", voice_seed)  # fixed seed → stable prosody too

    pj = json.dumps(params, ensure_ascii=False)
    records = [{"text": s, "params_json": pj} for s in segs]
    preds = _invoke(records)

    errs = [p.get("error") for p in preds if p.get("error")]
    if errs and all(p.get("error") for p in preds):
        raise RuntimeError(errs[0])
    b64s = [p.get("audio_wav_b64", "") for p in preds]
    audio_b64, total_audio = _concat_wavs(b64s)
    total_synth = round(sum(float(p.get("synth_seconds", 0) or 0) for p in preds), 2)
    notes = []
    for i, p in enumerate(preds):
        msg = p.get("messages")
        if msg and msg not in ("[]", ""):
            notes.append(f"[{i+1}] {msg}")
        if p.get("error"):
            notes.append(f"[{i+1}] ERROR {p['error']}")
    if ref_note:
        notes.insert(0, ref_note)
    if chunk_note:
        notes.insert(0, chunk_note)
    return {"audio_wav_b64": audio_b64, "segments": len(segs),
            "total_audio_seconds": total_audio, "total_synth_seconds": total_synth,
            "notes": "\n".join(notes)}


def handle_params() -> dict:
    preds = _invoke([{"text": "", "params_json": json.dumps({"__list_params__": True})}])
    msg = preds[0].get("messages", "{}")
    try:
        return json.loads(msg)
    except Exception:
        return {"supported_params": [], "raw": msg}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        sys.stderr.write("[app] " + (a[0] % a[1:]) + "\n")

    def _json(self, code, obj):
        blob = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.replace("%ENDPOINT%", CFG["endpoint"]).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/params":
            try:
                self._json(200, handle_params())
            except Exception as e:
                self._json(500, {"error": str(e)})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/tts":
            self.send_error(404); return
        n = int(self.headers.get("Content-Length", 0))
        try:
            out = handle_tts(json.loads(self.rfile.read(n) or b"{}"))
            self._json(200, out)
        except Exception as e:
            self._json(500, {"error": str(e)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--profile", default=CFG["profile"])
    ap.add_argument("--endpoint", default=CFG["endpoint"])
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    CFG["profile"], CFG["endpoint"] = args.profile, args.endpoint
    url = f"http://127.0.0.1:{args.port}"
    print(f"[app] Irodori-TTS UI → {url}")
    print(f"[app] endpoint={CFG['endpoint']} profile={CFG['profile']}")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
