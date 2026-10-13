#!/usr/bin/env python3
"""Google Slides のスピーカーノート → TTS → ナレーション付きMP4 を自動生成する PoC。

オプションで、本番デッキを Drive でコピーし、コピー側のスピーカーノートを LLM(FM API) で校正
（難読漢字→カタカナ、古い/重複した内容を削除して最新版のみ）してから音声化できる。本番は無変更。

各スライドのノートを Irodori-TTS で音声化（デッキ全体で同一のナレーター声＝1参照音声を使い回し）、
スライド画像と ffmpeg で合成して1本の解説MP4を作る。長いノートはセマンティック分割＋フェード連結。

使い方:
    # 本番をコピー→校正→先頭60ページをMP4化
    python slides_narrate.py --url "<本番URL>" --copy --clean --pages 60 --out ../outputs/slides_narration/deck60.mp4
    # 既存デッキをそのまま（校正なし）でMP4化
    python slides_narrate.py --pid <ID> --pages 3

依存: ffmpeg, Google 認証(google-auth スキル済), tts_app.py（同ディレクトリ）。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tts_app  # reuse _invoke / _fm_group / _chunk_text / _concat_wavs / _fm_call / CFG

# Path to a helper that prints a Google OAuth access token on `token`. Override with the
# GOOGLE_AUTH_SCRIPT env var. See README.md ("Apps") for how to obtain a token.
AUTH = os.environ.get("GOOGLE_AUTH_SCRIPT", os.path.expanduser(
    "~/.vibe/marketplace/plugins/fe-google-tools/skills/google-auth/resources/google_auth.py"))
QUOTA = "gcp-sandbox-field-eng"
REF_PHRASE = "皆さん、こんにちは。本日はどうぞよろしくお願いいたします。"  # 参照音声（声の基準）用の固定フレーズ
VOICE_SEED = 42
NUM_STEPS = 40
MAX_CHARS = 170
CHUNK_MODEL = "databricks-claude-haiku-4-5"      # セマンティック分割用（速い）
CLEAN_MODEL = "databricks-claude-sonnet-4-5"     # ノート校正用（品質重視）

CLEAN_SYSTEM = ("あなたは日本語ナレーション原稿の校正者です。音声合成(TTS)エンジンが誤読しないよう、"
                "読み方が難しい漢字語をカタカナ（読み）に置き換えます。")


def _token() -> str:
    return subprocess.run(["python3", AUTH, "token"], capture_output=True, text=True, timeout=60).stdout.strip()


def _api(method: str, url: str, tok: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {tok}", "x-goog-user-project": QUOTA,
        "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def copy_presentation(pid: str, tok: str, name: str) -> str:
    d = _api("POST", f"https://www.googleapis.com/drive/v3/files/{pid}/copy", tok, {"name": name})
    return d["id"]


def read_notes(pid: str, tok: str) -> list[tuple[str, str, str]]:
    """Return [(slideObjectId, speakerNotesObjectId, notes_text), ...] for every slide, in order."""
    fields = ("slides(objectId,slideProperties(notesPage(pageElements(objectId,"
              "shape(text(textElements(textRun(content))))),notesProperties(speakerNotesObjectId))))")
    d = _api("GET", f"https://slides.googleapis.com/v1/presentations/{pid}?fields={urllib.parse.quote(fields)}", tok)
    out = []
    for s in d.get("slides", []):
        np = s.get("slideProperties", {}).get("notesPage", {})
        spid = np.get("notesProperties", {}).get("speakerNotesObjectId")
        note = ""
        for pe in np.get("pageElements", []):
            if pe.get("objectId") == spid:
                for te in pe.get("shape", {}).get("text", {}).get("textElements", []):
                    if "textRun" in te:
                        note += te["textRun"].get("content", "")
        out.append((s.get("objectId"), spid, note.strip()))
    return out


def clean_note(note: str) -> str:
    user = ("次のスピーカーノートを整えてください。\n"
        "【カタカナ化のルール】誤読の可能性が少しでもある語は積極的にカタカナ（読み）へ置換する。対象:\n"
        "・人名/固有名詞（例: 山田太郎→ヤマダタロウ）\n"
        "・専門用語・四字熟語・熟語（例: 一気通貫→イッキツウカン、独自→ドクジ、領域→リョウイキ、施策→シサク、誠に→マコトに、兼ね備えた→カネソナえた）\n"
        "・複数の読みがある語や難読語。\n"
        "・アルファベット表記の英単語・製品名・略語・頭字語は必ずカタカナ読みに置換する"
        "（例: Agent Bricks→エージェントブリックス、Databricks→データブリックス、Gartner→ガートナー、"
        "Unity Catalog→ユニティカタログ、Genie→ジニー、Claude Code→クロードコード、Codex→コーデックス、"
        "Cursor→カーソル、MLflow→エムエルフロー、LLM→エルエルエム、AI→エーアイ、ML→エムエル、API→エーピーアイ、"
        "SQL→エスキューエル、RAG→ラグ、Choice→チョイス、Context→コンテキスト、Control→コントロール）。\n"
        "ひらがな・カタカナ・数字・ごく平易な常用語（本日/皆さん/最新/内容/会社 等）はそのまま。\n"
        "【古い内容】ノート内に古い版や重複した下書きが残っていれば、最新の1版だけ残し古い内容は削除。\n"
        "【厳守】意味・言い回し・句読点は変えない。要約や新情報の追加は禁止。カタカナ化と古い内容の削除のみ。\n"
        "出力は整えた本文のみを <note></note> で囲む。前置きや説明は書かない。\n\n"
        f"ノート:\n<<<\n{note}\n>>>")
    out = tts_app._fm_call([{"role": "system", "content": CLEAN_SYSTEM},
                            {"role": "user", "content": user}], CLEAN_MODEL, max_tokens=4000)
    m = re.search(r"<note>(.*?)</note>", out, re.S)
    return (m.group(1) if m else out).strip()


def write_notes(pid: str, tok: str, edits: list[tuple[str, str, str]]) -> None:
    """edits = [(speakerNotesObjectId, old_note, new_note)]. Replace notes text in batches."""
    reqs = []
    for spid, old, new in edits:
        if not spid or not new:
            continue
        if old:
            reqs.append({"deleteText": {"objectId": spid, "textRange": {"type": "ALL"}}})
        reqs.append({"insertText": {"objectId": spid, "insertionIndex": 0, "text": new}})
    for i in range(0, len(reqs), 100):  # chunk to keep each batchUpdate modest
        _api("POST", f"https://slides.googleapis.com/v1/presentations/{pid}:batchUpdate", tok,
             {"requests": reqs[i:i + 100]})


def slide_png(pid: str, slide_oid: str, tok: str, dest: str) -> None:
    d = _api("GET", f"https://slides.googleapis.com/v1/presentations/{pid}/pages/{slide_oid}/thumbnail"
             f"?thumbnailProperties.thumbnailSize=LARGE&thumbnailProperties.mimeType=PNG", tok)
    with urllib.request.urlopen(d["contentUrl"], timeout=60) as r:
        open(dest, "wb").write(r.read())


def _invoke_retry(recs: list[dict], tries: int = 4):
    """Endpoint calls can transiently hang/time out; retry with backoff so one bad call
    doesn't abort a 60-slide job."""
    import time
    for k in range(tries):
        try:
            return tts_app._invoke(recs)
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"    retry {k+1}/{tries-1} after {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * (k + 1))


def synth_note(note: str, ref_b64: str) -> str:
    try:
        chunks = tts_app._fm_group(note, MAX_CHARS, CHUNK_MODEL)
    except Exception:
        chunks = tts_app._chunk_text(note, MAX_CHARS)
    recs = [{"text": c, "params_json": json.dumps(
        {"num_steps": NUM_STEPS, "ref_wav_b64": ref_b64, "seed": VOICE_SEED})} for c in chunks]
    preds = _invoke_retry(recs)
    if all(p.get("error") for p in preds):
        raise RuntimeError(next(p["error"] for p in preds if p.get("error")))
    b64, _ = tts_app._concat_wavs([p.get("audio_wav_b64", "") for p in preds])
    return b64


def silent_wav_b64(seconds: float = 1.5, rate: int = 48000) -> str:
    buf = io.BytesIO()
    w = wave.open(buf, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(b"\x00\x00" * int(seconds * rate)); w.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_segment(png: str, wav: str, out_mp4: str) -> None:
    vf = ("scale=1600:900:force_original_aspect_ratio=decrease,"
          "pad=1600:900:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1,fps=25")
    subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", png, "-i", wav, "-vf", vf,
                    "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out_mp4],
                   check=True, capture_output=True)


def concat(segs: list[str], out_mp4: str) -> None:
    # concat via a file list with stream-copy is fragile across many inputs; use the concat
    # demuxer with re-encode for robustness.
    lst = out_mp4 + ".txt"
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{os.path.abspath(s)}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out_mp4],
                   check=True, capture_output=True)
    os.remove(lst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid"); ap.add_argument("--url")
    ap.add_argument("--pages", type=int, default=0, help="先頭から何ページ処理するか（0=全部）")
    ap.add_argument("--copy", action="store_true", help="本番をDriveでコピーしてから処理（本番は無変更）")
    ap.add_argument("--clean", action="store_true", help="ノートをLLMで校正（難読漢字→カタカナ、古い内容削除）")
    ap.add_argument("--copy-name", default="【TTSナレーション用コピー】")
    ap.add_argument("--out", default="../outputs/slides_narration/narrated.mp4")
    args = ap.parse_args()
    src = args.pid or re.search(r"/presentation/d/([\w-]+)", args.url or "").group(1)

    workdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(workdir, exist_ok=True)
    tok = _token()

    pid = src
    if args.copy:
        pid = copy_presentation(src, tok, args.copy_name)
        print(f"copied deck → https://docs.google.com/presentation/d/{pid}/edit", flush=True)

    notes = read_notes(pid, tok)
    if args.pages:
        notes = notes[:args.pages]
    print(f"processing {len(notes)} slides", flush=True)

    if args.clean:
        print("cleaning speaker notes (kanji/英字→katakana, drop stale content)…", flush=True)
        edits, cleaned, changed = [], [], 0
        for i, (oid, spid, note) in enumerate(notes):
            cn = clean_note(note) if note.strip() else note
            cleaned.append(cn)
            edits.append((spid, note, cn))
            if cn != note:  # note text changed → invalidate cached audio so it re-synths
                changed += 1
                wav = os.path.join(workdir, f"slide{i+1:02d}.wav")
                if os.path.exists(wav):
                    os.remove(wav)
            if i < 2:
                print(f"  [slide {i+1}] before: {note[:60]}…\n            after : {cn[:60]}…", flush=True)
            elif (i + 1) % 10 == 0:
                print(f"  cleaned {i+1}/{len(notes)} ({changed} changed so far)", flush=True)
        print(f"  {changed}/{len(notes)} slides changed → will re-synthesize those", flush=True)
        write_notes(pid, tok, edits)
        json.dump([{"slide": i + 1, "note": c} for i, c in enumerate(cleaned)],
                  open(os.path.join(workdir, "notes_cleaned.json"), "w"), ensure_ascii=False, indent=2)
        notes = [(notes[i][0], notes[i][1], cleaned[i]) for i in range(len(notes))]
        print("notes written back to the copy ✓", flush=True)

    # Shared reference voice — cached so a resume reuses the EXACT same voice as prior slides.
    ref_path = os.path.join(workdir, "_ref.wav")
    if os.path.exists(ref_path) and os.path.getsize(ref_path) > 1000:
        ref = base64.b64encode(open(ref_path, "rb").read()).decode("ascii")
        print("reusing cached reference voice", flush=True)
    else:
        print("generating shared reference voice…", flush=True)
        ref = _invoke_retry([{"text": REF_PHRASE, "params_json": json.dumps(
            {"no_ref": True, "seed": VOICE_SEED, "num_steps": NUM_STEPS})}])[0]["audio_wav_b64"]
        open(ref_path, "wb").write(base64.b64decode(ref))

    segs = []
    for i, (oid, spid, note) in enumerate(notes):
        png = os.path.join(workdir, f"slide{i+1:02d}.png")
        wav = os.path.join(workdir, f"slide{i+1:02d}.wav")
        seg = os.path.join(workdir, f"seg{i+1:02d}.mp4")
        if os.path.exists(wav) and os.path.getsize(wav) > 1000:  # resume: reuse done slides
            dur = wave.open(wav).getnframes() / 48000
            if not os.path.exists(png):
                slide_png(pid, oid, tok, png)
            make_segment(png, wav, seg)
            segs.append(seg)
            print(f"  slide {i+1}/{len(notes)}: reuse ({dur:.1f}s)", flush=True)
            continue
        slide_png(pid, oid, tok, png)
        b64 = synth_note(note, ref) if note.strip() else silent_wav_b64()
        open(wav, "wb").write(base64.b64decode(b64))
        dur = wave.open(wav).getnframes() / 48000
        make_segment(png, wav, seg)
        segs.append(seg)
        print(f"  slide {i+1}/{len(notes)}: {len(note)}字 → {dur:.1f}s", flush=True)

    concat(segs, os.path.abspath(args.out))
    total = sum(wave.open(os.path.join(workdir, f"slide{i+1:02d}.wav")).getnframes() / 48000
                for i in range(len(notes)))
    for i in range(len(notes)):  # tidy intermediate segments
        try:
            os.remove(os.path.join(workdir, f"seg{i+1:02d}.mp4"))
        except OSError:
            pass
    print(f"\n✅ {args.out}  ({len(segs)} slides, ~{total/60:.1f} min)", flush=True)
    if args.copy:
        print(f"   copy: https://docs.google.com/presentation/d/{pid}/edit", flush=True)


if __name__ == "__main__":
    main()
