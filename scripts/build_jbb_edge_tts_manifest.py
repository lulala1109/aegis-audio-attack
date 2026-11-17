#!/usr/bin/env python3
"""Build a 16 kHz mono WAV manifest for JailbreakBench using edge-tts.

edge-tts uses Microsoft's speech endpoint, which is reachable from this server
(unlike Google TTS used by gTTS).

Input:
    data/jbb/jbb_full.csv  (columns: prompt, stratum)

Output:
    data/jbb/audio/jbb_0001.wav ...
    data/manifests/jbb_harmful.jsonl
    data/manifests/jbb_harmful.summary.json

Usage:
    python scripts/build_jbb_edge_tts_manifest.py --limit 5
    python scripts/build_jbb_edge_tts_manifest.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

DEFAULT_CSV = "data/jbb/jbb_full.csv"
DEFAULT_AUDIO_DIR = "data/jbb/audio"
DEFAULT_MANIFEST = "data/manifests/jbb_harmful.jsonl"
DEFAULT_VOICE = "en-US-AriaNeural"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path(sys.executable).with_name("ffmpeg")
    if candidate.exists():
        return str(candidate)
    raise SystemExit(
        "ffmpeg not found. Install it first, e.g. "
        "`conda install -c conda-forge ffmpeg`.")


def convert_to_wav(mp3: Path, wav: Path, ffmpeg: str) -> None:
    cmd = [
        ffmpeg, "-y", "-i", str(mp3),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-1000:])


async def synthesize(text: str, mp3_path: Path, voice: str, rate: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(str(mp3_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out-audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()
    try:
        import edge_tts  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"edge-tts is not installed ({exc}). Run: pip install edge-tts")

    csv_path = Path(args.csv)
    audio_dir = Path(args.out_audio_dir)
    manifest_path = Path(args.out_manifest)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns:
        raise SystemExit(f"CSV must have a 'prompt' column, got {list(df.columns)}")
    if "stratum" not in df.columns:
        df["stratum"] = "unknown"

    rows = df.iloc[args.start:].reset_index(drop=True)
    if args.limit:
        rows = rows.iloc[: args.limit]

    manifest = []
    failures = []
    for local_idx, row in rows.iterrows():
        file_idx = args.start + local_idx + 1
        sample_id = f"jbb/{file_idx:04d}"
        prompt = str(row["prompt"])
        stratum = str(row["stratum"])
        wav_path = audio_dir / f"jbb_{file_idx:04d}.wav"
        mp3_path = audio_dir / f"jbb_{file_idx:04d}.mp3"

        try:
            if not wav_path.exists() or args.overwrite:
                last_error = None
                for attempt in range(1, args.retries + 1):
                    try:
                        asyncio.run(synthesize(
                            prompt, mp3_path, args.voice, args.rate))
                        convert_to_wav(mp3_path, wav_path, ffmpeg)
                        last_error = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        print(
                            f"[retry] {sample_id} attempt={attempt}/{args.retries}: {exc}",
                            file=sys.stderr)
                        time.sleep(min(2 ** attempt, 10))
                if last_error is not None:
                    raise last_error
                if mp3_path.exists():
                    mp3_path.unlink()

            info = sf.info(str(wav_path))
            if info.samplerate != 16000 or info.channels != 1:
                raise RuntimeError(
                    f"bad WAV format: sr={info.samplerate}, channels={info.channels}")

            manifest.append({
                "id": sample_id,
                "local_audio": str(wav_path.resolve()),
                "prompt": prompt,
                "source": "JailbreakBench",
                "stratum": stratum,
                "tts_engine": "edge-tts",
                "tts_voice": args.voice,
                "tts_rate": args.rate,
                "audio_sha256": sha256_file(wav_path),
                "duration_seconds": float(info.duration),
            })
            print(f"[OK] {sample_id} {info.duration:.2f}s {prompt[:70]!r}")
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "id": sample_id,
                "prompt": prompt,
                "stratum": stratum,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[FAIL] {sample_id}: {type(exc).__name__}: {exc}", file=sys.stderr)

        if args.sleep > 0:
            time.sleep(args.sleep)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "csv": str(csv_path),
        "out_manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
        "requested": len(rows),
        "success": len(manifest),
        "failed": len(failures),
        "voice": args.voice,
        "rate": args.rate,
        "failures": failures,
    }
    summary_path = manifest_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"\n[OK] manifest: {manifest_path}")
    print(f"[OK] summary:  {summary_path}")
    print(f"[INFO] success={len(manifest)} failed={len(failures)}")


if __name__ == "__main__":
    main()
