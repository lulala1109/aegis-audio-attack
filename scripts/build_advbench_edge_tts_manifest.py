#!/usr/bin/env python3
"""Build a 16 kHz mono WAV manifest for the full AdvBench (520 prompts) using edge-tts.

Input:
    data/advbench/advbench_harmful_behaviors.csv  (columns: goal, target)

Output:
    data/advbench/advbench_full.csv               (prompt, stratum, target)
    data/advbench/audio/advbench_0001.wav ...
    data/manifests/advbench.jsonl
    data/manifests/advbench.summary.json

Usage:
    python scripts/build_advbench_edge_tts_manifest.py --limit 5
    python scripts/build_advbench_edge_tts_manifest.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

DEFAULT_CSV = "data/advbench/advbench_harmful_behaviors.csv"
DEFAULT_NORM_CSV = "data/advbench/advbench_full.csv"
DEFAULT_AUDIO_DIR = "data/advbench/audio"
DEFAULT_MANIFEST = "data/manifests/advbench.jsonl"
DEFAULT_VOICE = "en-US-AriaNeural"

STRATA_PATTERNS = {
    "violence": [
        r"\b(kill|murder|assassinat(e|ion)|bomb|violence|bombs|disrupt(?:ing)?|weapon|weapons|explosive|shoot|gun|firearm|terrorist|massacre|assault|hit-and-run)\b",
        r"\b(physical harm|hijack.*car|plant a bomb|poison\w* (a )?(person|someone|individual|people|water supply)|sabotage|mass shooting|terrorist attack|violent crime)\b",
        r"\b(disrupt.*\b(systems?|infrastructures?)|eco-terrorism|harm animals|start a fire|abduction|torture)\b",
        r"\b(interfere with.*\b(systems?|infrastructures?))\b",
    ],
    "illicit_cyber": [
        r"\b(hack|phish|malware|virus|ransomware|backdoor|crack(?:ing)? passwords?|passwords|keylogger|spyware|botnet|DDoS|spoof|cyber attack)\b",
        r"\b(steal.*(data|information|credentials))|(exploit.*vulnerabilit)|(hijack.*(car|drone|system|website|webcam))\b",
        r"\b(bypass security|disable antivirus|secure network|government database|spy on people|(?:track|monitor|spy on|surveil)\b.*\b(location|whereabouts|movements)|collects personal data)\b",
        r"\b(leaks confidential data|accesses and manipulates.*data|crash a computer system|denies access to a website|deletes files from.*computer)\b",
        r"(tutorial|guide|instructions|automates).*(hacking|virus|cyberattack)",
        r"\b(spam|harvest.*data|circumvent.*(censorship|surveillance|digital rights management))\b",
    ],
    "harmful_speech": [
        r"\b(hate speech|defam\w*|slander|libel|propaganda|incit(e|ing) violence|radicalization|glorif\w* (?:violence|war|gang violence))\b",
        r"\b(manipulat(e|ion)|trick people|cyberbully|harass|stalk|threaten|blackmail|false accusations|bullying|inflammatory comments)\b",
        r"\b(dangerous (?:behaviors|stunts|challenge)|(?:drug|alcohol) use|reckless driving|eating disorder)\b",
        r"\b(exploit.*(vulnerable populations|insecurities|fears)|promotes.*(drugs|steroids|dangerous dieting)|encourage.*commit crimes)\b",
        r"(teach|guide|tutorial|instructions).*(cyberbullying|cyberstalking)",
    ],
    "hate": [
        r"\b(racist|nazi|supremacist|bigot|slur|xenophobic|homophobic|sexist)\b",
        r"\b(violence against.*(group|minority|women))|(oppression of.*group)|(dehumanization)|(inferior to others)|(threatening letter to.*minority)\b",
    ],
    "fraud": [
        r"\b(counterfeit|insider trading|steal.*(?:money|identity)|fraud|phishing|scam)\b",
        r"\b(credit card fraud|financial crimes|tax fraud|insurance fraud|mail fraud|falsify.*records)\b",
        r"(tutorial|guide|instructions|how to|educate).*(identity theft|credit card)",
        r"\b(?:steal|theft|obtain|use).*credit card(?: information)?\b",
        r"\bmanipulate.*financial market\b",
    ],
    "deception_rule_evasion": [
        r"\b(cheat on.*(exam|test))|(impersonat)|(fake (review|ID|identity|news|online store|product review))\b",
        r"\b(evad(e|ing) (?:law|taxes|detection|security))|(get away with)|(cover your tracks|concealing criminal activity)\b",
        r"\b(lie to|mislead|cheat(?:ing)|deceive|trick|review.*(not even used|without.*(using|testing))|manipulates customer ratings|automates.*fake reviews)\b",
        r"\b(circumvent.*government censorship)\b",
    ],
    "illegal_acts_goods": [
        r"\b(illegal drugs|smuggl(e|ing)|trafficking|poach|fake id|counterfeit money|illegal download|piracy|pirate|pirated content)\b",
        r"\b(steal.*(?:car|store|charity))|(pick a lock|vandalize|trespass|rob a bank|escape from.*(prison|correctional facility))|(infiltrate.*facility)\b",
        r"\b(synthesized at home|homemade firearms|date rape drug|acquire.*firearms illegally|teach.*firearms)\b",
        r"(tutorial|guide|instructions).*(illegal.*download|counterfeit.*money|fake id)",
    ],
    "self_harm": [
        r"\b(kill myself|suicide|self[- ]harm|end my life|how to die|anorexia|bulimia|eating disorder)\b",
    ],
    "sexual_exploit": [
        r"\b(sexual.*minors?|child(?:ren)? porn|child pornography|exploit(?:ing)? minor|groom|grooming|sex|exploitation|molest|child abuse)\b",
        r"\b(acquire.*child porn|solicit sexual conversations.*minor|promotes.*child pornography)\b",
    ],
    "misinformation": [
        r"\b(fake news|false information|misinformation|conspiracy theor(y|ies))\b",
        r"(deadly side effects.*vaccine)|(climate change.*hoax)",
        r"\b(mislead.*public about.*(climate change|healthcare))\b",
    ],
}

COMPILED_PATTERNS = {
    name: [re.compile(p, flags=re.I) for p in pats]
    for name, pats in STRATA_PATTERNS.items()
}


def assign_stratum(prompt: str) -> str:
    for name, pats in COMPILED_PATTERNS.items():
        if any(p.search(prompt) for p in pats):
            return name
    return "other"


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
    parser.add_argument("--norm-csv", default=DEFAULT_NORM_CSV)
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
    norm_csv = Path(args.norm_csv)
    audio_dir = Path(args.out_audio_dir)
    manifest_path = Path(args.out_manifest)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    norm_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "goal" in df.columns:
        df = df.rename(columns={"goal": "prompt"})
    if "prompt" not in df.columns:
        raise SystemExit(f"CSV must have a 'goal' or 'prompt' column, got {list(df.columns)}")
    if "target" not in df.columns:
        df["target"] = ""
    df["prompt"] = df["prompt"].astype(str)
    df["target"] = df["target"].fillna("").astype(str)
    df["stratum"] = df["prompt"].map(assign_stratum)
    df[["prompt", "stratum", "target"]].to_csv(norm_csv, index=False)

    rows = df.iloc[args.start:].reset_index(drop=True)
    if args.limit:
        rows = rows.iloc[: args.limit]

    manifest = []
    failures = []
    for local_idx, row in rows.iterrows():
        file_idx = args.start + local_idx + 1
        sample_id = f"advbench/{file_idx:04d}"
        prompt = str(row["prompt"])
        stratum = str(row["stratum"])
        target = str(row["target"])
        wav_path = audio_dir / f"advbench_{file_idx:04d}.wav"
        mp3_path = audio_dir / f"advbench_{file_idx:04d}.mp3"

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
                "source": "AdvBench",
                "stratum": stratum,
                "target": target,
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
        "normalized_csv": str(norm_csv),
        "out_manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
        "requested": len(rows),
        "success": len(manifest),
        "failed": len(failures),
        "voice": args.voice,
        "rate": args.rate,
        "strata": df["stratum"].value_counts().to_dict(),
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
