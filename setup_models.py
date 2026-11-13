#!/usr/bin/env python3
"""Download and stage every offline model into ./models (run once, needs internet).

Components staged:
  1. models/silero_vad.onnx            Silero VAD (voice activity detection)
  2. models/zipformer/                 sherpa-onnx streaming Zipformer EN (int8)
                                       + one real-speech test WAV
  3. models/voxlock/embedding.onnx     speaker-embedding network for VoxLock
  4. models/whisper/<size>/            faster-whisper CTranslate2 model (default small)

Every download has fallback mirrors.  After this script completes, the app runs
with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and never touches the network.

Usage:
    python setup_models.py                    # download everything (whisper small)
    python setup_models.py --whisper base     # lighter/faster whisper variant
    python setup_models.py --verify-only      # check staged files, no download
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# setup_models.py runs BEFORE offline mode exists -- explicitly re-enable network.
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.pop(_k, None)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

GITHUB = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

SILERO_URLS = [
    f"{GITHUB}/asr-models/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
]

ZIPFORMER_URLS = [
    f"{GITHUB}/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    f"{GITHUB}/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
]

SPEAKER_URLS = [
    # sherpa-onnx speaker-embedding ONNX mirrors (the old GitHub
    # speaker-recognition-models release tag was removed; these are the
    # current hosted files).
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/nemo_en_titanet_small.onnx",
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/wespeaker_en_voxceleb_resnet34.onnx",
]

# VoxLock: true target-speaker waveform extraction (enroll + chunk networks).
VOXLOCK_URLS = [
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock.tar.bz2",
]

WHISPER_REPO = "Systran/faster-whisper-{size}"


# ---------------------------------------------------------------------------
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 3) -> Path:
    """Download with progress, resume support and mirror fallback."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + f".part{attempt}")
        existing = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "VoxStream-setup/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "ab") as f:
                total = resp.headers.get("Content-Length")
                total = int(total) + existing if total else None
                start = time.time()
                got = existing
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    got += len(block)
                    if total:
                        pct = 100.0 * got / total
                        speed = got / max(0.1, time.time() - start) / 1e6
                        print(f"\r  {dest.name}: {pct:5.1f}%  ({got/1e6:8.1f} MB, "
                              f"{speed:5.1f} MB/s)", end="", flush=True)
            print()
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"\n  attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(2.0)
    raise RuntimeError(f"download failed: {url}")


def safe_extract_tarbz2(archive: Path, dest: Path) -> None:
    """Extract a .tar.bz2 rejecting path traversal / absolute entries / links."""
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            name = member.name
            if member.issym() or member.islnk():
                continue  # archives we consume never need links
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest)):
                raise ValueError(f"refusing unsafe archive path: {name}")
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def stage_silero() -> dict:
    dest = MODELS_DIR / "silero_vad.onnx"
    if dest.exists() and dest.stat().st_size > 500_000:
        print(f"[silero] present: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return {"path": str(dest.relative_to(ROOT)), "sha256": sha256(dest)}
    print("[silero] downloading Silero VAD...")
    last_error = None
    for url in SILERO_URLS:
        try:
            download(url, dest)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return {"path": str(dest.relative_to(ROOT)), "sha256": sha256(dest)}


def stage_zipformer() -> dict:
    zdir = MODELS_DIR / "zipformer"
    if (zdir / "tokens.txt").exists() and list(zdir.glob("*encoder*.onnx")):
        print("[zipformer] present")
        return {"path": str(zdir.relative_to(ROOT))}
    print("[zipformer] downloading streaming Zipformer (EN, ~250 MB)...")
    last_error = None
    for url in ZIPFORMER_URLS:
        try:
            with tempfile.TemporaryDirectory(prefix="voxstream-dl-") as tmp:
                archive = Path(tmp) / "model.tar.bz2"
                download(url, archive)
                extract_dir = Path(tmp) / "extracted"   # untrusted archive: own empty dir
                safe_extract_tarbz2(archive, extract_dir)
                zdir.mkdir(parents=True, exist_ok=True)
                onnx_files = list(extract_dir.rglob("*.onnx"))
                tokens = list(extract_dir.rglob("tokens.txt"))
                test_wavs = sorted(extract_dir.rglob("test_wavs/*.wav"))
                if not onnx_files or not tokens:
                    raise RuntimeError(f"unexpected archive layout for {url}")
                for kind in ("encoder", "decoder", "joiner"):
                    cands = [f for f in onnx_files if kind in f.name.lower()]
                    if not cands:
                        raise RuntimeError(f"no {kind} onnx in archive")
                    int8 = [f for f in cands if "int8" in f.name.lower()]
                    chosen = (int8 or cands)[0]
                    shutil.copyfile(chosen, zdir / f"{kind}.onnx")
                    print(f"  staged {kind}: {chosen.name} -> {kind}.onnx "
                          f"({zdir.joinpath(kind + '.onnx').stat().st_size/1e6:.1f} MB)")
                shutil.copyfile(tokens[0], zdir / "tokens.txt")
                if test_wavs:
                    shutil.copyfile(test_wavs[0], zdir / "test_wav.wav")
                    if len(test_wavs) > 1:
                        shutil.copyfile(test_wavs[1], zdir / "test_wav2.wav")
                break
        except Exception as exc:
            last_error = exc
            print(f"  source failed: {exc}")
    if last_error:
        raise last_error
    return {"path": str(zdir.relative_to(ROOT))}


def stage_voxlock() -> dict:
    vdir = MODELS_DIR / "voxlock"
    dest = vdir / "embedding.onnx"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[voxlock] present ({dest.stat().st_size/1e6:.1f} MB)")
        return {"path": str(dest.relative_to(ROOT)), "sha256": sha256(dest)}
    print("[voxlock] downloading speaker embedding network...")
    last_error = None
    for url in SPEAKER_URLS:
        try:
            vdir.mkdir(parents=True, exist_ok=True)
            if url.endswith(".onnx"):
                download(url, dest)                       # direct model file
            else:
                with tempfile.TemporaryDirectory(prefix="voxstream-dl-") as tmp:
                    archive = Path(tmp) / "spk.tar.bz2"
                    download(url, archive)
                    extract_dir = Path(tmp) / "extracted"  # untrusted archive: own empty dir
                    safe_extract_tarbz2(archive, extract_dir)
                    onnx_files = list(extract_dir.rglob("*.onnx"))
                    if not onnx_files:
                        raise RuntimeError(f"no onnx in archive {url}")
                    model = max(onnx_files, key=lambda f: f.stat().st_size)
                    shutil.copyfile(model, dest)
            if dest.stat().st_size > 1_000_000:
                print(f"  staged embedding.onnx ({dest.stat().st_size/1e6:.1f} MB)")
                break
            raise RuntimeError("downloaded file too small to be a model")
        except Exception as exc:
            last_error = exc
            print(f"  source failed: {exc}")
    if last_error:
        raise last_error
    return {"path": str(dest.relative_to(ROOT)), "sha256": sha256(dest)}


def stage_voxlock_bundle() -> dict:
    """Stage the VoxLock overlap-extraction bundle (enroll.onnx + chunk.onnx).

    Optional: when missing, VoxStream degrades to the fast TitaNet
    turn-taking gate ("Only Me (Fast)").
    """
    vdir = MODELS_DIR / "voxlock"
    enroll = vdir / "enroll.onnx"
    chunk = vdir / "chunk_int8.onnx"
    if enroll.exists() and chunk.exists():
        print(f"[voxlock] bundle present ({(enroll.stat().st_size + chunk.stat().st_size)/1e6:.0f} MB)")
        return {"path": str(vdir.relative_to(ROOT)),
                "sha256": {"enroll": sha256(enroll), "chunk": sha256(chunk)}}
    print("[voxlock] downloading overlap-extraction bundle (~70 MB)...")
    last_error = None
    for url in VOXLOCK_URLS:
        try:
            with tempfile.TemporaryDirectory(prefix="voxstream-dl-") as tmp:
                archive = Path(tmp) / "voxlock.tar.bz2"
                download(url, archive)
                extract_dir = Path(tmp) / "extracted"   # untrusted archive: own empty dir
                safe_extract_tarbz2(archive, extract_dir)
                enroll_cands = list(extract_dir.rglob("enroll*.onnx"))
                chunk_cands = [f for f in extract_dir.rglob("chunk*.onnx")]
                if not enroll_cands or not chunk_cands:
                    raise RuntimeError(f"unexpected voxlock bundle layout from {url}")
                vdir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(enroll_cands[0], enroll)
                # Prefer an int8 chunk model when present.
                int8 = [f for f in chunk_cands if "int8" in f.name.lower()]
                chosen = (int8 or chunk_cands)[0]
                shutil.copyfile(chosen, chunk)
                print(f"  staged enroll.onnx ({enroll.stat().st_size/1e6:.1f} MB) + "
                      f"chunk_int8.onnx ({chunk.stat().st_size/1e6:.1f} MB)")
                break
        except Exception as exc:
            last_error = exc
            print(f"  source failed: {exc}")
    if last_error:
        raise last_error
    return {"path": str(vdir.relative_to(ROOT)),
            "sha256": {"enroll": sha256(enroll), "chunk": sha256(chunk)}}


def stage_whisper(size: str) -> dict:
    dest = MODELS_DIR / "whisper" / size
    if (dest / "model.bin").exists():
        print(f"[whisper] present: {size} ({(dest / 'model.bin').stat().st_size/1e6:.0f} MB)")
        return {"path": str(dest.relative_to(ROOT)), "size": size}
    print(f"[whisper] downloading faster-whisper {size} via huggingface_hub...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=WHISPER_REPO.format(size=size), local_dir=str(dest))
    if not (dest / "model.bin").exists():
        raise RuntimeError("whisper snapshot incomplete (model.bin missing)")
    return {"path": str(dest.relative_to(ROOT)), "size": size}


# ---------------------------------------------------------------------------
def verify() -> list[str]:
    """Return a list of problems (empty = everything staged)."""
    problems = []
    checks = [
        (MODELS_DIR / "silero_vad.onnx", 100_000),
        (MODELS_DIR / "zipformer" / "tokens.txt", 100),
        (MODELS_DIR / "voxlock" / "embedding.onnx", 1_000_000),
    ]
    for kind in ("encoder", "decoder", "joiner"):
        checks.append((MODELS_DIR / "zipformer" / f"{kind}.onnx", 10_000))
    for path, min_size in checks:
        if not path.exists() or path.stat().st_size < min_size:
            problems.append(f"missing/invalid: {path.relative_to(ROOT)}")
    whisper_dirs = list((MODELS_DIR / "whisper").glob("*/model.bin")) \
        if (MODELS_DIR / "whisper").exists() else []
    if not whisper_dirs:
        problems.append("missing: models/whisper/<size>/model.bin")
    # VoxLock overlap bundle is optional (fast TitaNet gate is the fallback).
    if not (MODELS_DIR / "voxlock" / "enroll.onnx").exists() or \
            not (MODELS_DIR / "voxlock" / "chunk_int8.onnx").exists():
        print("NOTE   optional VoxLock overlap bundle missing "
              "(Only Me (Overlap) will fall back to the fast TitaNet gate)")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage VoxStream offline models")
    parser.add_argument("--whisper", default="small",
                        choices=["tiny", "base", "small", "medium"],
                        help="faster-whisper size (default: small)")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if files exist")
    args = parser.parse_args(argv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        problems = verify()
        if problems:
            for p in problems:
                print(f"MISSING  {p}")
            return 1
        print("OK  all models staged")
        return 0

    if args.force:
        for sub in ("silero_vad.onnx", "zipformer", "voxlock", "whisper"):
            p = MODELS_DIR / sub
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    manifest = {"created": time.time(), "whisper_size": args.whisper, "components": {}}
    steps = [
        ("silero", lambda: stage_silero()),
        ("zipformer", stage_zipformer),
        ("voxlock", stage_voxlock),
        ("voxlock_bundle", stage_voxlock_bundle),
        ("whisper", lambda: stage_whisper(args.whisper)),
    ]
    for name, fn in steps:
        try:
            manifest["components"][name] = fn()
        except Exception as exc:
            print(f"\nFAILED staging {name}: {exc}")
            return 2
        print()

    # keep config.json whisper dir in sync with what we downloaded
    cfg_path = ROOT / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        cfg.setdefault("whisper", {})["dir"] = f"models/whisper/{args.whisper}"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass

    (MODELS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    problems = verify()
    if problems:
        for p in problems:
            print(f"MISSING  {p}")
        return 3

    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"OK  all models staged into {MODELS_DIR.relative_to(ROOT)} "
          f"({total/1e6:.0f} MB total). The app is now 100% offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
