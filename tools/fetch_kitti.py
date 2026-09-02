"""Fetch and unpack KITTI raw drives listed in configs/dataset.yaml.

Downloads are resumable (HTTP Range), verified by SHA-256, and recorded in a
manifest so the exact bytes every benchmark row was computed on can be named
later. Unused sensor streams are pruned on unpack -- see configs/dataset.yaml.

Usage:
    python tools/fetch_kitti.py                      # every drive in the config
    python tools/fetch_kitti.py --split dev val      # only these splits
    python tools/fetch_kitti.py --drive 2011_09_26_drive_0013
    python tools/fetch_kitti.py --check              # verify what is on disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
REPO = Path(__file__).resolve().parents[1]
CHUNK = 1 << 20  # 1 MiB


def _ssl_context() -> ssl.SSLContext:
    """TLS context with a usable CA bundle.

    python.org macOS builds ship without a populated trust store unless
    `Install Certificates.command` was run, so the stdlib default fails with
    CERTIFICATE_VERIFY_FAILED. Prefer certifi's bundle when it is importable.
    Verification is never disabled -- an unverified fetch of the dataset every
    benchmark row rests on is not a shortcut worth taking.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = _ssl_context()


@dataclass(frozen=True)
class Asset:
    """One downloadable zip and where its contents land."""

    name: str
    url: str
    dest_dir: Path


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _content_length(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length else None
    except urllib.error.URLError:
        return None


def download(url: str, dest: Path, retries: int = 4) -> Path:
    """Resumable download. Returns the path to the complete file.

    Resume is not an optimisation here: these are 1--2 GB files over a link that
    will drop at least once, and restarting from zero each time is how a fetch
    step silently becomes an afternoon.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    total = _content_length(url)

    if dest.exists() and total is not None and dest.stat().st_size == total:
        print(f"  have  {dest.name} ({_human(total)})")
        return dest

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have == total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with (
                urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as resp,
                part.open("ab") as fh,
            ):
                if have and resp.status != 206:
                    # Server ignored the range request; start clean rather than
                    # append a second copy of the file onto the first.
                    fh.close()
                    part.unlink()
                    have = 0
                    raise urllib.error.URLError("range not honoured; restarting")
                t0, last = time.time(), have
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    have += len(chunk)
                    now = time.time()
                    if now - t0 > 5:
                        rate = (have - last) / (now - t0)
                        pct = f"{100 * have / total:5.1f}%" if total else "   ?%"
                        print(
                            f"  ..    {dest.name} {pct} {_human(have)} @ {_human(rate)}/s",
                            flush=True,
                        )
                        t0, last = now, have
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            print(f"  retry {attempt}/{retries} after {exc} -- sleeping {wait}s", flush=True)
            time.sleep(wait)

    if total is not None and part.stat().st_size != total:
        raise RuntimeError(f"{dest.name}: got {part.stat().st_size} bytes, expected {total}")
    part.rename(dest)
    print(f"  done  {dest.name} ({_human(dest.stat().st_size)})")
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def unpack(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def prune_streams(drive_dir: Path, keep: list[str]) -> int:
    """Delete sensor streams this project does not use. Returns bytes freed."""
    freed = 0
    if not drive_dir.is_dir():
        return 0
    for child in sorted(drive_dir.iterdir()):
        if child.is_dir() and child.name not in keep:
            freed += sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            shutil.rmtree(child)
            print(f"  prune {drive_dir.name}/{child.name}")
    return freed


def frame_count(drive_dir: Path) -> int:
    imgs = drive_dir / "image_02" / "data"
    return len(list(imgs.glob("*.png"))) if imgs.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "dataset.yaml")
    ap.add_argument("--split", nargs="*", help="only fetch these splits (dev/val/test)")
    ap.add_argument("--drive", nargs="*", help="only fetch these drive ids")
    ap.add_argument("--keep-zips", action="store_true", help="do not delete zips after unpack")
    ap.add_argument("--check", action="store_true", help="report on-disk state, download nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    root = REPO / cfg["root"]
    date = cfg["calib_date"]
    keep = cfg["keep_streams"]

    drives = cfg["drives"]
    if args.split:
        drives = [d for d in drives if d["split"] in args.split]
    if args.drive:
        drives = [d for d in drives if d["id"] in args.drive]
    if not drives:
        print("no drives selected", file=sys.stderr)
        return 1

    manifest_path = root / "MANIFEST.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    if args.check:
        print(f"{'drive':<28} {'split':<6} {'frames':>7} {'expected':>9}  status")
        ok = True
        for d in drives:
            dd = root / date / f"{d['id']}_sync"
            n = frame_count(dd)
            good = n == d["frames"]
            ok &= good
            print(
                f"{d['id']:<28} {d['split']:<6} {n:>7} {d['frames']:>9}  "
                f"{'ok' if good else 'MISSING/INCOMPLETE'}"
            )
        calib = root / date / "calib_cam_to_cam.txt"
        print(f"calibration: {'ok' if calib.exists() else 'MISSING'} ({calib})")
        return 0 if ok and calib.exists() else 1

    zips = root / "_zips"
    freed_total = 0

    # Calibration first: without it nothing downstream means anything.
    calib_zip = zips / f"{date}_calib.zip"
    if not (root / date / "calib_cam_to_cam.txt").exists():
        print(f"calibration {date}")
        download(f"{BASE}/{date}_calib.zip", calib_zip)
        manifest[calib_zip.name] = {
            "url": f"{BASE}/{date}_calib.zip",
            "bytes": calib_zip.stat().st_size,
            "sha256": sha256(calib_zip),
        }
        unpack(calib_zip, root)
        if not args.keep_zips:
            calib_zip.unlink()
    else:
        print(f"calibration {date} already present")

    for d in drives:
        did = d["id"]
        drive_dir = root / date / f"{did}_sync"
        print(f"\n{did}  [{d['split']}/{d['scene']}]  expect {d['frames']} frames")

        if frame_count(drive_dir) == d["frames"]:
            print("  have  complete drive, skipping")
        else:
            for suffix in ("sync", "tracklets"):
                url = f"{BASE}/{did}/{did}_{suffix}.zip"
                zp = zips / f"{did}_{suffix}.zip"
                download(url, zp)
                manifest[zp.name] = {
                    "url": url,
                    "bytes": zp.stat().st_size,
                    "sha256": sha256(zp),
                    "split": d["split"],
                }
                unpack(zp, root)
                if not args.keep_zips:
                    zp.unlink()

        freed_total += prune_streams(drive_dir, keep)
        n = frame_count(drive_dir)
        if n != d["frames"]:
            print(f"  WARN  {did}: {n} frames on disk, config says {d['frames']}")
        manifest.setdefault(f"{did}_sync.zip", {})["frames_on_disk"] = n

    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if zips.is_dir() and not any(zips.iterdir()):
        zips.rmdir()

    print(f"\npruned {_human(freed_total)} of unused sensor streams")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
