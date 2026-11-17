#!/usr/bin/env python3
"""Offline batch background removal - the production form of notebook cells 4-5.

Processes a folder (or a list of files) without starting the HTTP server, which
is handy for back-filling an existing image library.

    python scripts/batch.py ./input -o ./output
    python scripts/batch.py a.jpg b.png -o ./out --format webp --feather 1.5
    python scripts/batch.py ./input -o ./out --zip cutouts.zip --recursive
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Make the project importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.logging_config import configure_logging
from app.model import BackgroundRemover
from app.pipeline import process_image
from app.schemas import ProcessingOptions

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_inputs(paths: list[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            globber = path.rglob("*") if recursive else path.glob("*")
            found.extend(
                p for p in sorted(globber)
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file():
            found.append(path)
        else:
            print(f"  ! skipping, not found: {path}", file=sys.stderr)
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch background removal with RMBG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Image files and/or directories")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument(
        "-f", "--format", dest="output_format",
        choices=["png", "webp", "jpeg", "mask"], default="png",
    )
    parser.add_argument("--background", help="Hex colour behind the subject, e.g. #ffffff")
    parser.add_argument("--feather", type=float, default=0.0, help="Edge softening in px")
    parser.add_argument("--threshold", type=float, help="Hard-cut alpha at 0..1")
    parser.add_argument("--edge-contrast", type=float, default=0.0, help="Halo reduction 0..10")
    parser.add_argument("--max-dimension", type=int, help="Downscale the result's longest edge")
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories")
    parser.add_argument("--zip", dest="zip_path", help="Also write a zip of the outputs")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override the device")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    settings = get_settings()
    if args.device:
        settings = settings.model_copy(update={"device": args.device})
    configure_logging(settings.log_level, json_output=False)

    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        print("No input images found.", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    options = ProcessingOptions(
        output_format=args.output_format,
        background=args.background,
        feather=args.feather,
        threshold=args.threshold,
        edge_contrast=args.edge_contrast,
        max_output_dimension=args.max_dimension,
    )

    print(f"Loading {settings.model_id} ...")
    remover = BackgroundRemover(settings)
    info = remover.load()
    print(f"Ready on {info.device} in {info.load_seconds:.1f}s\n")

    ok = failed = skipped = 0
    started = time.perf_counter()

    for index, path in enumerate(files, start=1):
        suffix = "png" if args.output_format in ("png", "mask") else (
            "jpg" if args.output_format == "jpeg" else "webp"
        )
        destination = out_dir / f"{path.stem}_cutout.{suffix}"
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] {path.name} -> exists, skipped")
            skipped += 1
            continue

        try:
            result = process_image(path.read_bytes(), options, settings, remover)
            destination.write_bytes(result.payload)
            ok += 1
            print(
                f"[{index}/{len(files)}] {path.name} -> {destination.name} "
                f"({len(result.payload) / 1024:.0f} KB, "
                f"{result.timings_ms.get('inference', 0):.0f}ms)"
            )
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(files)}] {path.name} -> FAILED: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - started
    print(
        f"\n{ok} done, {failed} failed, {skipped} skipped in {elapsed:.1f}s "
        f"({elapsed / max(1, ok):.2f}s per image)"
    )

    if args.zip_path and ok:
        archive = Path(args.zip_path).with_suffix("")
        shutil.make_archive(str(archive), "zip", out_dir)
        print(f"Archive: {archive}.zip")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
