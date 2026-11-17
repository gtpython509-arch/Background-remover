#!/usr/bin/env python3
"""Pre-download the model weights into the configured cache directory.

Run this during the Docker build or once on the VPS, so the first container
start does not pull ~180 MB over the network and startup stays deterministic.
After running it you can set RMBG_LOCAL_FILES_ONLY=true.

    python scripts/download_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.logging_config import configure_logging


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)

    target = settings.model_cache_dir
    target.mkdir(parents=True, exist_ok=True)

    print(f"model      : {settings.model_id}")
    print(f"revision   : {settings.model_revision or 'main'}")
    print(f"cache dir  : {target}")

    from transformers import AutoModelForImageSegmentation

    AutoModelForImageSegmentation.from_pretrained(
        settings.model_id,
        revision=settings.model_revision,
        trust_remote_code=settings.trust_remote_code,
        cache_dir=str(target),
    )

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"\nDownloaded. Cache size: {size / 1e6:.1f} MB")
    print("You can now set RMBG_LOCAL_FILES_ONLY=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
