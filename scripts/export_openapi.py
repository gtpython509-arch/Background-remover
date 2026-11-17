#!/usr/bin/env python3
"""Write the OpenAPI schema to a file without starting a server.

Postman can import either this file (Import -> File -> openapi.json) or the
hand-tuned collection in postman/. The collection has example bodies and tests;
this file is always exactly in sync with the code.

    python scripts/export_openapi.py                  # -> postman/openapi.json
    python scripts/export_openapi.py -o /tmp/api.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the project importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the OpenAPI schema")
    parser.add_argument(
        "-o", "--output",
        default=str(Path(__file__).resolve().parent.parent / "postman" / "openapi.json"),
        help="Where to write the schema (default: postman/openapi.json)",
    )
    args = parser.parse_args()

    # The schema is built from the route signatures alone, so the model never
    # has to load - this runs in under a second with no weights present.
    os.environ.setdefault("RMBG_ENABLE_DOCS", "true")
    os.environ.setdefault("RMBG_ENVIRONMENT", "development")
    os.environ.setdefault("RMBG_LOG_LEVEL", "WARNING")

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    schema = app.openapi()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(schema, indent=2) + "\n")

    paths = schema.get("paths", {})
    operations = sum(len(methods) for methods in paths.values())
    print(f"Wrote {destination}")
    print(f"  {len(paths)} paths, {operations} operations")
    for path in sorted(paths):
        methods = ",".join(sorted(m.upper() for m in paths[path]))
        print(f"    {methods:<6} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
