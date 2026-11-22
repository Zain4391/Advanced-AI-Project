"""Export the built-in pixel-art sprites to assets/*.png (native 16x16 / 8x8 resolution).

    python scripts/make_assets.py            # writes only missing files
    python scripts/make_assets.py --force    # overwrite (discards your edits!)

To reskin the game, edit or replace any PNG in assets/ keeping its name; the
renderer prefers assets/<name>.png over the built-in definition.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from env.render_pygame import ASSET_DIR, build_default_sprites  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    pygame.init()
    ASSET_DIR.mkdir(exist_ok=True)
    written = 0
    for name, surf in build_default_sprites().items():
        path = ASSET_DIR / f"{name}.png"
        if path.exists() and not args.force:
            continue
        pygame.image.save(surf, str(path))
        written += 1
    print(f"wrote {written} sprites to {ASSET_DIR}")


if __name__ == "__main__":
    main()
