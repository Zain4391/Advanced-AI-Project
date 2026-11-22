"""Pixel-art sprite definitions -- pure data, no pygame.

Each sprite is a list of equal-length strings; each character maps to an RGB
colour in that sprite's palette ('.' = transparent). Sprites are 16x16 unless
noted and are scaled up with nearest-neighbour by env.render_pygame.

`python scripts/make_assets.py` exports these to assets/*.png. If a PNG with the
same name exists in assets/, the renderer uses it instead -- so the game can be
reskinned by editing / replacing PNGs without touching code.
"""
from __future__ import annotations

OUTLINE = (22, 18, 26)

# ----------------------------------------------------------------- survivors
SURVIVOR_PALETTE = {
    "K": OUTLINE,
    "S": (240, 192, 150),   # skin
    "E": (30, 30, 40),      # eyes
    "P": (58, 62, 84),      # pants
    "B": (44, 32, 26),      # boots
    "G": (120, 124, 136),   # gun
}
# per-survivor colours: hair H, shirt C, shirt shade c
SURVIVOR_VARIANTS = (
    {"H": (92, 58, 34), "C": (64, 128, 236), "c": (40, 88, 180)},     # survivor 1: blue jacket
    {"H": (232, 196, 86), "C": (238, 124, 64), "c": (186, 84, 40)},   # survivor 2: orange jacket
)
_SURV_TOP = [
    "................",
    "......KKKK......",
    ".....KHHHHK.....",
    "....KHHHHHHK....",
    "....KHSSSSHK....",
    "....KSESSESK....",
    "....KSSSSSSK....",
    ".....KSSSSK.....",
    "....KCCCCCCK....",
    "...KSCCCCCCKGGK.",
    "...KSCcCCcCKGK..",
    "....KCCCCCCK....",
    "....KPPPPPPK....",
]
SURVIVOR_FRAMES = [
    _SURV_TOP + [
        "....KPPKKPPK....",
        "....KBBK.KBBK...",
        "................",
    ],
    _SURV_TOP + [
        "....KPPKKPPK....",
        "...KBBK..KBBK...",
        "................",
    ],
]

# ------------------------------------------------------------------- zombies
ZOMBIE_PALETTE = {
    "K": OUTLINE,
    "Z": (122, 168, 98),    # rotten skin
    "z": (92, 132, 74),
    "R": (236, 48, 40),     # glowing eyes
    "M": (70, 22, 22),      # mouth
    "T": (112, 92, 122),    # torn shirt
    "t": (80, 64, 92),
    "P": (72, 62, 56),
    "B": (40, 32, 28),
}
_ZOMB_HEAD = [
    "................",
    "......KKKK......",
    ".....KZZZZK.....",
    "....KZZzZZZK....",
    "....KZRZZRZK....",
    "....KZZZZZZK....",
    "....KZKMMKZK....",
    ".....KZZZZK.....",
]
ZOMBIE_FRAMES = [
    _ZOMB_HEAD + [
        "..KKKTTTTTTKKK..",
        ".KZZKTtTTtTKZZK.",
        "..KK.KTTTTK.KK..",
        ".....KTTTTK.....",
        ".....KPPPPK.....",
        ".....KPKKPK.....",
        "....KBK..KBK....",
        "................",
    ],
    _ZOMB_HEAD + [
        ".KZZKTTTTTTKZZK.",
        "..KKKTtTTtTKKK..",
        ".....KTTTTK.....",
        ".....KTTTTK.....",
        ".....KPPPPK.....",
        ".....KPKKPK.....",
        ".....KBKKBK.....",
        "................",
    ],
]

# --------------------------------------------------------------------- items
MEDKIT_PALETTE = {"K": OUTLINE, "W": (238, 238, 232), "w": (186, 186, 180), "R": (220, 40, 44)}
MEDKIT = [
    "................",
    "................",
    "......KKKK......",
    ".....K....K.....",
    "..KKKKKKKKKKKK..",
    "..KWWWWWWWWWWK..",
    "..KWWWWRRWWWWK..",
    "..KWWWWRRWWWWK..",
    "..KWWRRRRRRWWK..",
    "..KWWRRRRRRWWK..",
    "..KWWWWRRWWWWK..",
    "..KWWWWRRWWWWK..",
    "..KwwwwwwwwwwK..",
    "..KKKKKKKKKKKK..",
    "................",
    "................",
]

AMMO_PALETTE = {"K": OUTLINE, "O": (150, 106, 56), "o": (108, 74, 40),
                "Y": (232, 190, 72), "y": (250, 226, 140)}
_AMMO_INNER = ["OOOOOOOOOO", "oooooooooo", "OyyOyyOyyO", "OYYOYYOYYO", "OYYOYYOYYO",
               "OYYOYYOYYO", "oooooooooo", "OOOOOOOOOO", "oooooooooo", "OOOOOOOOOO"]
AMMO = (["................", "................", "..KKKKKKKKKKKK.."]
        + ["..K" + row + "K.." for row in _AMMO_INNER]
        + ["..KKKKKKKKKKKK..", "................", "................", "................"])[:16]

# ---------------------------------------------------------------------- exit
HELIPAD_PALETTE = {"K": (60, 60, 58), "c": (132, 134, 126), "Y": (236, 196, 60), "W": (236, 236, 228)}
HELIPAD = [
    "KKKKKKKKKKKKKKKK",
    "KccccccccccccccK",
    "KccccYYYYYYccccK",
    "KcccYccccccYcccK",
    "KccYccccccccYccK",
    "KcYccWccccWccYcK",
    "KcYccWccccWccYcK",
    "KcYccWWWWWWccYcK",
    "KcYccWWWWWWccYcK",
    "KcYccWccccWccYcK",
    "KcYccWccccWccYcK",
    "KccYccccccccYccK",
    "KcccYccccccYcccK",
    "KccccYYYYYYccccK",
    "KccccccccccccccK",
    "KKKKKKKKKKKKKKKK",
]

# ------------------------------------------------------------ HUD icons (8x8)
ICON_PALETTE = {"K": OUTLINE, "R": (228, 52, 60), "r": (255, 140, 140), "D": (70, 66, 76),
                "Y": (232, 190, 72), "y": (250, 226, 140), "o": (150, 106, 56)}
HEART = [
    ".KK.KK..",
    "KrRKRRK.",
    "KRRRRRK.",
    "KRRRRRK.",
    ".KRRRK..",
    "..KRK...",
    "...K....",
    "........",
]
HEART_EMPTY = [row.replace("R", "D").replace("r", "D") for row in HEART]
BULLET = [
    "...KK...",
    "..KyyK..",
    "..KYYK..",
    "..KYYK..",
    "..KYYK..",
    "..KooK..",
    "..KooK..",
    "..KKKK..",
]
BULLET_EMPTY = [row.replace("Y", "D").replace("y", "D").replace("o", "D") for row in BULLET]


def _check(name, rows, size):
    assert len(rows) == size, f"{name}: {len(rows)} rows"
    for i, r in enumerate(rows):
        assert len(r) == size, f"{name} row {i}: len {len(r)}"


for _n, _rows, _s in [("survivor0", SURVIVOR_FRAMES[0], 16), ("survivor1", SURVIVOR_FRAMES[1], 16),
                      ("zombie0", ZOMBIE_FRAMES[0], 16), ("zombie1", ZOMBIE_FRAMES[1], 16),
                      ("medkit", MEDKIT, 16), ("ammo", AMMO, 16), ("helipad", HELIPAD, 16),
                      ("heart", HEART, 8), ("bullet", BULLET, 8)]:
    _check(_n, _rows, _s)
