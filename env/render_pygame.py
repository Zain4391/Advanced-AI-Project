"""Pygame renderer: pixel-art sprites, tweened movement and effects. Draws state only.

No game logic lives here -- the renderer only *looks at* EnvState snapshots and the
`info["events"]` dict returned by env.step(). Import it only from scripts/ (play,
demo viewer). For headless use set SDL_VIDEODRIVER=dummy before importing.

    r = Renderer(env.config, cell=80)
    screen = pygame.display.set_mode(r.size)
    prev = env.get_state(); obs, rew, term, trunc, info = env.step(a)
    r.on_step(prev, env.state, info)          # spawns animations / effects
    r.draw(screen, env.state, pending=..., seed=..., total_return=...)

draw_board() renders only the map (used twice by the side-by-side demo viewer).
"""
from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame

from . import sprites as SP
from .config import ACTION_NAMES, SHOOT, EnvConfig
from .outbreak_env import shot_target
from .state import EnvState

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
NATIVE = 16          # native sprite resolution
ANIM_TIME = 0.16     # seconds to tween a move
EVAC_TIME = 0.7

C = {
    "bg": (14, 14, 18),
    "panel": (24, 24, 30),
    "panel_card": (34, 34, 42),
    "panel_edge": (52, 52, 64),
    "text": (232, 232, 236),
    "muted": (140, 140, 152),
    "title": (226, 70, 60),
    "good": (96, 214, 110),
    "bad": (232, 72, 64),
    "warn": (240, 196, 72),
    "ooze": (86, 182, 64),
    "blood": (120, 18, 22),
    "tracer": (255, 236, 150),
}


# ======================================================================= assets
def _from_rows(rows, palette) -> pygame.Surface:
    surf = pygame.Surface((len(rows[0]), len(rows)), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                surf.set_at((x, y), palette[ch])
    return surf


def _from_array(img: np.ndarray) -> pygame.Surface:
    h, w = img.shape[:2]
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    img = np.clip(img, 0, 255).astype(int)
    for y in range(h):
        for x in range(w):
            surf.set_at((x, y), tuple(img[y, x]))
    return surf


def _noise(seed, base, amp):
    rng = np.random.default_rng(seed)
    return np.array(base, float) + rng.integers(-amp, amp + 1, (NATIVE, NATIVE, 1)), rng


def _floor_tile(k: int) -> np.ndarray:
    img, rng = _noise(100 + k, (50, 53, 60), 4)
    img[rng.random((NATIVE, NATIVE)) < 0.06] += 9
    yy, xx = np.mgrid[0:NATIVE, 0:NATIVE]
    if k == 1:    # crack
        x, y = 2, 3
        for _ in range(16):
            img[y, x] = (30, 31, 36)
            if rng.random() < 0.6 and x < 15:
                x += 1
            elif y < 15:
                y += 1
    elif k == 2:  # faded road paint
        img[1:6, 7:9] = (148, 138, 90)
        img[10:15, 7:9] = (148, 138, 90)
    elif k == 3:  # manhole
        d = np.hypot(yy - 7.5, xx - 7.5)
        img[d <= 5.3] = (72, 74, 80)
        img[d <= 4.5] = (40, 42, 47)
        img[(d <= 4.5) & ((yy + xx) % 3 == 0)] = (56, 58, 64)
    elif k == 4:  # debris / leaves
        for _ in range(4):
            y, x = rng.integers(1, 15, 2)
            img[y, x] = (100, 74, 44)
            img[y, min(x + 1, 15)] = (84, 62, 38)
    elif k == 5:  # oil stain
        d = np.hypot(yy - 9, xx - 5)
        img[d < 3.6] *= 0.72
    return img


def _roof_tile(k: int) -> np.ndarray:
    img, rng = _noise(200 + k, (84, 78, 92), 3)
    img[0, :] = img[:, 0] = (116, 110, 124)
    img[15, :] = img[:, 15] = (56, 50, 64)
    if k == 1:    # vent
        img[4:8, 9:13] = (118, 122, 130)
        img[4, 9:13] = (150, 154, 162)
        img[7, 9:13] = (76, 80, 88)
    elif k == 2:  # AC unit
        img[7:12, 3:9] = (148, 150, 152)
        img[8:11, 4:8:2] = (92, 94, 98)
        img[11, 3:9] = (96, 98, 102)
    elif k == 3:  # skylight
        img[3:7, 3:8] = (64, 104, 134)
        img[3, 3:8] = (122, 164, 194)
    return img


def _facade(k: int) -> np.ndarray:
    """16 x 6 brick front, drawn at the bottom of a wall cell whose south neighbour is open."""
    img = np.zeros((6, NATIVE, 3))
    img[:] = (122, 64, 52)
    for y in range(6):
        for x in range(NATIVE):
            if y % 2 == 1 or (x + (y // 2) % 2 * 4) % 8 == 0:
                img[y, x] = (88, 46, 38)
    img[0, :] = (46, 30, 30)  # eave shadow
    if k in (0, 2):
        lit = (238, 206, 118) if k == 2 else (42, 46, 64)
        img[2:5, 5:11] = lit
        img[2:5, 5] = img[2:5, 10] = (36, 26, 24)
        img[2, 5:11] = (36, 26, 24)
    return img


def build_default_sprites() -> dict[str, pygame.Surface]:
    """All sprites at native resolution, generated from env/sprites.py + procedural tiles."""
    out: dict[str, pygame.Surface] = {}
    for v, overrides in enumerate(SP.SURVIVOR_VARIANTS):
        pal = {**SP.SURVIVOR_PALETTE, **overrides}
        for f, frame in enumerate(SP.SURVIVOR_FRAMES):
            out[f"survivor{v + 1}_{f}"] = _from_rows(frame, pal)
    for f, frame in enumerate(SP.ZOMBIE_FRAMES):
        out[f"zombie_{f}"] = _from_rows(frame, SP.ZOMBIE_PALETTE)
    out["medkit"] = _from_rows(SP.MEDKIT, SP.MEDKIT_PALETTE)
    out["ammo"] = _from_rows(SP.AMMO, SP.AMMO_PALETTE)
    out["helipad"] = _from_rows(SP.HELIPAD, SP.HELIPAD_PALETTE)
    out["heart"] = _from_rows(SP.HEART, SP.ICON_PALETTE)
    out["heart_empty"] = _from_rows(SP.HEART_EMPTY, SP.ICON_PALETTE)
    out["bullet"] = _from_rows(SP.BULLET, SP.ICON_PALETTE)
    out["bullet_empty"] = _from_rows(SP.BULLET_EMPTY, SP.ICON_PALETTE)
    for k in range(6):
        out[f"floor_{k}"] = _from_array(_floor_tile(k))
    for k in range(4):
        out[f"roof_{k}"] = _from_array(_roof_tile(k))
    for k in range(3):
        out[f"facade_{k}"] = _from_array(_facade(k))
    return out


def load_sprites() -> dict[str, pygame.Surface]:
    """Defaults, overridden by any assets/<name>.png of the same name."""
    sprites = build_default_sprites()
    for name in sprites:
        p = ASSET_DIR / f"{name}.png"
        if p.exists():
            try:
                sprites[name] = pygame.image.load(str(p))
            except pygame.error:
                pass
    return sprites


# ====================================================================== helpers
def _font(size: int, bold: bool = False) -> pygame.font.Font:
    for name in ("bahnschrift", "segoeui", "verdana", "dejavusans", "arial"):
        if pygame.font.match_font(name):
            return pygame.font.SysFont(name, size, bold=bold)
    return pygame.font.Font(None, size + 6)


def _text(surf, font, text, pos, color, shadow=True, center=False):
    img = font.render(text, True, color)
    rect = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
    if shadow:
        surf.blit(font.render(text, True, (0, 0, 0)), rect.move(2, 2))
    surf.blit(img, rect)
    return rect


def _lerp(a, b, t):
    return a + (b - a) * t


def _ease(t):
    return 1 - (1 - t) ** 3


@dataclass
class _Fx:
    kind: str
    x: float
    y: float
    t0: float
    dur: float
    data: dict = field(default_factory=dict)


# ===================================================================== renderer
class Renderer:
    def __init__(self, config: EnvConfig, cell: int = 80, panel_width: int = 360):
        if not pygame.get_init():
            pygame.init()
        pygame.font.init()
        self.cfg = config
        self.cell = cell
        self.n = config.grid_size
        self.board_px = self.n * cell
        self.panel_w = panel_width
        self.size = (self.board_px + panel_width, self.board_px)

        raw = load_sprites()
        sc = lambda s, k=cell: pygame.transform.scale(s, (k, k * s.get_height() // s.get_width()))
        self.spr = {k: sc(v) for k, v in raw.items() if not k.startswith(("heart", "bullet"))}
        self.icon = {k: pygame.transform.scale(v, (22, 22)) for k, v in raw.items()
                     if k.startswith(("heart", "bullet"))}
        self.portrait = {i: pygame.transform.scale(raw[f"survivor{i + 1}_0"], (64, 64)) for i in (0, 1)}
        self._variants: dict = {}

        self.f_title = _font(30, bold=True)
        self.f_big = _font(44, bold=True)
        self.f_med = _font(19, bold=True)
        self.f_small = _font(15)
        self.f_fx = _font(18, bold=True)

        self._board = pygame.Surface((self.board_px, self.board_px))
        self._bg_key = None
        self._bg = None
        self._ooze = [self._make_ooze(i) for i in range(4)]
        self._vignette = self._make_vignette()
        self.reset()

    # ------------------------------------------------------------ lifecycle ---
    def reset(self) -> None:
        self.prev: EnvState | None = None
        self.step_time = -1e9
        self.effects: list[_Fx] = []
        self.decals: list[tuple[int, int, int]] = []
        self.facing = [1, 1]
        self.zfacing = np.ones(self.cfg.max_zombies, int)
        self.hurt = [-1e9, -1e9]
        self.spread: dict[tuple, float] = {}
        self.drops: dict[tuple, float] = {}
        self.evac_time: list[float | None] = [None, None]

    def animating(self, now: float | None = None) -> bool:
        now = time.perf_counter() if now is None else now
        return now - self.step_time < ANIM_TIME or bool(self.effects)

    def on_step(self, prev: EnvState, s: EnvState, info: dict, now: float | None = None,
                action: tuple[int, int] | None = None) -> None:
        """Spawn tweens and effects for the transition prev -> s.
        Pass the joint `action` to also get "NO AMMO" feedback for empty-gun shots."""
        now = time.perf_counter() if now is None else now
        self.prev, self.step_time = prev.copy(), now
        ev = info.get("events") or {}
        c = self.cell

        for i in (0, 1):
            dx = int(s.surv_pos[i, 1] - prev.surv_pos[i, 1])
            if dx:
                self.facing[i] = 1 if dx > 0 else -1
        dz = s.zombie_pos[:, 1] - prev.zombie_pos[:, 1]
        self.zfacing = np.where(dz != 0, np.sign(dz), self.zfacing)

        killed = np.flatnonzero(prev.zombie_alive & ~s.zombie_alive)
        targets = ev.get("shot_targets", (-1, -1))
        for i, shot in enumerate(ev.get("shots", (False, False))):
            ox, oy = self._center(prev.surv_pos[i])
            if not shot:
                if action is not None and action[i] == SHOOT and not prev.evacuated[i]:
                    self.effects.append(_Fx("text", ox, oy - c * 0.45, now, 1.1,
                                            {"text": "NO AMMO", "color": C["muted"]}))
                continue
            self.effects.append(_Fx("muzzle", ox, oy, now, 0.18))
            if targets[i] >= 0:
                tx, ty = self._center(prev.zombie_pos[targets[i]])
                self.effects.append(_Fx("tracer", ox, oy, now, 0.25, {"to": (tx, ty)}))
            else:
                self.effects.append(_Fx("text", ox, oy - c * 0.45, now, 1.1,
                                        {"text": "MISS", "color": C["muted"]}))
        for z in killed:
            r, col = prev.zombie_pos[z]
            self.decals.append((int(r), int(col), int(z)))
            x, y = self._center((r, col))
            self.effects.append(_Fx("puff", x, y, now, 0.45))
            self.effects.append(_Fx("text", x, y - c * 0.3, now, 0.9, {"text": "+KILL", "color": C["warn"]}))

        dmg = ev.get("damage", (0, 0))
        shots = ev.get("shots", (False, False))
        for i in (0, 1):
            x, y = self._center(s.surv_pos[i])
            if dmg[i]:
                self.hurt[i] = now
                self.effects.append(_Fx("text", x, y - c * 0.45, now, 0.9, {"text": f"-{dmg[i]}", "color": C["bad"]}))
            heal = int(s.health[i] - prev.health[i] + dmg[i])
            if heal > 0:
                self.effects.append(_Fx("text", x, y - c * 0.7, now + 0.05, 1.0, {"text": f"+{heal} HP", "color": C["good"]}))
                self.effects.append(_Fx("ring", x, y, now, 0.5, {"color": C["good"]}))
            gain = int(s.ammo[i] - prev.ammo[i] + int(shots[i]))
            if gain > 0:
                self.effects.append(_Fx("text", x, y - c * 0.7, now + 0.05, 1.0, {"text": "+1 AMMO", "color": C["warn"]}))
                self.effects.append(_Fx("ring", x, y, now, 0.5, {"color": C["warn"]}))
            if s.evacuated[i] and not prev.evacuated[i]:
                self.evac_time[i] = now + ANIM_TIME
            if (ev.get("deaths") or (False, False))[i]:
                self.effects.append(_Fx("text", x, y - c * 0.9, now + 0.1, 1.8,
                                        {"text": f"SURVIVOR {i + 1} DOWN", "color": C["bad"]}))
                self.effects.append(_Fx("puff", x, y, now, 0.5))

        for r, col in np.argwhere(s.infection & ~prev.infection):
            self.spread[(int(r), int(col))] = now
        for r, col in np.argwhere((s.medkits & ~prev.medkits) | (s.ammo_crates & ~prev.ammo_crates)):
            self.drops[(int(r), int(col))] = now
        if s.gate_open and not prev.gate_open:
            x, y = self._center(s.exit_pos)
            self.effects.append(_Fx("ring", x, y, now, 0.8, {"color": C["good"], "big": True}))
            self.effects.append(_Fx("text", x, y - c * 0.5, now, 1.4, {"text": "GATE OPEN", "color": C["good"]}))

    # -------------------------------------------------------------- drawing ---
    def draw(self, surface, s: EnvState, *, pending=(None, None), seed=None, total_return=0.0,
             banner: tuple[str, str, tuple] | None = None, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        surface.fill(C["bg"])
        self.draw_board(surface, s, (0, 0), now=now)
        if banner and now - self.step_time > 0.6:
            self.draw_banner(surface, pygame.Rect(0, 0, self.board_px, self.board_px), *banner)
        self.draw_panel(surface, pygame.Rect(self.board_px, 0, self.panel_w, self.board_px), s,
                        pending=pending, seed=seed, total_return=total_return, now=now)

    def draw_board(self, surface, s: EnvState, origin=(0, 0), now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        b, c = self._board, self.cell
        b.blit(self._background(s), (0, 0))
        prev = self.prev if self.prev is not None and self.prev.walls is s.walls else None
        a = _ease(min(1.0, max(0.0, (now - self.step_time) / ANIM_TIME))) if prev is not None else 1.0

        # decals (blood splats)
        for r, col, z in self.decals:
            self._splat(b, r, col, z)

        # exit pad
        er, ec = s.exit_pos
        ex, ey = ec * c, er * c
        b.blit(self.spr["helipad"], (ex, ey))
        pulse = 0.5 + 0.5 * math.sin(now * (4 if s.gate_open else 2.5))
        light = C["good"] if s.gate_open else C["bad"]
        glow = pygame.Surface((c, c), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*light, int(60 + 120 * pulse)), glow.get_rect(), max(3, c // 16), border_radius=4)
        b.blit(glow, (ex, ey))
        for lx, ly in ((0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)):
            pygame.draw.circle(b, light if pulse > 0.35 else (40, 40, 40), (ex + int(lx * c), ey + int(ly * c)), max(2, c // 22))
        if not s.gate_open:
            self._hazard_bar(b, ex, ey + int(c * 0.42), c)

        # infection
        for r, col in np.argwhere(s.infection):
            r, col = int(r), int(col)
            k = (r * 7 + col * 13) % 4
            ooze = self._ooze[k]
            t0 = self.spread.get((r, col))
            alpha = int(150 + 45 * math.sin(now * 2.2 + r * 0.9 + col * 1.3))
            if t0 is not None and now - t0 < 0.45:
                p = _ease((now - t0) / 0.45)
                size = max(2, int(c * p))
                img = pygame.transform.smoothscale(ooze, (size, size))
                img.set_alpha(alpha)
                b.blit(img, (col * c + (c - size) // 2, r * c + (c - size) // 2))
            else:
                ooze.set_alpha(alpha)
                b.blit(ooze, (col * c, r * c))
            # bubbles
            ph = (now * 0.8 + (r * 31 + col * 17) % 10 / 10) % 1.0
            bx = col * c + int(c * (0.25 + 0.5 * (((r * 5 + col * 3) % 7) / 7)))
            by = r * c + int(c * (0.85 - 0.6 * ph))
            pygame.draw.circle(b, (170, 240, 140), (bx, by), max(2, int(c * 0.05 * (1 - ph))) + 1, 1)

        # items
        for grid, name in ((s.medkits, "medkit"), (s.ammo_crates, "ammo")):
            for r, col in np.argwhere(grid):
                r, col = int(r), int(col)
                bob = int(math.sin(now * 3 + r + col) * c * 0.03)
                off = 0
                t0 = self.drops.get((r, col))
                if t0 is not None and now - t0 < 0.5:
                    off = -int(c * 2.5 * (1 - _ease((now - t0) / 0.5)))
                self._shadow(b, col * c + c // 2, r * c + int(c * 0.82), 0.5)
                b.blit(self.spr[name], (col * c, r * c + bob + off))

        # characters, painter's order by y
        draws = []
        for z in range(self.cfg.max_zombies):
            alive_now = bool(s.zombie_alive[z])
            died = prev is not None and prev.zombie_alive[z] and not alive_now
            if not (alive_now or (died and now - self.step_time < 0.3)):
                continue
            p0 = prev.zombie_pos[z] if prev is not None and prev.zombie_alive[z] else s.zombie_pos[z]
            pos = p0 if died else _lerp(np.asarray(p0, float), s.zombie_pos[z], a)
            fade = 1.0 - (now - self.step_time) / 0.3 if died else 1.0
            draws.append((pos[0], "z", z, pos, fade))
        for i in (0, 1):
            if s.evacuated[i] and (prev is None or prev.evacuated[i]):
                et = self.evac_time[i]
                if et is None or now - et > EVAC_TIME:
                    continue
            p0 = prev.surv_pos[i] if prev is not None and not prev.evacuated[i] else s.surv_pos[i]
            pos = _lerp(np.asarray(p0, float), s.surv_pos[i], a)
            draws.append((pos[0] + 0.01, "s", i, pos, 1.0))
        for _, kind, idx, pos, fade in sorted(draws, key=lambda d: d[0]):
            if kind == "z":
                self._draw_zombie(b, idx, pos, fade, now)
            else:
                self._draw_survivor(b, s, idx, pos, now, moving=a < 1 and prev is not None
                                    and (prev.surv_pos[idx] != s.surv_pos[idx]).any())

        # crosshairs: which zombie each armed survivor would hit right now
        if not s.done and a >= 1:
            for i in (0, 1):
                if s.evacuated[i] or s.health[i] <= 0 or s.ammo[i] <= 0:
                    continue
                z = shot_target(s, self.cfg, i)
                if z is not None:
                    self._crosshair(b, s.zombie_pos[z], SP.SURVIVOR_VARIANTS[i]["C"], now, i)

        self._draw_effects(b, now)
        b.blit(self._vignette, (0, 0))
        surface.blit(b, origin)

    def _draw_survivor(self, b, s, i, pos, now, moving):
        c = self.cell
        x, y = pos[1] * c, pos[0] * c
        frame = int(now * 10) % 2 if moving else 0
        img = self._variant(f"survivor{i + 1}_{frame}", flip=self.facing[i] < 0,
                            hurt=now - self.hurt[i] < 0.35 and int(now * 24) % 2 == 0,
                            down=s.health[i] <= 0)
        bob = 0 if moving or s.health[i] <= 0 else int(round(math.sin(now * 3.5 + i * 2) * c * 0.025))
        alpha = 255
        et = self.evac_time[i]
        if s.evacuated[i] and et is not None and now >= et:
            p = min(1.0, (now - et) / EVAC_TIME)
            bob = -int(c * 0.8 * _ease(p))
            alpha = int(255 * (1 - p))
            pygame.draw.line(b, C["good"], (x + c // 2, y + c), (x + c // 2, y + bob), max(2, c // 20))
        self._shadow(b, x + c // 2, y + int(c * 0.9), 0.55)
        if alpha < 255:
            img = img.copy()
            img.set_alpha(alpha)
        b.blit(img, (x, y + bob))
        if not s.evacuated[i] and s.health[i] > 0:
            # hp bar + player tag
            w = int(c * 0.6)
            bx, by = int(x + (c - w) / 2), int(y + bob - c * 0.02)
            pygame.draw.rect(b, (0, 0, 0), (bx - 1, by - 1, w + 2, 7))
            seg = w / self.cfg.max_health
            for k in range(self.cfg.max_health):
                col = C["good"] if k < s.health[i] else (60, 60, 66)
                if s.health[i] <= 2 and k < s.health[i]:
                    col = C["bad"]
                pygame.draw.rect(b, col, (bx + int(k * seg) + 1, by, max(1, int(seg) - 2), 5))
            tag_col = SP.SURVIVOR_VARIANTS[i]["C"]
            _text(b, self.f_small, f"P{i + 1}", (max(2, bx - 20), by - 7), tag_col)

    def _draw_zombie(self, b, z, pos, fade, now):
        c = self.cell
        x, y = pos[1] * c, pos[0] * c
        frame = int(now * 3 + z * 0.7) % 2
        img = self._variant(f"zombie_{frame}", flip=self.zfacing[z] < 0)
        sway = int(math.sin(now * 2 + z) * c * 0.03)
        self._shadow(b, x + c // 2, y + int(c * 0.9), 0.55)
        if fade < 1:
            img = img.copy()
            img.set_alpha(int(255 * max(0, fade)))
        b.blit(img, (x + sway, y))

    def _draw_effects(self, b, now):
        alive = []
        c = self.cell
        for fx in self.effects:
            t = (now - fx.t0) / fx.dur
            if t < 0:
                alive.append(fx)
                continue
            if t >= 1:
                continue
            alive.append(fx)
            if fx.kind == "tracer":
                tx, ty = fx.data["to"]
                col = (*C["tracer"], int(255 * (1 - t)))
                layer = pygame.Surface(b.get_size(), pygame.SRCALPHA)
                pygame.draw.line(layer, col, (fx.x, fx.y), (tx, ty), max(2, c // 20))
                b.blit(layer, (0, 0))
            elif fx.kind == "muzzle":
                pygame.draw.circle(b, (255, 230, 140), (int(fx.x), int(fx.y)), int(c * 0.22 * (1 - t)) + 2)
            elif fx.kind == "puff":
                rad = int(c * (0.2 + 0.4 * t))
                layer = pygame.Surface((rad * 2 + 2, rad * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(layer, (*C["blood"], int(200 * (1 - t))), (rad + 1, rad + 1), rad)
                b.blit(layer, (fx.x - rad, fx.y - rad))
            elif fx.kind == "ring":
                rad = int(c * ((0.3 + 0.9 * t) if fx.data.get("big") else (0.2 + 0.4 * t)))
                layer = pygame.Surface((rad * 2 + 4, rad * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(layer, (*fx.data["color"], int(230 * (1 - t))), (rad + 2, rad + 2), rad, 3)
                b.blit(layer, (fx.x - rad - 2, fx.y - rad - 2))
            elif fx.kind == "text":
                img = self.f_fx.render(fx.data["text"], True, fx.data["color"])
                sh = self.f_fx.render(fx.data["text"], True, (0, 0, 0))
                a = int(255 * min(1.0, 2 * (1 - t)))
                img.set_alpha(a)
                sh.set_alpha(a)
                y = fx.y - c * 0.5 * _ease(t)
                r = img.get_rect(center=(fx.x, y))
                r.clamp_ip(b.get_rect().inflate(-4, -4))
                b.blit(sh, r.move(2, 2))
                b.blit(img, r)
        self.effects = alive

    # ---------------------------------------------------------------- panel ---
    def draw_panel(self, surface, rect: pygame.Rect, s: EnvState, *, pending=(None, None),
                   seed=None, total_return=0.0, now: float | None = None, show_controls=True) -> None:
        now = time.perf_counter() if now is None else now
        x0, y0, w = rect.x, rect.y, rect.w
        pygame.draw.rect(surface, C["panel"], rect)
        pygame.draw.line(surface, C["title"], (x0, y0), (x0, rect.bottom), 3)
        pad = 18
        x = x0 + pad
        inner = w - 2 * pad

        _text(surface, self.f_title, "ZOMBIE OUTBREAK", (x, y0 + 14), C["title"])
        _text(surface, self.f_small, "Get both survivors to the helipad.", (x, y0 + 52), C["muted"], shadow=False)

        # step bar
        y = y0 + 82
        _text(surface, self.f_med, f"STEP {s.t} / {self.cfg.max_steps}", (x, y), C["text"])
        bar = pygame.Rect(x, y + 28, inner, 8)
        pygame.draw.rect(surface, C["panel_card"], bar, border_radius=4)
        frac = s.t / self.cfg.max_steps
        pygame.draw.rect(surface, C["bad"] if frac > 0.8 else C["warn"],
                         (bar.x, bar.y, max(4, int(bar.w * frac)), bar.h), border_radius=4)

        # gate
        y += 50
        gate = pygame.Rect(x, y, inner, 34)
        col = C["good"] if s.gate_open else C["bad"]
        pygame.draw.rect(surface, C["panel_card"], gate, border_radius=6)
        pygame.draw.rect(surface, col, gate, 2, border_radius=6)
        pygame.draw.circle(surface, col, (gate.x + 18, gate.centery), 6)
        _text(surface, self.f_med, "GATE OPEN" if s.gate_open else "GATE SEALED", (gate.x + 34, gate.y + 6), col, shadow=False)
        hint = "survivors evacuate on the pad" if s.gate_open else "opens when BOTH stand next to the pad"
        _text(surface, self.f_small, hint, (gate.x + 2, gate.bottom + 4), C["muted"], shadow=False)

        # survivor cards
        y += 66
        for i in (0, 1):
            card = pygame.Rect(x, y, inner, 106)
            accent = SP.SURVIVOR_VARIANTS[i]["C"]
            pygame.draw.rect(surface, C["panel_card"], card, border_radius=8)
            pygame.draw.rect(surface, accent, (card.x, card.y, 5, card.h), border_top_left_radius=8, border_bottom_left_radius=8)
            surface.blit(self.portrait[i], (card.x + 12, card.y + 18))
            tx = card.x + 88
            _text(surface, self.f_med, f"SURVIVOR {i + 1}", (tx, card.y + 8), accent)
            for k in range(self.cfg.max_health):
                surface.blit(self.icon["heart" if k < s.health[i] else "heart_empty"], (tx + k * 25, card.y + 36))
            for k in range(self.cfg.max_ammo):
                surface.blit(self.icon["bullet" if k < s.ammo[i] else "bullet_empty"], (tx + k * 25, card.y + 60))
            if s.evacuated[i]:
                status, scol = "EVACUATED", C["good"]
            elif s.health[i] <= 0:
                status, scol = "DOWN", C["bad"]
            elif s.done:
                status, scol = "-", C["muted"]
            elif pending[i] is None:
                status, scol = "choosing" + "." * (int(now * 3) % 4), C["muted"]
            else:
                status, scol = f"READY: {ACTION_NAMES[pending[i]].upper()}", C["good"]
            _text(surface, self.f_small, status, (tx + 90, card.y + 62), scol, shadow=False)
            if not (s.evacuated[i] or s.health[i] <= 0 or s.done):
                if s.ammo[i] <= 0:
                    hint, hcol = "out of ammo - find a crate", C["bad"]
                elif shot_target(s, self.cfg, i) is not None:
                    hint, hcol = "zombie in sights - shoot!", C["warn"]
                else:
                    hint, hcol = "no zombie in line (2 tiles)", C["muted"]
                _text(surface, self.f_small, hint, (tx, card.y + 82), hcol, shadow=False)
            y += 114

        # controls
        if show_controls:
            y += 2
            box = pygame.Rect(x, y, inner, 96)
            pygame.draw.rect(surface, C["panel_card"], box, border_radius=8)
            rows = [("P1", "Arrows move   Space shoot   Enter wait", SP.SURVIVOR_VARIANTS[0]["C"]),
                    ("P2", "WASD move   F shoot   Q wait", SP.SURVIVOR_VARIANTS[1]["C"]),
                    ("", "R restart   N next map   Esc quit", C["muted"])]
            for k, (tag, txt, col) in enumerate(rows):
                yy = box.y + 12 + k * 28
                if tag:
                    _text(surface, self.f_med, tag, (box.x + 12, yy - 2), col, shadow=False)
                _text(surface, self.f_small, txt, (box.x + 48, yy), C["text"] if tag else C["muted"], shadow=False)

        foot = f"map seed {seed}" if seed is not None else ""
        _text(surface, self.f_small, foot, (x, rect.bottom - 30), C["muted"], shadow=False)
        _text(surface, self.f_small, f"team return {total_return:+.1f}", (x + inner - 130, rect.bottom - 30), C["muted"], shadow=False)

    def draw_banner(self, surface, rect: pygame.Rect, title: str, subtitle: str, color) -> None:
        dim = pygame.Surface(rect.size, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surface.blit(dim, rect.topleft)
        band = pygame.Rect(rect.x, rect.centery - 60, rect.w, 120)
        pygame.draw.rect(surface, (10, 10, 14), band)
        pygame.draw.line(surface, color, band.topleft, band.topright, 3)
        pygame.draw.line(surface, color, band.bottomleft, band.bottomright, 3)
        _text(surface, self.f_big, title, (band.centerx, band.centery - 16), color, center=True)
        _text(surface, self.f_small, subtitle, (band.centerx, band.centery + 30), C["text"], center=True, shadow=False)

    # ------------------------------------------------------------- internals ---
    def _center(self, p):
        return int(p[1]) * self.cell + self.cell // 2, int(p[0]) * self.cell + self.cell // 2

    def _variant(self, name, flip=False, hurt=False, down=False):
        key = (name, flip, hurt, down)
        img = self._variants.get(key)
        if img is None:
            img = self.spr[name]
            if flip:
                img = pygame.transform.flip(img, True, False)
            if down:
                img = pygame.transform.rotate(img, -90)
                grey = img.copy()
                grey.fill((70, 70, 70), special_flags=pygame.BLEND_RGB_SUB)
                img = grey
            if hurt:
                img = img.copy()
                img.fill((150, 0, 0), special_flags=pygame.BLEND_RGB_ADD)
            self._variants[key] = img
        return img

    def _crosshair(self, b, p, color, now, i):
        c = self.cell
        cx, cy = self._center(p)
        rad = int(c * (0.36 + 0.06 * i) + c * 0.03 * math.sin(now * 6 + i * 1.5))
        w = max(2, c // 32)
        pygame.draw.circle(b, color, (cx, cy), rad, w)
        tick = int(c * 0.14)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pygame.draw.line(b, color, (cx + dx * (rad - tick // 2), cy + dy * (rad - tick // 2)),
                             (cx + dx * (rad + tick), cy + dy * (rad + tick)), w)

    def _shadow(self, b, cx, cy, scale):
        w, h = int(self.cell * scale), int(self.cell * scale * 0.3)
        sh = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 90), sh.get_rect())
        b.blit(sh, (cx - w // 2, cy - h // 2))

    def _hazard_bar(self, b, x, y, c):
        h = max(6, c // 7)
        bar = pygame.Surface((c - 8, h), pygame.SRCALPHA)
        bar.fill((236, 196, 60, 230))
        step = max(6, c // 8)
        for k in range(-h, c, step * 2):
            pygame.draw.polygon(bar, (30, 30, 30, 230), [(k, h), (k + step, h), (k + step + h, 0), (k + h, 0)])
        b.blit(bar, (x + 4, y))

    def _splat(self, b, r, col, z):
        c = self.cell
        rng = np.random.default_rng(r * 131 + col * 17 + z)
        cx, cy = col * c + c // 2, r * c + c // 2
        layer = pygame.Surface((c, c), pygame.SRCALPHA)
        for _ in range(7):
            ox, oy = rng.normal(0, c * 0.14, 2)
            rad = int(rng.uniform(0.05, 0.16) * c)
            pygame.draw.circle(layer, (*C["blood"], 170), (int(c / 2 + ox), int(c / 2 + oy)), rad)
        b.blit(layer, (cx - c // 2, cy - c // 2))

    def _make_ooze(self, k):
        c = self.cell
        rng = np.random.default_rng(300 + k)
        s = pygame.Surface((c, c), pygame.SRCALPHA)
        s.fill((*C["ooze"], 95))
        for _ in range(9):
            x, y = rng.integers(0, c, 2)
            pygame.draw.circle(s, (110, 206, 80, 120), (int(x), int(y)), int(rng.uniform(0.08, 0.2) * c))
        for _ in range(5):
            x, y = rng.integers(0, c, 2)
            pygame.draw.circle(s, (40, 104, 30, 140), (int(x), int(y)), int(rng.uniform(0.04, 0.1) * c))
        return s

    def _make_vignette(self):
        n = self.board_px
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        d = np.hypot((xx - n / 2) / (n / 2), (yy - n / 2) / (n / 2))
        alpha = (np.clip((d - 0.65) / 0.75, 0, 1) ** 1.6 * 170).astype(np.uint8)
        v = pygame.Surface((n, n), pygame.SRCALPHA)
        v.fill((0, 0, 8, 0))
        pygame.surfarray.pixels_alpha(v)[:] = alpha.T
        return v

    def _background(self, s: EnvState) -> pygame.Surface:
        key = (s.walls.tobytes(), tuple(s.exit_pos))
        if key == self._bg_key:
            return self._bg
        c, n = self.cell, self.n
        bg = pygame.Surface((self.board_px, self.board_px))
        h0 = zlib.crc32(key[0])
        walls = s.walls
        for r in range(n):
            for col in range(n):
                h = (r * 73856093 ^ col * 19349663 ^ h0) % 100
                if walls[r, col]:
                    bg.blit(self.spr[f"roof_{0 if h < 55 else 1 + h % 3}"], (col * c, r * c))
                    if r == n - 1 or not walls[r + 1, col]:
                        fac = self.spr[f"facade_{h % 3}"]
                        bg.blit(fac, (col * c, r * c + c - fac.get_height()))
                else:
                    k = (0 if h < 70 else 4 if h < 80 else 1 if h < 88 else 2 if h < 94 else 5 if h < 97 else 3)
                    bg.blit(self.spr[f"floor_{k}"], (col * c, r * c))
        # ambient occlusion: shadows cast by buildings down / right
        sh = pygame.Surface((c, c), pygame.SRCALPHA)
        for r in range(n):
            for col in range(n):
                if walls[r, col]:
                    continue
                sh.fill((0, 0, 0, 0))
                if r > 0 and walls[r - 1, col]:
                    for k in range(c // 3):
                        pygame.draw.line(sh, (0, 0, 0, int(110 * (1 - k / (c / 3)))), (0, k), (c, k))
                if col > 0 and walls[r, col - 1]:
                    for k in range(c // 5):
                        pygame.draw.line(sh, (0, 0, 0, int(80 * (1 - k / (c / 5)))), (k, 0), (k, c))
                bg.blit(sh, (col * c, r * c))
        # subtle grid so moves stay readable
        grid = pygame.Surface((self.board_px, self.board_px), pygame.SRCALPHA)
        for k in range(1, n):
            pygame.draw.line(grid, (255, 255, 255, 14), (k * c, 0), (k * c, self.board_px))
            pygame.draw.line(grid, (255, 255, 255, 14), (0, k * c), (self.board_px, k * c))
        bg.blit(grid, (0, 0))
        self._bg_key, self._bg = key, bg
        return bg
