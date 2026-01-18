from __future__ import annotations

from dataclasses import dataclass

import pygame
from OpenGL import GL


@dataclass
class HUDInfo:
    health: int
    ammo: int
    ammo_max: int
    weapon: str
    wave: int
    kills: int
    points: int
    fps: float
    alive: bool
    prompt: str = ""
    perks: str = ""
    intermission_left: float = 0.0
    hitmarker: float = 0.0
    blood: float = 0.0


class HUD:
    """Minimal 2D HUD drawn with pygame surfaces + glDrawPixels.

    Note: because we use legacy glDrawPixels, we must be careful to restore
    OpenGL state so it doesn't break the 3D pipeline.
    """

    def __init__(self) -> None:
        self.font = pygame.font.SysFont("consolas", 16)
        self._surface_cache: dict[str, pygame.Surface] = {}
        self._crosshair_surf: pygame.Surface | None = None
        self._blood_cache: dict[tuple[int, int], pygame.Surface] = {}
        self._hitmarker_cache: dict[int, pygame.Surface] = {}

    def _text(self, s: str, color=(255, 255, 255)) -> pygame.Surface:
        key = f"{s}|{color}"
        surf = self._surface_cache.get(key)
        if surf is None:
            surf = self.font.render(s, True, color)
            self._surface_cache[key] = surf
        return surf

    def _blit(self, surf: pygame.Surface, x: int, y: int) -> None:
        # Convert to RGBA bytes (top-left origin) and draw at window pixel coords.
        data = pygame.image.tostring(surf, "RGBA", True)
        GL.glWindowPos2i(int(x), int(y))
        GL.glDrawPixels(surf.get_width(), surf.get_height(), GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)

    def _ensure_crosshair(self) -> pygame.Surface:
        if self._crosshair_surf is not None:
            return self._crosshair_surf

        size = 17
        thickness = 2
        gap = 4
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c = (255, 255, 255, 210)
        cx = size // 2
        cy = size // 2
        # Up
        pygame.draw.line(surf, c, (cx, 0), (cx, cy - gap), thickness)
        # Down
        pygame.draw.line(surf, c, (cx, cy + gap), (cx, size - 1), thickness)
        # Left
        pygame.draw.line(surf, c, (0, cy), (cx - gap, cy), thickness)
        # Right
        pygame.draw.line(surf, c, (cx + gap, cy), (size - 1, cy), thickness)
        # Small center dot
        pygame.draw.circle(surf, (255, 255, 255, 230), (cx, cy), 1)

        self._crosshair_surf = surf
        return surf

    def _draw_crosshair(self) -> None:
        vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
        # vp = (x, y, w, h)
        w = int(vp[2])
        h = int(vp[3])
        surf = self._ensure_crosshair()
        x = (w - surf.get_width()) // 2
        y = (h - surf.get_height()) // 2
        self._blit(surf, x, y)

    def _ensure_blood(self, w: int, h: int) -> pygame.Surface:
        key = (w, h)
        surf = self._blood_cache.get(key)
        if surf is not None:
            return surf
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # Simple vignette-ish blood overlay (cheap but effective)
        surf.fill((0, 0, 0, 0))
        # Dark corners
        corner_alpha = 180
        r = int(max(w, h) * 0.55)
        for cx, cy in ((0, 0), (w, 0), (0, h), (w, h)):
            pygame.draw.circle(surf, (120, 0, 0, corner_alpha), (cx, cy), r)
        # Light full-screen tint
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((120, 0, 0, 70))
        surf.blit(overlay, (0, 0))
        self._blood_cache[key] = surf
        return surf

    def _ensure_hitmarker(self, size: int = 21) -> pygame.Surface:
        surf = self._hitmarker_cache.get(size)
        if surf is not None:
            return surf
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c = (255, 255, 255, 230)
        cx = size // 2
        cy = size // 2
        # X-shaped hitmarker
        pygame.draw.line(surf, c, (cx - 7, cy - 7), (cx - 2, cy - 2), 2)
        pygame.draw.line(surf, c, (cx + 7, cy - 7), (cx + 2, cy - 2), 2)
        pygame.draw.line(surf, c, (cx - 7, cy + 7), (cx - 2, cy + 2), 2)
        pygame.draw.line(surf, c, (cx + 7, cy + 7), (cx + 2, cy + 2), 2)
        self._hitmarker_cache[size] = surf
        return surf

    def render(self, info: HUDInfo) -> None:
        # --- Save OpenGL state that our glDrawPixels calls might disturb ---
        prev_program = GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)
        prev_vao = GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING)
        prev_array_buffer = GL.glGetIntegerv(GL.GL_ARRAY_BUFFER_BINDING)
        prev_element_array = GL.glGetIntegerv(GL.GL_ELEMENT_ARRAY_BUFFER_BINDING)
        prev_active_tex = GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE)
        prev_tex = GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D)

        prev_blend = GL.glIsEnabled(GL.GL_BLEND)
        prev_depth = GL.glIsEnabled(GL.GL_DEPTH_TEST)
        prev_cull = GL.glIsEnabled(GL.GL_CULL_FACE)

        prev_blend_src = GL.glGetIntegerv(GL.GL_BLEND_SRC)
        prev_blend_dst = GL.glGetIntegerv(GL.GL_BLEND_DST)

        # --- Switch to a safe state for pixel drawing ---
        GL.glUseProgram(0)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

        if prev_depth:
            GL.glDisable(GL.GL_DEPTH_TEST)
        if prev_cull:
            GL.glDisable(GL.GL_CULL_FACE)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # --- Draw text and bars ---
        x0, y0 = 18, 18
        lines = [
            f"HP: {info.health:>3}",
            f"{info.weapon}  {info.ammo}/{info.ammo_max}",
            f"Wave: {info.wave}  Kills: {info.kills}",
            f"Points: {info.points}",
            f"FPS: {info.fps:.0f}",
        ]

        y = y0
        for s in lines:
            self._blit(self._text(s), x0, y)
            y += 18

        if not info.alive:
            msg = "YOU DIED - Press ENTER to respawn"
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            w = int(vp[2])
            h = int(vp[3])
            surf = self._text(msg, (255, 80, 80))
            self._blit(surf, (w - surf.get_width()) // 2, (h - surf.get_height()) // 2)
        else:
            # Crosshair
            self._draw_crosshair()

            # Hitmarker
            if info.hitmarker > 0.0:
                vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
                w = int(vp[2]); h = int(vp[3])
                hm = self._ensure_hitmarker()
                alpha = int(255 * min(1.0, info.hitmarker))
                hm2 = hm.copy()
                hm2.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
                self._blit(hm2, (w - hm2.get_width()) // 2, (h - hm2.get_height()) // 2)

            # Blood overlay
            if info.blood > 0.0:
                vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
                w = int(vp[2]); h = int(vp[3])
                blood = self._ensure_blood(w, h).copy()
                alpha = int(255 * min(1.0, info.blood))
                blood.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
                self._blit(blood, 0, 0)

            # Perks (top-right)
            if info.perks:
                vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
                w = int(vp[2])
                surf = self._text(f"Perks: {info.perks}", (180, 220, 255))
                self._blit(surf, max(0, w - surf.get_width() - 18), 18)

            # Intermission timer
            if info.intermission_left > 0.01:
                msg = f"INTERMISSION: {int(info.intermission_left)}s - Buy perks / weapons (E)"
                vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
                w = int(vp[2]); h = int(vp[3])
                surf = self._text(msg, (255, 230, 140))
                self._blit(surf, (w - surf.get_width()) // 2, 18)

            # Interaction prompt (bottom)
            if info.prompt:
                vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
                w = int(vp[2]); h = int(vp[3])
                surf = self._text(info.prompt, (255, 255, 255))
                self._blit(surf, (w - surf.get_width()) // 2, h - 34)

        # --- Restore previous OpenGL state ---
        if not prev_blend:
            GL.glDisable(GL.GL_BLEND)
        GL.glBlendFunc(int(prev_blend_src), int(prev_blend_dst))

        if prev_depth:
            GL.glEnable(GL.GL_DEPTH_TEST)
        if prev_cull:
            GL.glEnable(GL.GL_CULL_FACE)

        GL.glActiveTexture(int(prev_active_tex))
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(prev_tex))

        GL.glBindVertexArray(int(prev_vao))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(prev_array_buffer))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, int(prev_element_array))

        GL.glUseProgram(int(prev_program))
