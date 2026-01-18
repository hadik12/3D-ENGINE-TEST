from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame
from OpenGL import GL


@dataclass
class Texture:
    id: int
    width: int
    height: int

    def bind(self, unit: int = 0) -> None:
        GL.glActiveTexture(GL.GL_TEXTURE0 + int(unit))
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(self.id))

    @staticmethod
    def load(path: str | Path, repeat: bool = True, mipmaps: bool = True) -> "Texture":
        """Load an image file into an OpenGL 2D texture.

        Uses pygame to decode the image. The texture is uploaded as RGBA.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Texture not found: {p}")

        surf = pygame.image.load(str(p)).convert_alpha()
        width, height = surf.get_size()
        # Flip vertically to match OpenGL UV convention.
        surf = pygame.transform.flip(surf, False, True)
        data = pygame.image.tostring(surf, "RGBA", True)

        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)

        wrap = GL.GL_REPEAT if repeat else GL.GL_CLAMP_TO_EDGE
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap)

        if mipmaps:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        else:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        # Optional: anisotropic filtering (sharper textures at glancing angles)
        # Works on most modern GPUs; safe to skip if extension isn't present.
        try:
            max_aniso = None
            if hasattr(GL, "GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT") and hasattr(GL, "GL_TEXTURE_MAX_ANISOTROPY_EXT"):
                max_aniso = GL.glGetFloatv(GL.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                # Clamp to a reasonable value to avoid perf spikes.
                aniso = float(min(8.0, max_aniso))
                GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso)
        except Exception:
            # Extension not available or driver rejected the call.
            pass

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            data,
        )

        if mipmaps:
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

        # Unbind
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return Texture(id=int(tex_id), width=width, height=height)

    def destroy(self) -> None:
        if self.id:
            GL.glDeleteTextures([int(self.id)])
            self.id = 0
