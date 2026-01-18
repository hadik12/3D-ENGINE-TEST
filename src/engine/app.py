from __future__ import annotations

import pygame
from OpenGL.GL import glViewport

from .input import InputState
from .renderer import Renderer
from .time import Time


class App:
    def __init__(self, width: int, height: int, title: str) -> None:
        pygame.init()

        # --- Anti-aliasing (MSAA) ---
        # Must be set BEFORE creating the OpenGL context.
        # If your GPU/driver doesn't support the requested sample count,
        # it will fall back automatically.
        try:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        except Exception:
            # Some platforms may not expose these attributes; safe to ignore.
            pass

        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        self.width = width
        self.height = height
        self.renderer = Renderer()
        self.time = Time()
        self.input = InputState()
        glViewport(0, 0, width, height)

    def poll(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        self.input.update()
        return True

    def swap(self) -> None:
        pygame.display.flip()

    def shutdown(self) -> None:
        pygame.quit()
