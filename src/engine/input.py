from __future__ import annotations

import pygame


class InputState:
    def __init__(self) -> None:
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.keys = pygame.key.get_pressed()
        self.prev_keys = self.keys
        self.mouse_buttons = pygame.mouse.get_pressed(3)
        self.prev_mouse_buttons = self.mouse_buttons

    def update(self) -> None:
        self.mouse_dx, self.mouse_dy = pygame.mouse.get_rel()

        # Keep previous snapshots to allow "pressed" (edge) queries
        self.prev_keys = self.keys
        self.prev_mouse_buttons = self.mouse_buttons

        self.keys = pygame.key.get_pressed()
        self.mouse_buttons = pygame.mouse.get_pressed(3)

    def key(self, key: int) -> bool:
        return bool(self.keys[key])

    def key_pressed(self, key: int) -> bool:
        # True only on the frame the key becomes down
        return bool(self.keys[key]) and not bool(self.prev_keys[key])

    def mouse(self, button: int) -> bool:
        return bool(self.mouse_buttons[button])

    def mouse_pressed(self, button: int) -> bool:
        return bool(self.mouse_buttons[button]) and not bool(self.prev_mouse_buttons[button])
