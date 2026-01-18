from __future__ import annotations

import math

from .math3d import Mat4, Vec3


class Camera:
    def __init__(self, position: Vec3) -> None:
        self.position = position
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 70.0

    def forward(self) -> Vec3:
        cos_pitch = math.cos(self.pitch)
        return Vec3(
            math.sin(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            math.cos(self.yaw) * cos_pitch,
        ).normalized()

    def right(self) -> Vec3:
        forward = self.forward()
        return Vec3(forward.z, 0.0, -forward.x).normalized()

    def view_matrix(self) -> Mat4:
        target = self.position + self.forward()
        return Mat4.look_at(self.position, target, Vec3(0.0, 1.0, 0.0))
