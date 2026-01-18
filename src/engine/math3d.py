from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def copy(self) -> "Vec3":
        return Vec3(self.x, self.y, self.z)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        length = self.length()
        if length == 0:
            return Vec3(0.0, 0.0, 0.0)
        return self / length

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z]


class Mat4:
    def __init__(self, values: Iterable[float] | None = None) -> None:
        if values is None:
            self.values = [0.0] * 16
        else:
            self.values = list(values)
            # Defensive: if someone accidentally passes a 4x4 nested list/ndarray,
            # we'd rather fail loudly than render with undefined data.
            if len(self.values) != 16:
                raise ValueError(
                    f"Mat4 expects 16 floats, got {len(self.values)}. "
                    "(You likely passed a nested 4x4 list/array; flatten it or pass 16 numbers.)"
                )

    @staticmethod
    def identity() -> "Mat4":
        values = [0.0] * 16
        for i in range(4):
            values[i * 5] = 1.0
        return Mat4(values)

    @staticmethod
    def translation(pos: Vec3) -> "Mat4":
        mat = Mat4.identity()
        mat.values[12] = pos.x
        mat.values[13] = pos.y
        mat.values[14] = pos.z
        return mat

    @staticmethod
    def scale(scale: Vec3) -> "Mat4":
        mat = Mat4.identity()
        mat.values[0] = scale.x
        mat.values[5] = scale.y
        mat.values[10] = scale.z
        return mat

    @staticmethod
    def rotation_y(angle: float) -> "Mat4":
        c = math.cos(angle)
        s = math.sin(angle)
        return Mat4([
            c, 0.0, -s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            s, 0.0, c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def rotation_x(angle: float) -> "Mat4":
        c = math.cos(angle)
        s = math.sin(angle)
        return Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, c, s, 0.0,
            0.0, -s, c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def rotation_z(angle: float) -> "Mat4":
        """Rotation around Z axis.

        IMPORTANT: this matches the same internal storage & handedness convention
        as rotation_x/rotation_y in this file.
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return Mat4([
            c, s, 0.0, 0.0,
            -s, c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    @staticmethod
    def perspective(fov_deg: float, aspect: float, near: float, far: float) -> "Mat4":
        f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        nf = 1.0 / (near - far)
        return Mat4([
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, (far + near) * nf, -1.0,
            0.0, 0.0, (2 * far * near) * nf, 0.0,
        ])


    @staticmethod
    def ortho(left: float, right: float, bottom: float, top: float, near: float, far: float) -> "Mat4":
        """Orthographic projection matrix.

        This code follows the same row-major convention used by the rest of this file.
        """
        rl = right - left
        tb = top - bottom
        fn = far - near
        if rl == 0 or tb == 0 or fn == 0:
            return Mat4.identity()
        return Mat4([
            2.0 / rl, 0.0, 0.0, 0.0,
            0.0, 2.0 / tb, 0.0, 0.0,
            0.0, 0.0, -2.0 / fn, 0.0,
            -(right + left) / rl, -(top + bottom) / tb, -(far + near) / fn, 1.0,
        ])

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3) -> "Mat4":
        forward = (target - eye).normalized()
        # NOTE: Cross product order matters.
        # Using forward x up produces a mirrored (left-handed) basis, which makes
        # left/right controls feel inverted. We want a standard right-handed basis.
        right = up.normalized().cross(forward).normalized()
        up_vec = forward.cross(right)
        return Mat4([
            right.x, up_vec.x, -forward.x, 0.0,
            right.y, up_vec.y, -forward.y, 0.0,
            right.z, up_vec.z, -forward.z, 0.0,
            -right.dot(eye), -up_vec.dot(eye), forward.dot(eye), 1.0,
        ])

    def __matmul__(self, other: "Mat4") -> "Mat4":
        """Matrix multiplication.

        IMPORTANT: Mat4 in this project is stored in *column-major* order,
        matching OpenGL's default convention and the matrices we construct in
        rotation_*/translation/perspective/look_at.

        That means element (row, col) lives at index: col*4 + row.
        """

        a = self.values
        b = other.values
        result = [0.0] * 16

        # Column-major multiply: C = A * B
        for col in range(4):
            c0 = col * 4
            b0 = b[c0 + 0]
            b1 = b[c0 + 1]
            b2 = b[c0 + 2]
            b3 = b[c0 + 3]

            # Each row r: C[r,c] = sum_i A[r,i] * B[i,c]
            # Using column-major indexing: A[r,i] -> a[i*4 + r]
            result[c0 + 0] = a[0 * 4 + 0] * b0 + a[1 * 4 + 0] * b1 + a[2 * 4 + 0] * b2 + a[3 * 4 + 0] * b3
            result[c0 + 1] = a[0 * 4 + 1] * b0 + a[1 * 4 + 1] * b1 + a[2 * 4 + 1] * b2 + a[3 * 4 + 1] * b3
            result[c0 + 2] = a[0 * 4 + 2] * b0 + a[1 * 4 + 2] * b1 + a[2 * 4 + 2] * b2 + a[3 * 4 + 2] * b3
            result[c0 + 3] = a[0 * 4 + 3] * b0 + a[1 * 4 + 3] * b1 + a[2 * 4 + 3] * b2 + a[3 * 4 + 3] * b3

        return Mat4(result)

    def to_list(self) -> List[float]:
        return list(self.values)
