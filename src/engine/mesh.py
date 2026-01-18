from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import List

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glVertexAttribPointer,
)


@dataclass
class Mesh:
    vao: int
    vbo: int
    vertex_count: int

    def draw(self) -> None:
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)


def _build_mesh(vertices: List[float]) -> Mesh:
    """Build a mesh from an interleaved vertex buffer.

    Vertex format (per-vertex):
      position: vec3
      normal:   vec3
      uv:       vec2

    Total stride: 8 floats.
    """

    # PyOpenGL can't reliably infer array types from plain Python lists.
    vertices_np = np.asarray(vertices, dtype=np.float32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL_STATIC_DRAW)

    stride = 8 * 4
    # position
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, None)
    # normal
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(3 * 4))
    # uv
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, False, stride, ctypes.c_void_p(6 * 4))

    glBindVertexArray(0)
    return Mesh(vao=vao, vbo=vbo, vertex_count=vertices_np.size // 8)


def create_cube(size: float = 1.0) -> Mesh:
    s = size / 2.0

    # Helper to emit a face (2 triangles) with a constant normal
    # and standard UVs.
    def face(v0, v1, v2, v3, n):
        # Tri 1: v0, v1, v2
        # Tri 2: v0, v2, v3
        (nx, ny, nz) = n
        # UVs (v0..v3): (0,0), (1,0), (1,1), (0,1)
        return [
            *v0, nx, ny, nz, 0.0, 0.0,
            *v1, nx, ny, nz, 1.0, 0.0,
            *v2, nx, ny, nz, 1.0, 1.0,
            *v0, nx, ny, nz, 0.0, 0.0,
            *v2, nx, ny, nz, 1.0, 1.0,
            *v3, nx, ny, nz, 0.0, 1.0,
        ]

    # Cube corners
    p000 = (-s, -s, -s)
    p001 = (-s, -s,  s)
    p010 = (-s,  s, -s)
    p011 = (-s,  s,  s)
    p100 = ( s, -s, -s)
    p101 = ( s, -s,  s)
    p110 = ( s,  s, -s)
    p111 = ( s,  s,  s)

    vertices: List[float] = []

    # -Z (back)
    vertices += face(p000, p100, p110, p010, (0.0, 0.0, -1.0))
    # +Z (front)
    vertices += face(p001, p011, p111, p101, (0.0, 0.0, 1.0))
    # -X (left)
    vertices += face(p000, p010, p011, p001, (-1.0, 0.0, 0.0))
    # +X (right)
    vertices += face(p100, p101, p111, p110, (1.0, 0.0, 0.0))
    # -Y (bottom)
    vertices += face(p000, p001, p101, p100, (0.0, -1.0, 0.0))
    # +Y (top)
    vertices += face(p010, p110, p111, p011, (0.0, 1.0, 0.0))

    return _build_mesh(vertices)


def create_plane(size: float = 10.0) -> Mesh:
    s = size / 2.0
    n = (0.0, 1.0, 0.0)

    # Plane lies on XZ, y=0
    # UVs go 0..1 across the plane, and we can tile in the shader via u_uvScale.
    vertices = [
        -s, 0.0, -s,  *n, 0.0, 0.0,
         s, 0.0, -s,  *n, 1.0, 0.0,
         s, 0.0,  s,  *n, 1.0, 1.0,
        -s, 0.0, -s,  *n, 0.0, 0.0,
         s, 0.0,  s,  *n, 1.0, 1.0,
        -s, 0.0,  s,  *n, 0.0, 1.0,
    ]
    return _build_mesh(vertices)


def create_quad(size: float = 1.0) -> Mesh:
    """Create a quad in the XY plane (z=0), centered at origin.

    Vertex format matches the rest of this engine: pos(3), normal(3), uv(2).
    Normal points towards +Z.

    This is handy for sprites/billboards (muzzle flash, smoke) and also
    for screen-space post-processing (when rendered with an identity model).
    """
    s = size / 2.0
    n = (0.0, 0.0, 1.0)
    # Two triangles (CCW)
    vertices = [
        -s, -s, 0.0,  *n, 0.0, 0.0,
         s, -s, 0.0,  *n, 1.0, 0.0,
         s,  s, 0.0,  *n, 1.0, 1.0,
        -s, -s, 0.0,  *n, 0.0, 0.0,
         s,  s, 0.0,  *n, 1.0, 1.0,
        -s,  s, 0.0,  *n, 0.0, 1.0,
    ]
    return _build_mesh(vertices)
