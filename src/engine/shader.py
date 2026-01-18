from __future__ import annotations

from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_VERTEX_SHADER,
    glAttachShader,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteShader,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glLinkProgram,
    glShaderSource,
    glUseProgram,
)


class Shader:
    def __init__(self, vertex_src: str, fragment_src: str) -> None:
        self.program = glCreateProgram()
        vert = self._compile(GL_VERTEX_SHADER, vertex_src)
        frag = self._compile(GL_FRAGMENT_SHADER, fragment_src)
        glAttachShader(self.program, vert)
        glAttachShader(self.program, frag)
        glLinkProgram(self.program)
        if glGetProgramiv(self.program, GL_LINK_STATUS) != 1:
            raise RuntimeError(glGetProgramInfoLog(self.program).decode("utf-8"))
        glDeleteShader(vert)
        glDeleteShader(frag)

    def _compile(self, shader_type: int, source: str) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != 1:
            raise RuntimeError(glGetShaderInfoLog(shader).decode("utf-8"))
        return shader

    def use(self) -> None:
        glUseProgram(self.program)
