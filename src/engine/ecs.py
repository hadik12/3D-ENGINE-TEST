from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .math3d import Vec3


@dataclass
class Transform:
    position: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    rotation: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    scale: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))


@dataclass
class MeshRenderer:
    mesh: object
    color: tuple[float, float, float]


@dataclass
class Collider:
    radius: float


@dataclass
class RigidBody:
    velocity: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))


@dataclass
class AI:
    attack_cooldown: float = 0.0


@dataclass
class Health:
    current: float
    max_health: float


class Entity:
    _next_id = 1

    def __init__(self) -> None:
        self.id = Entity._next_id
        Entity._next_id += 1
        self.transform = Transform()
        self.mesh: Optional[MeshRenderer] = None
        self.collider: Optional[Collider] = None
        self.rigidbody: Optional[RigidBody] = None
        self.ai: Optional[AI] = None
        self.health: Optional[Health] = None
        self.tags: Dict[str, bool] = {}

    def has_tag(self, tag: str) -> bool:
        return self.tags.get(tag, False)

    def add_tag(self, tag: str) -> None:
        self.tags[tag] = True
