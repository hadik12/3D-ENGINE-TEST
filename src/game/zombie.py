from __future__ import annotations

from engine.ecs import AI, Collider, Entity, Health
from engine.math3d import Vec3
from engine.mesh import Mesh

from . import config


def create_zombie(position: Vec3, mesh: Mesh, wave: int) -> Entity:
    zombie = Entity()
    zombie.transform.position = position
    zombie.collider = Collider(radius=0.5)
    zombie.ai = AI(attack_cooldown=0.0)
    health = config.ZOMBIE_BASE_HEALTH + wave * 5
    zombie.health = Health(current=health, max_health=health)
    zombie.add_tag("zombie")
    return zombie
