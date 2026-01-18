from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from engine.camera import Camera
from engine.ecs import Collider, Entity, Health, MeshRenderer, RigidBody
from engine.math3d import Vec3
from engine.mesh import Mesh

from . import config
from .weapons import Pistol, Rifle, SMG, Shotgun, Sniper, Weapon


@dataclass
class Perks:
    juggernog: bool = False
    speed_cola: bool = False
    double_tap: bool = False
    stamin_up: bool = False


@dataclass
class PlayerState:
    alive: bool = True


class Player:
    def __init__(self) -> None:
        self.entity = Entity()
        self.entity.transform.position = Vec3(0.0, 0.0, -4.0)
        self.entity.collider = Collider(radius=config.PLAYER_RADIUS)
        self.entity.rigidbody = RigidBody()
        self.entity.health = Health(current=config.PLAYER_MAX_HEALTH, max_health=config.PLAYER_MAX_HEALTH)
        self.state = PlayerState()
        # Economy / perks
        self.points: int = int(config.START_POINTS)
        self.perks = Perks()

        # 1..5 to switch (COD-style: you start with a pistol, others are wall-buys)
        self.weapons = [Pistol(), SMG(), Rifle(), Shotgun(), Sniper()]
        self.weapon_index = 0
        self.unlocked = [True, False, False, False, False]

        # HUD feedback
        self.hitmarker_timer: float = 0.0
        self.blood_timer: float = 0.0

    @property
    def weapon(self) -> Weapon:
        return self.weapons[self.weapon_index]

    @property
    def move_speed(self) -> float:
        return config.PLAYER_SPEED * (config.STAMIN_UP_SPEED_MULT if self.perks.stamin_up else 1.0)

    @property
    def damage_multiplier(self) -> float:
        return config.DOUBLE_TAP_DAMAGE_MULT if self.perks.double_tap else 1.0

    @property
    def reload_multiplier(self) -> float:
        return config.SPEED_COLA_RELOAD_MULT if self.perks.speed_cola else 1.0

    def can_switch_to(self, idx: int) -> bool:
        return 0 <= idx < len(self.weapons) and bool(self.unlocked[idx])

    def unlock_weapon(self, idx: int) -> None:
        if 0 <= idx < len(self.unlocked):
            self.unlocked[idx] = True

    def refill_ammo(self, idx: int | None = None) -> None:
        if idx is None:
            idx = self.weapon_index
        if 0 <= idx < len(self.weapons):
            w = self.weapons[idx]
            w.ammo = w.ammo_max
            w.reloading = 0.0

    def start_reload(self) -> None:
        w = self.weapon
        if w.reloading == 0.0 and w.ammo < w.ammo_max:
            w.reloading = w.reload_time * self.reload_multiplier

    def give_points(self, amount: int) -> None:
        self.points = max(0, int(self.points) + int(amount))

    def spend_points(self, cost: int) -> bool:
        cost = int(cost)
        if self.points < cost:
            return False
        self.points -= cost
        return True

    def set_hitmarker(self) -> None:
        self.hitmarker_timer = 0.12

    def update(self, delta: float, input_state: object, camera: Camera) -> None:
        if not self.state.alive:
            return

        # Timers for HUD overlays
        if self.hitmarker_timer > 0.0:
            self.hitmarker_timer = max(0.0, self.hitmarker_timer - delta)
        if self.blood_timer > 0.0:
            self.blood_timer = max(0.0, self.blood_timer - delta)

        self.weapon.update(delta)
        move = Vec3(0.0, 0.0, 0.0)
        if input_state.key(pygame.K_w):
            move += camera.forward()
        if input_state.key(pygame.K_s):
            move -= camera.forward()
        if input_state.key(pygame.K_a):
            move -= camera.right()
        if input_state.key(pygame.K_d):
            move += camera.right()
        if move.length() > 0:
            move = move.normalized()
        speed = self.move_speed
        self.entity.transform.position += Vec3(move.x, 0.0, move.z) * speed * delta

        sensitivity = 0.002
        camera.yaw += input_state.mouse_dx * sensitivity
        camera.pitch -= input_state.mouse_dy * sensitivity
        camera.pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, camera.pitch))

        if input_state.key(pygame.K_1) and self.can_switch_to(0):
            self.weapon_index = 0
        if input_state.key(pygame.K_2) and self.can_switch_to(1):
            self.weapon_index = 1
        if input_state.key(pygame.K_3) and self.can_switch_to(2):
            self.weapon_index = 2
        if input_state.key(pygame.K_4) and self.can_switch_to(3):
            self.weapon_index = 3
        if input_state.key(pygame.K_5) and self.can_switch_to(4):
            self.weapon_index = 4
        if input_state.key(pygame.K_r):
            self.start_reload()

    def damage(self, amount: float) -> None:
        if self.entity.health is None:
            return
        self.entity.health.current -= amount
        self.blood_timer = 0.65
        if self.entity.health.current <= 0:
            self.state.alive = False

    def reset(self) -> None:
        self.entity.transform.position = Vec3(0.0, 0.0, -4.0)
        if self.entity.health:
            self.entity.health.max_health = config.PLAYER_MAX_HEALTH
            self.entity.health.current = self.entity.health.max_health
        self.state.alive = True

        # Reset economy/perks
        self.points = int(config.START_POINTS)
        self.perks = Perks()
        self.unlocked = [True, False, False, False, False]
        self.weapon_index = 0

        for weapon in self.weapons:
            weapon.ammo = weapon.ammo_max
            weapon.cooldown = 0.0
            weapon.reloading = 0.0

        self.hitmarker_timer = 0.0
        self.blood_timer = 0.0
