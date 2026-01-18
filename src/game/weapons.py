from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from engine.math3d import Vec3
from engine.physics import raycast_spheres

from . import config


@dataclass
class HitResult:
    target_index: int
    distance: float


@dataclass
class RayResult:
    direction: Vec3
    distance: float
    target_index: Optional[int]


@dataclass
class FireResult:
    hits: List[HitResult]
    rays: List[RayResult]


def _spread_direction(direction: Vec3, spread: float, seed: int) -> Vec3:
    """Apply a small deterministic spread around `direction`.

    `spread` is a small number (0..0.12-ish) that defines a cone size.
    We build a local (right, up) basis from the direction and offset in that plane.
    """
    if spread <= 0.0:
        return direction

    # Deterministic pseudo-random in [-1..1]
    x = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    y = (x * 1664525 + 1013904223) & 0xFFFFFFFF
    rx = ((x & 0xFFFF) / 65535.0) * 2.0 - 1.0
    ry = (((y >> 16) & 0xFFFF) / 65535.0) * 2.0 - 1.0

    world_up = Vec3(0.0, 1.0, 0.0)
    right = world_up.cross(direction)
    if right.length() < 1e-6:
        right = Vec3(1.0, 0.0, 0.0)
    right = right.normalized()
    up = direction.cross(right).normalized()

    return (direction + right * (rx * spread) + up * (ry * spread)).normalized()


class Weapon:
    def __init__(
        self,
        name: str,
        ammo_max: int,
        fire_rate: float,
        reload_time: float,
        damage: int,
        spread: float = 0.0,
    ) -> None:
        self.name = name
        self.ammo_max = ammo_max
        self.ammo = ammo_max
        self.fire_rate = fire_rate
        self.reload_time = reload_time
        self.damage = damage
        self.spread = spread
        self.cooldown = 0.0
        self.reloading = 0.0
        self._shot_counter = 0

    def update(self, delta: float) -> None:
        if self.cooldown > 0:
            self.cooldown = max(0.0, self.cooldown - delta)
        if self.reloading > 0:
            self.reloading = max(0.0, self.reloading - delta)
            if self.reloading == 0.0:
                self.ammo = self.ammo_max

    def can_fire(self) -> bool:
        return self.cooldown == 0.0 and self.reloading == 0.0 and self.ammo > 0

    def start_reload(self) -> None:
        if self.reloading == 0.0 and self.ammo < self.ammo_max:
            self.reloading = self.reload_time

    def fire(self, origin: Vec3, direction: Vec3, targets: List[tuple[Vec3, float]]) -> FireResult:
        raise NotImplementedError


class Pistol(Weapon):
    def __init__(self) -> None:
        super().__init__(
            "Pistol",
            config.PISTOL_AMMO,
            config.PISTOL_FIRE_RATE,
            config.PISTOL_RELOAD_TIME,
            damage=config.PISTOL_DAMAGE,
            spread=0.0,
        )

    def fire(self, origin: Vec3, direction: Vec3, targets: List[tuple[Vec3, float]]) -> FireResult:
        if not self.can_fire():
            return FireResult(hits=[], rays=[])

        self.ammo -= 1
        self.cooldown = self.fire_rate
        self._shot_counter += 1

        hit = raycast_spheres(origin, direction, targets)
        if hit is None:
            return FireResult(hits=[], rays=[RayResult(direction=direction, distance=config.BULLET_MAX_DISTANCE, target_index=None)])

        idx, dist = hit
        return FireResult(
            hits=[HitResult(target_index=idx, distance=dist)],
            rays=[RayResult(direction=direction, distance=dist, target_index=idx)],
        )


class SMG(Weapon):
    """Close-range automatic weapon: high spread + big magazine."""

    def __init__(self) -> None:
        super().__init__(
            "SMG",
            config.SMG_AMMO,
            config.SMG_FIRE_RATE,
            config.SMG_RELOAD_TIME,
            damage=config.SMG_DAMAGE,
            spread=config.SMG_SPREAD,
        )

    def fire(self, origin: Vec3, direction: Vec3, targets: List[tuple[Vec3, float]]) -> FireResult:
        if not self.can_fire():
            return FireResult(hits=[], rays=[])

        self.ammo -= 1
        self.cooldown = self.fire_rate
        self._shot_counter += 1

        d = _spread_direction(direction, self.spread, self._shot_counter)
        hit = raycast_spheres(origin, d, targets)
        if hit is None:
            return FireResult(hits=[], rays=[RayResult(direction=d, distance=config.BULLET_MAX_DISTANCE, target_index=None)])

        idx, dist = hit
        return FireResult(hits=[HitResult(target_index=idx, distance=dist)], rays=[RayResult(direction=d, distance=dist, target_index=idx)])


class Rifle(Weapon):
    """Mid-range rifle: low spread, steady damage."""

    def __init__(self) -> None:
        super().__init__(
            "Rifle",
            config.RIFLE_AMMO,
            config.RIFLE_FIRE_RATE,
            config.RIFLE_RELOAD_TIME,
            damage=config.RIFLE_DAMAGE,
            spread=config.RIFLE_SPREAD,
        )

    def fire(self, origin: Vec3, direction: Vec3, targets: List[tuple[Vec3, float]]) -> FireResult:
        if not self.can_fire():
            return FireResult(hits=[], rays=[])

        self.ammo -= 1
        self.cooldown = self.fire_rate
        self._shot_counter += 1

        d = _spread_direction(direction, self.spread, 9000 + self._shot_counter)
        hit = raycast_spheres(origin, d, targets)
        if hit is None:
            return FireResult(hits=[], rays=[RayResult(direction=d, distance=config.BULLET_MAX_DISTANCE, target_index=None)])

        idx, dist = hit
        return FireResult(hits=[HitResult(target_index=idx, distance=dist)], rays=[RayResult(direction=d, distance=dist, target_index=idx)])


class Sniper(Weapon):
    """High damage, slow fire, small magazine."""

    def __init__(self) -> None:
        super().__init__(
            "Sniper",
            config.SNIPER_AMMO,
            config.SNIPER_FIRE_RATE,
            config.SNIPER_RELOAD_TIME,
            damage=config.SNIPER_DAMAGE,
            spread=config.SNIPER_SPREAD,
        )

    def fire(self, origin: Vec3, direction: Vec3, targets: List[tuple[Vec3, float]]) -> FireResult:
        if not self.can_fire():
            return FireResult(hits=[], rays=[])

        self.ammo -= 1
        self.cooldown = self.fire_rate
        self._shot_counter += 1

        d = _spread_direction(direction, self.spread, 20000 + self._shot_counter)
        hit = raycast_spheres(origin, d, targets)
        if hit is None:
            return FireResult(hits=[], rays=[RayResult(direction=d, distance=config.BULLET_MAX_DISTANCE, target_index=None)])

        idx, dist = hit
        return FireResult(hits=[HitResult(target_index=idx, distance=dist)], rays=[RayResult(direction=d, distance=dist, target_index=idx)])


class Shotgun(Weapon):
    def __init__(self) -> None:
        super().__init__(
            "Shotgun",
            config.SHOTGUN_AMMO,
            config.SHOTGUN_FIRE_RATE,
            config.SHOTGUN_RELOAD_TIME,
            damage=config.SHOTGUN_DAMAGE,
            spread=config.SHOTGUN_SPREAD,
        )

    def fire(self, origin: Vec3, direction: Vec3, targets: List[tuple[Vec3, float]]) -> FireResult:
        if not self.can_fire():
            return FireResult(hits=[], rays=[])

        self.ammo -= 1
        self.cooldown = self.fire_rate
        self._shot_counter += 1

        hits: List[HitResult] = []
        rays: List[RayResult] = []

        for pellet in range(config.SHOTGUN_PELLETS):
            d = _spread_direction(direction, self.spread, self._shot_counter * 100 + pellet)
            hit = raycast_spheres(origin, d, targets)
            if hit is None:
                rays.append(RayResult(direction=d, distance=config.BULLET_MAX_DISTANCE, target_index=None))
                continue

            idx, dist = hit
            hits.append(HitResult(target_index=idx, distance=dist))
            rays.append(RayResult(direction=d, distance=dist, target_index=idx))

        return FireResult(hits=hits, rays=rays)
