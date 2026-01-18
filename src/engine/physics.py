from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .math3d import Vec3


@dataclass
class AABB:
    min: Vec3
    max: Vec3

    def contains(self, point: Vec3) -> bool:
        return (
            self.min.x <= point.x <= self.max.x
            and self.min.y <= point.y <= self.max.y
            and self.min.z <= point.z <= self.max.z
        )


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def sphere_aabb_intersect(center: Vec3, radius: float, box: AABB) -> bool:
    closest = Vec3(
        clamp(center.x, box.min.x, box.max.x),
        clamp(center.y, box.min.y, box.max.y),
        clamp(center.z, box.min.z, box.max.z),
    )
    delta = center - closest
    return delta.length() <= radius


def resolve_sphere_aabb(center: Vec3, radius: float, boxes: Iterable[AABB]) -> Vec3:
    adjusted = center.copy()
    for box in boxes:
        if sphere_aabb_intersect(adjusted, radius, box):
            if adjusted.x < box.min.x:
                adjusted.x = box.min.x - radius
            elif adjusted.x > box.max.x:
                adjusted.x = box.max.x + radius
            if adjusted.z < box.min.z:
                adjusted.z = box.min.z - radius
            elif adjusted.z > box.max.z:
                adjusted.z = box.max.z + radius
    return adjusted


def ray_sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: float) -> Optional[float]:
    oc = origin - center
    a = direction.dot(direction)
    b = 2.0 * oc.dot(direction)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_d = discriminant ** 0.5
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    if t1 > 0:
        return t1
    if t2 > 0:
        return t2
    return None


def raycast_spheres(
    origin: Vec3,
    direction: Vec3,
    spheres: Iterable[Tuple[Vec3, float]],
) -> Optional[Tuple[int, float]]:
    closest_idx = None
    closest_dist = float("inf")
    for idx, (center, radius) in enumerate(spheres):
        hit = ray_sphere(origin, direction, center, radius)
        if hit is not None and hit < closest_dist:
            closest_dist = hit
            closest_idx = idx
    if closest_idx is None:
        return None
    return closest_idx, closest_dist
