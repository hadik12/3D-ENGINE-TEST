from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pygame

from engine.camera import Camera
from engine.ecs import Entity
from engine.math3d import Mat4, Vec3
from engine.mesh import create_cube, create_plane, create_quad
from engine.physics import AABB, resolve_sphere_aabb
from engine.renderer import RenderItem, Renderer
from engine.texture import Texture

from . import config
from .player import Player
from .weapons import Weapon
from .zombie import create_zombie


@dataclass
class Wall:
    position: Vec3
    size: Vec3
    aabb: AABB


@dataclass
class Barrier:
    position: Vec3
    size: Vec3
    aabb: AABB
    max_health: float = config.BARRIER_MAX_HEALTH
    health: float = config.BARRIER_MAX_HEALTH

    def alive(self) -> bool:
        return self.health > 0.0


@dataclass
class InteractSpot:
    kind: str  # 'wall_weapon', 'perk', 'ammo'
    position: Vec3
    radius: float
    title: str
    cost: int
    weapon_index: Optional[int] = None
    perk_id: Optional[str] = None


@dataclass
class Tracer:
    start: Vec3
    end: Vec3
    age: float = 0.0
    lifetime: float = config.TRACER_LIFETIME

    def alive(self) -> bool:
        return self.age < self.lifetime


@dataclass
class MuzzleFlash:
    position: Vec3
    age: float = 0.0
    lifetime: float = 0.06

    def alive(self) -> bool:
        return self.age < self.lifetime


@dataclass
class SmokeParticle:
    position: Vec3
    velocity: Vec3
    size: float
    age: float = 0.0
    lifetime: float = 0.9

    def alive(self) -> bool:
        return self.age < self.lifetime


@dataclass
class SmokeParticle:
    position: Vec3
    velocity: Vec3
    size: float
    age: float = 0.0
    lifetime: float = 0.85

    def alive(self) -> bool:
        return self.age < self.lifetime


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(a: Tuple[float, float, float], b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t), _lerp(a[2], b[2], t))


class World:
    def __init__(self) -> None:
        self.cube_mesh = create_cube(1.0)
        # We render the ground as tiles around the player ("infinite" world).
        self.ground_tile_mesh = create_plane(config.GROUND_TILE_SIZE)
        # For sprites/billboards (muzzle flash, smoke)
        self.quad_mesh = create_quad(1.0)
        # For billboards/sprites (muzzle flash, smoke)
        self.quad_mesh = create_quad(1.0)

        # Textures (procedural, shipped with the project)
        self.tex_grass = Texture.load("assets/textures/grass.png", repeat=True, mipmaps=True)
        self.tex_dirt = Texture.load("assets/textures/dirt.png", repeat=True, mipmaps=True)
        self.tex_concrete = Texture.load("assets/textures/concrete.png", repeat=True, mipmaps=True)
        self.tex_brick = Texture.load("assets/textures/brick.png", repeat=True, mipmaps=True)
        self.tex_zombie = Texture.load("assets/textures/zombie.png", repeat=True, mipmaps=True)
        self.tex_metal = Texture.load("assets/textures/metal.png", repeat=True, mipmaps=True)

        self.player = Player()
        self.camera = Camera(self.player.entity.transform.position)
        self.walls: List[Wall] = []
        self.barriers: List[Barrier] = []
        self.interact_spots: List[InteractSpot] = []
        self.interact_prompt: str = ""
        self.zombies: List[Entity] = []
        self.tracers: List[Tracer] = []
        self.muzzle_flashes: List[MuzzleFlash] = []
        self.smoke_particles: List[SmokeParticle] = []

        self.wave = 0
        self.kills = 0

        # Waves: combat + a small intermission for buying
        self.intermission_left: float = 0.0
        self.in_intermission: bool = False

        # Environment (day/night + fog)
        self._day_target = 1.0  # 1 = day, 0 = night
        self.day_factor = 1.0
        self.fog_enabled = True

        # Extra rendering toggles
        self.godrays_enabled = True
        self.msaa_enabled = True

        # Sun movement (purely visual - the sun slowly travels even if you toggle day/night)
        self.sun_time = 0.0
        self.current_light_dir = Vec3(-0.25, -1.0, -0.35)

        self.spawn_walls()
        self.spawn_barriers_and_shop()
        self.start_wave()

    def _tile_hash(self, x: int, z: int) -> int:
        # Deterministic hash for procedural variety (no RNG state needed)
        return (x * 73856093) ^ (z * 19349663)

    def spawn_walls(self) -> None:
        layouts = [
            (Vec3(-3.0, 0.5, -2.0), Vec3(2.0, 1.0, 0.5)),
            (Vec3(2.5, 0.5, 3.0), Vec3(1.5, 1.0, 0.5)),
            (Vec3(0.0, 0.5, 0.0), Vec3(0.6, 1.0, 3.0)),
        ]
        for pos, size in layouts:
            min_v = Vec3(pos.x - size.x / 2, 0.0, pos.z - size.z / 2)
            max_v = Vec3(pos.x + size.x / 2, size.y, pos.z + size.z / 2)
            self.walls.append(Wall(position=pos, size=size, aabb=AABB(min_v, max_v)))

    def spawn_barriers_and_shop(self) -> None:
        """Place COD-like interactables near spawn.

        This stays simple on purpose (no external UI) - you just walk close and press E.
        """

        self.barriers = []
        self.interact_spots = []

        def add_barrier(pos: Vec3, size: Vec3) -> None:
            min_v = Vec3(pos.x - size.x / 2, 0.0, pos.z - size.z / 2)
            max_v = Vec3(pos.x + size.x / 2, size.y, pos.z + size.z / 2)
            self.barriers.append(Barrier(position=pos, size=size, aabb=AABB(min_v, max_v)))

        # Four repairable windows around spawn (zombies bump into them first)
        add_barrier(Vec3(-6.5, 0.75, -2.0), Vec3(2.4, 1.5, 0.35))
        add_barrier(Vec3( 6.5, 0.75,  1.5), Vec3(2.4, 1.5, 0.35))
        add_barrier(Vec3( 0.0, 0.75, -6.5), Vec3(2.4, 1.5, 0.35))
        add_barrier(Vec3( 0.0, 0.75,  6.5), Vec3(2.4, 1.5, 0.35))

        # Wall-buys
        self.interact_spots.append(
            InteractSpot(
                kind="wall_weapon",
                position=Vec3(-3.2, 0.0, -2.0),
                radius=1.8,
                title="SMG",
                cost=int(config.COST_SMG),
                weapon_index=1,
            )
        )
        self.interact_spots.append(
            InteractSpot(
                kind="wall_weapon",
                position=Vec3(2.4, 0.0, -2.0),
                radius=1.8,
                title="Rifle",
                cost=int(config.COST_RIFLE),
                weapon_index=2,
            )
        )
        self.interact_spots.append(
            InteractSpot(
                kind="wall_weapon",
                position=Vec3(-2.6, 0.0, 2.8),
                radius=1.8,
                title="Shotgun",
                cost=int(config.COST_SHOTGUN),
                weapon_index=3,
            )
        )
        self.interact_spots.append(
            InteractSpot(
                kind="wall_weapon",
                position=Vec3(3.1, 0.0, 3.0),
                radius=1.8,
                title="Sniper",
                cost=int(config.COST_SNIPER),
                weapon_index=4,
            )
        )

        # Ammo crate (refill current weapon)
        self.interact_spots.append(
            InteractSpot(
                kind="ammo",
                position=Vec3(0.0, 0.0, -1.0),
                radius=1.9,
                title="Ammo Crate",
                cost=int(config.COST_AMMO_REFILL),
            )
        )

        # Perk machines
        self.interact_spots.append(
            InteractSpot(
                kind="perk",
                position=Vec3(-5.0, 0.0, 4.5),
                radius=2.0,
                title="Juggernog",
                cost=int(config.COST_JUGGERNOG),
                perk_id="juggernog",
            )
        )
        self.interact_spots.append(
            InteractSpot(
                kind="perk",
                position=Vec3(5.0, 0.0, 4.5),
                radius=2.0,
                title="Speed Cola",
                cost=int(config.COST_SPEED_COLA),
                perk_id="speed_cola",
            )
        )
        self.interact_spots.append(
            InteractSpot(
                kind="perk",
                position=Vec3(-5.0, 0.0, -4.5),
                radius=2.0,
                title="Double Tap",
                cost=int(config.COST_DOUBLE_TAP),
                perk_id="double_tap",
            )
        )
        self.interact_spots.append(
            InteractSpot(
                kind="perk",
                position=Vec3(5.0, 0.0, -4.5),
                radius=2.0,
                title="Stamin-Up",
                cost=int(config.COST_STAMIN_UP),
                perk_id="stamin_up",
            )
        )

    def start_wave(self) -> None:
        self.in_intermission = False
        self.intermission_left = 0.0
        self.wave += 1
        count = config.WAVE_BASE_COUNT + (self.wave - 1) * config.WAVE_GROWTH
        self.zombies = []
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            radius = config.WAVE_SPAWN_RADIUS
            center = self.player.entity.transform.position
            spawn = center + Vec3(math.cos(angle) * radius, 0.0, math.sin(angle) * radius)
            zombie = create_zombie(spawn, self.cube_mesh, self.wave)
            self.zombies.append(zombie)

    def handle_environment_input(self, input_state: object) -> None:
        # Toggle day/night
        if getattr(input_state, "key_pressed", None) and input_state.key_pressed(pygame.K_n):
            self._day_target = 0.0 if self._day_target > 0.5 else 1.0

        # Toggle fog
        if getattr(input_state, "key_pressed", None) and input_state.key_pressed(pygame.K_f):
            self.fog_enabled = not self.fog_enabled

        # Toggle volumetric sun rays (god rays)
        if getattr(input_state, "key_pressed", None) and input_state.key_pressed(pygame.K_g):
            self.godrays_enabled = not self.godrays_enabled

        # Toggle MSAA (anti-aliasing). This just enables/disables GL_MULTISAMPLE.
        if getattr(input_state, "key_pressed", None) and input_state.key_pressed(pygame.K_m):
            self.msaa_enabled = not self.msaa_enabled

        # Toggle volumetric sun rays (god rays)
        if getattr(input_state, "key_pressed", None) and input_state.key_pressed(pygame.K_g):
            self.godrays_enabled = not self.godrays_enabled

        # Toggle MSAA (OpenGL multisampling)
        if getattr(input_state, "key_pressed", None) and input_state.key_pressed(pygame.K_m):
            self.msaa_enabled = not self.msaa_enabled

    def apply_environment(self, renderer: Renderer, delta: float) -> Tuple[Tuple[float, float, float], Mat4]:
        """Apply environment settings to the renderer.

        Returns:
          clear_color, light_space_matrix
        """

        # Smooth transition day <-> night
        speed = 2.0
        self.day_factor = _lerp(self.day_factor, self._day_target, min(1.0, delta * speed))

        # Day settings (tuned to be less overbright)
        # NOTE: because we use tonemapping, lowering exposure is the most
        # effective way to reduce "washed out" look.
        day_clear = (0.45, 0.65, 0.95)
        day_light = (0.92, 0.90, 0.86)
        day_ambient = (0.16, 0.16, 0.18)
        day_fog = (0.55, 0.70, 0.95)
        day_fog_start, day_fog_end = 12.0, 44.0
        day_shadow = 0.92
        day_exposure = 0.75

        # Night settings
        night_clear = (0.02, 0.03, 0.06)
        night_light = (0.25, 0.30, 0.42)
        night_ambient = (0.06, 0.06, 0.10)
        night_fog = (0.02, 0.03, 0.06)
        night_fog_start, night_fog_end = 6.0, 26.0
        night_shadow = 0.80
        night_exposure = 0.45

        f = float(self.day_factor)
        clear = _lerp3(night_clear, day_clear, f)
        renderer.light_color = _lerp3(night_light, day_light, f)
        renderer.ambient_color = _lerp3(night_ambient, day_ambient, f)
        renderer.shadow_strength = _lerp(night_shadow, day_shadow, f)
        renderer.exposure = _lerp(night_exposure, day_exposure, f)

        # Fog
        if self.fog_enabled:
            renderer.fog_color = _lerp3(night_fog, day_fog, f)
            renderer.fog_start = _lerp(night_fog_start, day_fog_start, f)
            renderer.fog_end = _lerp(night_fog_end, day_fog_end, f)
        else:
            renderer.fog_color = clear
            renderer.fog_start = 1e9
            renderer.fog_end = 1e9

        # Sun/moon direction (acts as the main directional light)
        # We treat u_lightDir as the direction the light rays travel (from sky -> ground).
        # The sun slowly rotates; at night we blend to a moon direction.
        self.sun_time += delta * 0.35
        az = self.sun_time

        # Day sun: higher in the sky when azimuth is near "midday".
        elev = max(math.sin(az), 0.0)
        sx = math.cos(az) * 0.6
        sz = math.sin(az) * 0.6
        sy = -0.35 - 0.75 * elev
        sun_to_scene = Vec3(sx, sy, sz).normalized()

        # Moon is opposite direction and a bit steeper downwards (so you still get shadows at night).
        moon_to_scene = Vec3(-sx, -0.85, -sz).normalized()

        ld = _lerp3(moon_to_scene.as_list(), sun_to_scene.as_list(), f)
        renderer.light_dir = (ld[0], ld[1], ld[2])
        self.current_light_dir = Vec3(ld[0], ld[1], ld[2])

        # Extra renderer toggles
        renderer.godrays_enabled = bool(self.godrays_enabled and self.fog_enabled)
        renderer.godrays_strength = 0.15 + 0.85 * f
        renderer.godrays_steps = 14
        renderer.msaa_enabled = bool(self.msaa_enabled)

        # Shadow camera follows the player (keeps high detail where you're fighting)
        center = self.player.entity.transform.position
        ld = Vec3(renderer.light_dir[0], renderer.light_dir[1], renderer.light_dir[2]).normalized()
        light_pos = center - ld * 22.0 + Vec3(0.0, 10.0, 0.0)
        light_view = Mat4.look_at(light_pos, center, Vec3(0.0, 1.0, 0.0))

        ortho = 22.0
        light_proj = Mat4.ortho(-ortho, ortho, -ortho, ortho, 1.0, 70.0)
        light_space = light_proj @ light_view

        return clear, light_space

    def update(self, delta: float, input_state: object) -> None:
        self.player.update(delta, input_state, self.camera)
        self.camera.position = self.player.entity.transform.position + Vec3(0.0, 0.6, 0.0)
        self.handle_player_collision()
        self.handle_environment_input(input_state)

        # Interaction prompt + actions (E)
        self._update_interaction_prompt()
        if getattr(input_state, "key_pressed", None) and input_state.key_pressed(pygame.K_e):
            self.try_interact()

        # Combat/intermission loop
        if self.in_intermission:
            self.intermission_left = max(0.0, self.intermission_left - delta)
            if self.intermission_left <= 0.01:
                self.start_wave()
        else:
            self.update_zombies(delta)
            if not self.zombies:
                self._start_intermission()

        self.update_tracers(delta)
        self._update_muzzle_fx(delta)

    def handle_player_collision(self) -> None:
        boxes = [wall.aabb for wall in self.walls]
        transform = self.player.entity.transform
        transform.position = resolve_sphere_aabb(transform.position, config.PLAYER_RADIUS, boxes)

    def _barrier_at(self, pos: Vec3, extra: float = 0.15) -> Optional[Barrier]:
        """Returns a barrier whose AABB contains the given point (XZ), if any."""
        for b in self.barriers:
            if not b.alive():
                continue
            if (b.aabb.min.x - extra <= pos.x <= b.aabb.max.x + extra) and (b.aabb.min.z - extra <= pos.z <= b.aabb.max.z + extra):
                return b
        return None

    def player_perks_string(self) -> str:
        p = self.player.perks
        perks: List[str] = []
        if p.juggernog:
            perks.append("Juggernog")
        if p.speed_cola:
            perks.append("Speed Cola")
        if p.double_tap:
            perks.append("Double Tap")
        if p.stamin_up:
            perks.append("Stamin-Up")
        return ", ".join(perks) if perks else "нет"

    def _start_intermission(self) -> None:
        self.in_intermission = True
        self.intermission_left = float(config.INTERMISSION_SECONDS)

    def _has_perk(self, perk_id: str) -> bool:
        p = self.player.perks
        if perk_id == "juggernog":
            return p.juggernog
        if perk_id == "speed_cola":
            return p.speed_cola
        if perk_id == "double_tap":
            return p.double_tap
        if perk_id == "stamin_up":
            return p.stamin_up
        return False

    def _grant_perk(self, perk_id: str) -> None:
        p = self.player.perks
        if perk_id == "juggernog" and not p.juggernog:
            p.juggernog = True
            # Increase max health immediately
            old_max = float(self.player.entity.health.max)
            new_max = float(config.JUGGER_MAX_HEALTH)
            delta = new_max - old_max
            self.player.entity.health.max = new_max
            self.player.entity.health.current = min(new_max, self.player.entity.health.current + delta)
        elif perk_id == "speed_cola":
            p.speed_cola = True
        elif perk_id == "double_tap":
            p.double_tap = True
        elif perk_id == "stamin_up":
            p.stamin_up = True

    def _update_interaction_prompt(self) -> None:
        """Choose a single prompt string for the closest interactable."""
        self.interact_prompt = ""
        if not self.player.state.alive:
            return

        ppos = self.player.entity.transform.position
        best_d = 1e9
        best_text = ""

        # Barriers first (repair)
        for b in self.barriers:
            if not b.alive():
                continue
            if b.health >= b.max_health - 1e-3:
                continue
            d = (Vec3(b.position.x, ppos.y, b.position.z) - ppos).length()
            if d <= config.BARRIER_REPAIR_RADIUS and d < best_d:
                best_d = d
                best_text = f"E - Починить барьер (+{config.POINTS_REPAIR})"

        # Shop spots
        for s in self.interact_spots:
            d = (Vec3(s.position.x, ppos.y, s.position.z) - ppos).length()
            if d > s.radius or d >= best_d:
                continue

            if s.kind == "wall_weapon" and s.weapon_index is not None:
                if not self.player.unlocked_weapons[s.weapon_index]:
                    price = s.cost
                    afford = self.player.points >= price
                    suffix = "" if afford else " (не хватает)"
                    best_text = f"E - Купить {s.name} ({price}){suffix}"
                else:
                    price = int(config.COST_AMMO_REFILL)
                    afford = self.player.points >= price
                    suffix = "" if afford else " (не хватает)"
                    best_text = f"E - Патроны {s.name} ({price}){suffix}"

            elif s.kind == "ammo":
                price = int(config.COST_AMMO_REFILL)
                afford = self.player.points >= price
                suffix = "" if afford else " (не хватает)"
                best_text = f"E - Ящик патронов ({price}){suffix}"

            elif s.kind == "perk" and s.perk_id is not None:
                if self._has_perk(s.perk_id):
                    best_text = f"{s.name} - уже куплено"
                else:
                    price = s.cost
                    afford = self.player.points >= price
                    suffix = "" if afford else " (не хватает)"
                    best_text = f"E - {s.name} ({price}){suffix}"

            best_d = d

        self.interact_prompt = best_text

    def try_interact(self) -> None:
        """Perform the action of the closest interactable if any."""
        if not self.player.state.alive:
            return

        ppos = self.player.entity.transform.position

        # Repair barrier
        best_b: Optional[Barrier] = None
        best_d = 1e9
        for b in self.barriers:
            if not b.alive() or b.health >= b.max_health - 1e-3:
                continue
            d = (Vec3(b.position.x, ppos.y, b.position.z) - ppos).length()
            if d <= config.BARRIER_REPAIR_RADIUS and d < best_d:
                best_d = d
                best_b = b
        if best_b is not None:
            best_b.health = min(best_b.max_health, best_b.health + float(config.BARRIER_REPAIR_AMOUNT))
            self.player.give_points(int(config.POINTS_REPAIR))
            self.interact_prompt = f"Барьер починен (+{config.POINTS_REPAIR})"
            return

        # Shop spots
        best_s: Optional[InteractSpot] = None
        best_d = 1e9
        for s in self.interact_spots:
            d = (Vec3(s.position.x, ppos.y, s.position.z) - ppos).length()
            if d <= s.radius and d < best_d:
                best_d = d
                best_s = s

        if best_s is None:
            return

        s = best_s
        if s.kind == "wall_weapon" and s.weapon_index is not None:
            if not self.player.unlocked_weapons[s.weapon_index]:
                if self.player.spend_points(s.cost):
                    self.player.unlock_weapon(s.weapon_index)
                    self.player.weapon_index = s.weapon_index
                    self.interact_prompt = f"Куплено: {s.name}"
            else:
                price = int(config.COST_AMMO_REFILL)
                if self.player.spend_points(price):
                    self.player.refill_weapon_ammo(s.weapon_index)
                    self.interact_prompt = f"Патроны: {s.name}"
            return

        if s.kind == "ammo":
            price = int(config.COST_AMMO_REFILL)
            if self.player.spend_points(price):
                self.player.refill_current_ammo()
                self.interact_prompt = "Патроны пополнены"
            return

        if s.kind == "perk" and s.perk_id is not None:
            if self._has_perk(s.perk_id):
                return
            if self.player.spend_points(s.cost):
                self._grant_perk(s.perk_id)
                self.interact_prompt = f"Перк: {s.name}"

    def update_zombies(self, delta: float) -> None:
        player_pos = self.player.entity.transform.position
        living: List[Entity] = []
        for zombie in self.zombies:
            if zombie.health is None:
                continue
            if zombie.health.current <= 0:
                self.kills += 1
                self.player.give_points(config.POINTS_PER_KILL)
                continue

            transform = zombie.transform
            direction = (player_pos - transform.position)
            distance = direction.length()
            if distance > 0:
                # Rotate zombie to face the player (used by multi-part rendering)
                transform.rotation.y = math.atan2(direction.x, direction.z)
                move_dir = direction.normalized()
                speed = config.ZOMBIE_BASE_SPEED + self.wave * 0.1
                desired = transform.position + move_dir * speed * delta

                # Barriers: zombies bump and start breaking them
                b = self._barrier_at(desired)
                if b is not None:
                    b.health = max(0.0, b.health - config.BARRIER_ZOMBIE_DAMAGE * delta)
                else:
                    transform.position = desired

            ai = zombie.ai
            # If a zombie is currently on a barrier, don't let it "hit" the player through it
            hitting_barrier = self._barrier_at(transform.position, extra=0.35) is not None
            if ai and (not hitting_barrier) and distance <= config.ZOMBIE_ATTACK_RANGE:
                if ai.attack_cooldown <= 0:
                    self.player.damage(config.ZOMBIE_ATTACK_DAMAGE)
                    ai.attack_cooldown = config.ZOMBIE_ATTACK_COOLDOWN
            if ai and ai.attack_cooldown > 0:
                ai.attack_cooldown -= delta
            living.append(zombie)
        self.zombies = living

    def update_tracers(self, delta: float) -> None:
        if not self.tracers:
            return
        alive: List[Tracer] = []
        for t in self.tracers:
            t.age += delta
            if t.alive():
                alive.append(t)
        self.tracers = alive

    def _update_muzzle_fx(self, delta: float) -> None:
        if self.muzzle_flashes:
            mf_alive: List[MuzzleFlash] = []
            for f in self.muzzle_flashes:
                f.age += delta
                if f.alive():
                    mf_alive.append(f)
            self.muzzle_flashes = mf_alive

        if self.smoke_particles:
            sp_alive: List[SmokeParticle] = []
            for p in self.smoke_particles:
                p.age += delta
                if not p.alive():
                    continue
                # Basic buoyancy + drag
                p.velocity = Vec3(p.velocity.x * 0.98, p.velocity.y + 0.30 * delta, p.velocity.z * 0.98)
                p.position += p.velocity * delta
                sp_alive.append(p)
            self.smoke_particles = sp_alive

    def _camera_basis(self) -> tuple[Vec3, Vec3, Vec3]:
        forward = self.camera.forward()
        world_up = Vec3(0.0, 1.0, 0.0)
        right = world_up.cross(forward).normalized()
        up = forward.cross(right).normalized()
        return forward, right, up

    def _muzzle_position(self) -> Vec3:
        forward, right, up = self._camera_basis()
        # Slightly forward and to the right/down
        return self.camera.position + forward * 0.65 + right * 0.26 + up * (-0.20)

    def shoot(self) -> None:
        weapon = self.player.weapon
        direction = self.camera.forward()
        origin = self.camera.position
        targets = [(z.transform.position + Vec3(0.0, 0.5, 0.0), z.collider.radius) for z in self.zombies]

        result = weapon.fire(origin, direction, targets)

        # Apply damage (each hit is one "bullet". Shotgun pellets produce multiple hits.)
        dmg = float(weapon.damage) * float(self.player.damage_multiplier)
        hit_targets: set[int] = set()
        for hit in result.hits:
            if hit.target_index >= len(self.zombies):
                continue
            zombie = self.zombies[hit.target_index]
            if zombie.health is None:
                continue
            if zombie.health.current <= 0:
                continue
            zombie.health.current -= dmg
            hit_targets.add(hit.target_index)

        # Points + hitmarker
        if hit_targets:
            self.player.hitmarker_timer = 0.12
            self.player.give_points(len(hit_targets) * config.POINTS_PER_HIT)

        # Tracers (visual only)
        muzzle = self._muzzle_position()
        for ray in result.rays:
            # Ray distances are computed from the camera origin.
            # Use the real hit point for the tracer end, but start it at the muzzle.
            hit_point = origin + ray.direction * ray.distance
            self.tracers.append(Tracer(start=muzzle.copy(), end=hit_point))

        # Muzzle flash + smoke (visual only)
        self._spawn_muzzle_fx()

    def _spawn_muzzle_fx(self) -> None:
        muzzle = self._muzzle_position()
        self.muzzle_flashes.append(MuzzleFlash(position=muzzle.copy()))

        # A few smoke puffs that drift forward and rise
        forward, right, up = self._camera_basis()
        for i in range(6):
            jitter = (right * (random.uniform(-0.03, 0.03)) + up * (random.uniform(-0.03, 0.03)))
            vel = forward * random.uniform(0.7, 1.2) + up * random.uniform(0.5, 0.9) + jitter * 3.0
            self.smoke_particles.append(
                SmokeParticle(position=(muzzle + forward * 0.12 + jitter).copy(), velocity=vel, size=random.uniform(0.10, 0.16), lifetime=random.uniform(0.7, 1.1))
            )

    def _render_tracers(self) -> List[RenderItem]:
        if not self.tracers:
            return []

        items: List[RenderItem] = []
        for t in self.tracers:
            d = (t.end - t.start)
            length = d.length()
            if length <= 1e-4:
                continue
            dir_n = d / length

            mid = t.start + dir_n * (length * 0.5)
            # IMPORTANT: Don't build the tracer orientation from yaw/pitch.
            # Different matrix conventions can make rotation_x behave unexpectedly.
            # Build an orthonormal basis directly from the direction instead.
            world_up = Vec3(0.0, 1.0, 0.0)
            right = world_up.cross(dir_n)
            if right.length() < 1e-6:
                right = Vec3(1.0, 0.0, 0.0)
            right = right.normalized()
            up = dir_n.cross(right).normalized()

            fade = max(0.0, 1.0 - (t.age / max(t.lifetime, 1e-6)))
            em = (3.0 * fade, 2.4 * fade, 1.2 * fade)

            # Column-major basis matrix: columns are (right, up, forward, translation)
            r = right * config.TRACER_THICKNESS
            u = up * config.TRACER_THICKNESS
            f = dir_n * length
            model = Mat4([
                r.x, r.y, r.z, 0.0,
                u.x, u.y, u.z, 0.0,
                f.x, f.y, f.z, 0.0,
                mid.x, mid.y, mid.z, 1.0,
            ])

            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=model,
                    color=(1.0, 1.0, 1.0),
                    texture=None,
                    uv_scale=1.0,
                    spec_strength=0.0,
                    shininess=2.0,
                    emissive=em,
                    cast_shadows=False,
                    receive_shadows=False,
                    ssao_mask=0.0,
                )
            )

        return items

    def _render_viewmodel(self) -> List[RenderItem]:
        """Very simple weapon model made from cubes."""
        if not self.player.state.alive:
            return []

        forward, right, up = self._camera_basis()

        # Base placement
        base_pos = self.camera.position + forward * 0.55 + right * 0.28 + up * (-0.26)
        base = (
            Mat4.translation(base_pos)
            @ Mat4.rotation_y(self.camera.yaw)
            @ Mat4.rotation_x(self.camera.pitch)
            @ Mat4.rotation_x(-0.12)
            @ Mat4.rotation_y(0.10)
        )

        items: List[RenderItem] = []

        wname = self.player.weapon.name

        # A tiny "viewmodel" made of cubes. Dimensions vary per weapon so the new guns
        # actually look different.
        if wname == "Pistol":
            body = (0.36, 0.20, 0.60)
            barrel = (0.10, 0.10, 0.48)
            barrel_pos = (0.0, 0.02, 0.58)
            grip = (0.14, 0.22, 0.20)
            grip_pos = (0.12, -0.18, 0.05)
        elif wname == "SMG":
            body = (0.42, 0.18, 0.88)
            barrel = (0.08, 0.08, 0.72)
            barrel_pos = (0.0, 0.03, 0.78)
            grip = (0.14, 0.20, 0.18)
            grip_pos = (0.12, -0.18, 0.12)
        elif wname == "Rifle":
            body = (0.46, 0.18, 0.96)
            barrel = (0.07, 0.07, 0.92)
            barrel_pos = (0.0, 0.04, 0.92)
            grip = (0.14, 0.20, 0.18)
            grip_pos = (0.12, -0.18, 0.10)
        elif wname == "Sniper":
            body = (0.48, 0.16, 1.02)
            barrel = (0.06, 0.06, 1.05)
            barrel_pos = (0.0, 0.05, 1.00)
            grip = (0.14, 0.20, 0.18)
            grip_pos = (0.12, -0.18, 0.08)
        else:  # Shotgun
            body = (0.52, 0.20, 0.98)
            barrel = (0.09, 0.09, 0.92)
            barrel_pos = (0.0, 0.04, 0.92)
            grip = (0.16, 0.22, 0.20)
            grip_pos = (0.14, -0.18, 0.06)

        # Body
        body_model = base @ Mat4.translation(Vec3(0.0, -0.05, 0.0)) @ Mat4.scale(Vec3(*body))
        items.append(
            RenderItem(
                mesh=self.cube_mesh,
                model=body_model,
                color=(1.0, 1.0, 1.0),
                texture=self.tex_metal,
                uv_scale=1.0,
                spec_strength=1.25,
                shininess=96.0,
                emissive=(0.0, 0.0, 0.0),
                cast_shadows=False,
                receive_shadows=False,
                ssao_mask=0.0,
                depth_test=False,
                depth_write=False,
            )
        )

        # Barrel
        barrel_model = base @ Mat4.translation(Vec3(*barrel_pos)) @ Mat4.scale(Vec3(*barrel))
        items.append(
            RenderItem(
                mesh=self.cube_mesh,
                model=barrel_model,
                color=(1.0, 1.0, 1.0),
                texture=self.tex_metal,
                uv_scale=1.0,
                spec_strength=1.35,
                shininess=110.0,
                emissive=(0.0, 0.0, 0.0),
                cast_shadows=False,
                receive_shadows=False,
                ssao_mask=0.0,
                depth_test=False,
                depth_write=False,
            )
        )

        # Grip
        grip_model = base @ Mat4.translation(Vec3(*grip_pos)) @ Mat4.scale(Vec3(*grip))
        items.append(
            RenderItem(
                mesh=self.cube_mesh,
                model=grip_model,
                color=(0.25, 0.25, 0.28),
                texture=None,
                uv_scale=1.0,
                spec_strength=0.35,
                shininess=32.0,
                emissive=(0.0, 0.0, 0.0),
                cast_shadows=False,
                receive_shadows=False,
                ssao_mask=0.0,
                depth_test=False,
                depth_write=False,
            )
        )

        # Extra parts to visually differentiate weapons
        if wname in ("SMG", "Rifle", "Sniper"):
            # Magazine
            mag_model = base @ Mat4.translation(Vec3(0.10, -0.24, 0.05)) @ Mat4.scale(Vec3(0.12, 0.26, 0.22))
            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=mag_model,
                    color=(0.22, 0.22, 0.24),
                    texture=None,
                    uv_scale=1.0,
                    spec_strength=0.45,
                    shininess=54.0,
                    emissive=(0.0, 0.0, 0.0),
                    cast_shadows=False,
                    receive_shadows=False,
                    ssao_mask=0.0,
                    depth_test=False,
                    depth_write=False,
                )
            )

        if wname in ("Rifle", "Sniper", "Shotgun"):
            # Stock
            stock_model = base @ Mat4.translation(Vec3(0.0, -0.03, -0.58)) @ Mat4.scale(Vec3(0.28, 0.14, 0.32))
            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=stock_model,
                    color=(1.0, 1.0, 1.0),
                    texture=self.tex_metal,
                    uv_scale=1.0,
                    spec_strength=0.90,
                    shininess=70.0,
                    emissive=(0.0, 0.0, 0.0),
                    cast_shadows=False,
                    receive_shadows=False,
                    ssao_mask=0.0,
                    depth_test=False,
                    depth_write=False,
                )
            )

        if wname == "Sniper":
            # Scope
            scope_model = base @ Mat4.translation(Vec3(0.0, 0.14, 0.25)) @ Mat4.scale(Vec3(0.16, 0.10, 0.32))
            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=scope_model,
                    color=(0.20, 0.20, 0.22),
                    texture=None,
                    uv_scale=1.0,
                    spec_strength=0.65,
                    shininess=64.0,
                    emissive=(0.0, 0.0, 0.0),
                    cast_shadows=False,
                    receive_shadows=False,
                    ssao_mask=0.0,
                    depth_test=False,
                    depth_write=False,
                )
            )

        return items

    def _render_muzzle_fx(self) -> List[RenderItem]:
        if not self.muzzle_flashes and not self.smoke_particles:
            return []

        forward, right, up = self._camera_basis()
        items: List[RenderItem] = []

        for f in self.muzzle_flashes:
            t = min(1.0, f.age / max(f.lifetime, 1e-6))
            fade = max(0.0, 1.0 - t)
            size = 0.28 + 0.42 * t
            pos = f.position + forward * 0.02

            r = right * size
            u = up * size
            fw = forward * 0.001
            model = Mat4([
                r.x, r.y, r.z, 0.0,
                u.x, u.y, u.z, 0.0,
                fw.x, fw.y, fw.z, 0.0,
                pos.x, pos.y, pos.z, 1.0,
            ])

            # Additive flash (very bright, short)
            intensity = 18.0 * fade
            items.append(
                RenderItem(
                    mesh=self.quad_mesh,
                    model=model,
                    color=(1.0, 1.0, 1.0),
                    texture=None,
                    uv_scale=1.0,
                    spec_strength=0.0,
                    shininess=2.0,
                    emissive=(intensity, intensity * 0.75, intensity * 0.35),
                    cast_shadows=False,
                    receive_shadows=False,
                    depth_test=False,
                    depth_write=False,
                    blend_mode=2,
                    alpha=fade,
                    sprite=1,
                    ssao_mask=0.0,
                )
            )

        for s in self.smoke_particles:
            t = min(1.0, s.age / max(s.lifetime, 1e-6))
            fade = max(0.0, 1.0 - t)
            size = s.size * (1.0 + 1.8 * t)
            pos = s.position

            r = right * size
            u = up * size
            fw = forward * 0.001
            model = Mat4([
                r.x, r.y, r.z, 0.0,
                u.x, u.y, u.z, 0.0,
                fw.x, fw.y, fw.z, 0.0,
                pos.x, pos.y, pos.z, 1.0,
            ])

            items.append(
                RenderItem(
                    mesh=self.quad_mesh,
                    model=model,
                    color=(0.55, 0.56, 0.60),
                    texture=None,
                    uv_scale=1.0,
                    spec_strength=0.0,
                    shininess=2.0,
                    emissive=(0.0, 0.0, 0.0),
                    cast_shadows=False,
                    receive_shadows=False,
                    depth_test=True,
                    depth_write=False,
                    blend_mode=1,
                    alpha=0.55 * fade,
                    sprite=2,
                    ssao_mask=0.0,
                )
            )

        return items

    def render_items(self) -> List[RenderItem]:
        items: List[RenderItem] = []

        # Sun disk (a simple emissive cube placed in the sky in the sun direction)
        # It is rendered with depth off so it always stays behind the world.
        to_sun = (self.current_light_dir * -1.0).normalized()
        sun_pos = self.camera.position + to_sun * 55.0 + Vec3(0.0, 6.0, 0.0)
        sun_scale = 1.6 if self.day_factor > 0.2 else 0.9
        sun_model = Mat4.translation(sun_pos) @ Mat4.scale(Vec3(sun_scale, sun_scale, sun_scale * 0.2))
        items.append(
            RenderItem(
                mesh=self.cube_mesh,
                model=sun_model,
                color=(1.0, 0.98, 0.92),
                texture=None,
                uv_scale=1.0,
                spec_strength=0.0,
                shininess=2.0,
                emissive=(10.0, 9.0, 6.0) if self.day_factor > 0.2 else (1.0, 1.0, 1.5),
                cast_shadows=False,
                receive_shadows=False,
                ssao_mask=0.0,
                depth_test=False,
                depth_write=False,
            )
        )

        # Infinite ground (tiles around the player)
        tile = float(config.GROUND_TILE_SIZE)
        radius = int(config.GROUND_TILES_RADIUS)
        uv = float(config.GROUND_UV_SCALE)
        p = self.player.entity.transform.position
        cx = int(math.floor(p.x / tile))
        cz = int(math.floor(p.z / tile))

        for tz in range(cz - radius, cz + radius + 1):
            for tx in range(cx - radius, cx + radius + 1):
                wx = tx * tile
                wz = tz * tile
                h = self._tile_hash(tx, tz)
                # 1/6 tiles become dirt for variety
                is_dirt = (h % 6) == 0
                tex = self.tex_dirt if is_dirt else self.tex_grass
                spec = 0.10 if is_dirt else 0.18
                shiny = 6.0 if is_dirt else 10.0
                items.append(
                    RenderItem(
                        mesh=self.ground_tile_mesh,
                        model=Mat4.translation(Vec3(wx, 0.0, wz)),
                        color=(1.0, 1.0, 1.0),
                        texture=tex,
                        uv_scale=uv,
                        spec_strength=spec,
                        shininess=shiny,
                        cast_shadows=False,
                        receive_shadows=True,
                    )
                )

        # Walls (mix of concrete and brick)
        for i, wall in enumerate(self.walls):
            model = Mat4.translation(wall.position) @ Mat4.scale(wall.size)
            tex = self.tex_concrete if (i % 2 == 0) else self.tex_brick
            uv = 2.2 if (i % 2 == 0) else 2.6
            spec = 0.28 if (i % 2 == 0) else 0.20
            shiny = 34.0 if (i % 2 == 0) else 22.0
            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=model,
                    color=(1.0, 1.0, 1.0),
                    texture=tex,
                    uv_scale=uv,
                    spec_strength=spec,
                    shininess=shiny,
                )
            )

        # Repairable barriers
        for b in self.barriers:
            if not b.alive():
                continue
            ratio = max(0.15, min(1.0, b.health / max(b.max_health, 1e-6)))
            size = Vec3(b.size.x, b.size.y * ratio, b.size.z)
            pos = Vec3(b.position.x, (size.y * 0.5), b.position.z)
            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=Mat4.translation(pos) @ Mat4.scale(size),
                    color=(1.0, 1.0, 1.0),
                    texture=self.tex_brick,
                    uv_scale=2.0,
                    spec_strength=0.10,
                    shininess=12.0,
                )
            )

        # Shop + perk machines
        for s in self.interact_spots:
            if s.kind == "wall_weapon":
                pos = s.position + Vec3(0.0, 1.0, 0.0)
                model = Mat4.translation(pos) @ Mat4.scale(Vec3(0.65, 0.65, 0.12))
                items.append(
                    RenderItem(
                        mesh=self.cube_mesh,
                        model=model,
                        color=(1.0, 1.0, 1.0),
                        texture=self.tex_metal,
                        uv_scale=1.4,
                        spec_strength=0.55,
                        shininess=64.0,
                        emissive=(0.2, 0.25, 0.3),
                        cast_shadows=False,
                        receive_shadows=False,
                        ssao_mask=0.0,
                    )
                )
            elif s.kind == "ammo":
                pos = s.position + Vec3(0.0, 0.45, 0.0)
                model = Mat4.translation(pos) @ Mat4.scale(Vec3(0.9, 0.9, 0.9))
                items.append(
                    RenderItem(
                        mesh=self.cube_mesh,
                        model=model,
                        color=(1.0, 1.0, 1.0),
                        texture=self.tex_concrete,
                        uv_scale=1.2,
                        spec_strength=0.20,
                        shininess=18.0,
                        emissive=(0.15, 0.25, 0.5),
                        cast_shadows=True,
                        receive_shadows=True,
                    )
                )
            elif s.kind == "perk":
                pos = s.position + Vec3(0.0, 0.85, 0.0)
                model = Mat4.translation(pos) @ Mat4.scale(Vec3(0.8, 1.7, 0.8))
                # Color hint per perk
                col = (0.6, 0.2, 0.2)
                em = (0.8, 0.15, 0.15)
                if s.perk_id == "speed_cola":
                    col, em = (0.2, 0.45, 0.8), (0.15, 0.35, 0.9)
                elif s.perk_id == "double_tap":
                    col, em = (0.75, 0.65, 0.25), (0.75, 0.55, 0.18)
                elif s.perk_id == "stamin_up":
                    col, em = (0.25, 0.65, 0.25), (0.12, 0.55, 0.12)
                items.append(
                    RenderItem(
                        mesh=self.cube_mesh,
                        model=model,
                        color=col,
                        texture=None,
                        uv_scale=1.0,
                        spec_strength=0.05,
                        shininess=6.0,
                        emissive=em,
                        cast_shadows=True,
                        receive_shadows=True,
                    )
                )

        # Zombies
        for zombie in self.zombies:
            items.extend(self._render_zombie(zombie))

        # Tracers (emissive)
        items.extend(self._render_tracers())

        # Viewmodel weapon (render last)
        items.extend(self._render_viewmodel())

        # Muzzle flash + smoke should appear above the weapon
        items.extend(self._render_muzzle_fx())

        return items

    def _render_zombie(self, zombie: Entity) -> List[RenderItem]:
        """Low-poly zombie made out of multiple cubes (head/torso/arms/legs)."""
        t = zombie.transform
        pos = t.position

        # Face the player
        yaw = float(t.rotation.y)
        base = Mat4.translation(pos) @ Mat4.rotation_y(yaw)

        items: List[RenderItem] = []

        # Torso
        torso = base @ Mat4.translation(Vec3(0.0, 0.95, 0.0)) @ Mat4.scale(Vec3(0.55, 0.70, 0.30))
        items.append(
            RenderItem(
                mesh=self.cube_mesh,
                model=torso,
                color=(1.0, 1.0, 1.0),
                texture=self.tex_zombie,
                uv_scale=1.0,
                spec_strength=0.08,
                shininess=10.0,
            )
        )

        # Head
        head = base @ Mat4.translation(Vec3(0.0, 1.55, 0.0)) @ Mat4.scale(Vec3(0.34, 0.34, 0.34))
        items.append(
            RenderItem(
                mesh=self.cube_mesh,
                model=head,
                color=(1.0, 1.0, 1.0),
                texture=self.tex_zombie,
                uv_scale=1.0,
                spec_strength=0.06,
                shininess=8.0,
            )
        )

        # Arms
        arm_l = base @ Mat4.translation(Vec3(-0.45, 1.05, 0.0)) @ Mat4.rotation_z(0.10) @ Mat4.scale(Vec3(0.18, 0.60, 0.18))
        arm_r = base @ Mat4.translation(Vec3( 0.45, 1.05, 0.0)) @ Mat4.rotation_z(-0.10) @ Mat4.scale(Vec3(0.18, 0.60, 0.18))
        for arm in (arm_l, arm_r):
            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=arm,
                    color=(0.95, 1.0, 0.95),
                    texture=self.tex_zombie,
                    uv_scale=1.0,
                    spec_strength=0.05,
                    shininess=8.0,
                )
            )

        # Legs (pants use concrete for a different material look)
        leg_l = base @ Mat4.translation(Vec3(-0.18, 0.35, 0.0)) @ Mat4.scale(Vec3(0.22, 0.70, 0.22))
        leg_r = base @ Mat4.translation(Vec3( 0.18, 0.35, 0.0)) @ Mat4.scale(Vec3(0.22, 0.70, 0.22))
        for leg in (leg_l, leg_r):
            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=leg,
                    color=(0.85, 0.90, 0.95),
                    texture=self.tex_concrete,
                    uv_scale=1.2,
                    spec_strength=0.25,
                    shininess=28.0,
                )
            )

        # Eyes (tiny emissive cubes so they don't look like a "green box")
        eye_l = base @ Mat4.translation(Vec3(-0.10, 1.58, 0.18)) @ Mat4.scale(Vec3(0.06, 0.06, 0.04))
        eye_r = base @ Mat4.translation(Vec3( 0.10, 1.58, 0.18)) @ Mat4.scale(Vec3(0.06, 0.06, 0.04))
        for eye in (eye_l, eye_r):
            items.append(
                RenderItem(
                    mesh=self.cube_mesh,
                    model=eye,
                    color=(1.0, 1.0, 1.0),
                    texture=None,
                    uv_scale=1.0,
                    spec_strength=0.0,
                    shininess=2.0,
                    emissive=(0.6, 0.9, 0.5),
                    cast_shadows=False,
                    receive_shadows=False,
                )
            )

        return items

    def reset(self) -> None:
        self.player.reset()
        self.tracers = []
        self.wave = 0
        self.kills = 0
        self.start_wave()
