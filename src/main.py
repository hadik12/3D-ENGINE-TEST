from __future__ import annotations

import pygame

from engine.app import App
from engine.math3d import Mat4
from game.ui import HUD, HUDInfo
from game.world import World


def main() -> None:
    app = App(1280, 720, "Zombie Survival 3D")
    world = World()
    hud = HUD()

    running = True
    while running:
        running = app.poll()
        app.time.tick()
        delta = app.time.delta

        # Environment toggles (N = day/night, F = fog)
        world.handle_environment_input(app.input)

        # Graphics toggles
        # F1 = SSAO, F2 = Bloom, F3 = FXAA, F4 = Godrays
        if app.input.key_pressed(pygame.K_F1):
            app.renderer.enable_ssao = not app.renderer.enable_ssao
        if app.input.key_pressed(pygame.K_F2):
            app.renderer.enable_bloom = not app.renderer.enable_bloom
        if app.input.key_pressed(pygame.K_F3):
            app.renderer.enable_fxaa = not app.renderer.enable_fxaa
        if app.input.key_pressed(pygame.K_F4):
            app.renderer.enable_godrays = not app.renderer.enable_godrays

        if world.player.state.alive:
            world.update(delta, app.input)
            if app.input.mouse(0):
                world.shoot()
        else:
            if app.input.key_pressed(pygame.K_RETURN):
                world.reset()

        # Apply environment to renderer and compute light space for shadow mapping
        clear_color, light_space = world.apply_environment(app.renderer, delta)

        projection = Mat4.perspective(world.camera.fov, app.width / app.height, 0.1, 120.0)
        view = world.camera.view_matrix()

        # Build render list once (used by both passes)
        items = world.render_items()

        # Shadow pass (real shadow map)
        app.renderer.render_shadow_map(items, light_space, app.width, app.height)

        # Main pass
        app.renderer.begin(
            clear_color=clear_color,
            view=view,
            projection=projection,
            view_pos=world.camera.position.as_list(),
            light_space=light_space,
            screen_w=app.width,
            screen_h=app.height,
        )
        for item in items:
            app.renderer.draw_item(item)

        # Post-processing (SSAO + Bloom + FXAA). This draws the 3D scene to the backbuffer.
        # HUD is rendered after this so it stays crisp.
        app.renderer.end()

        health = int(world.player.entity.health.current) if world.player.entity.health else 0
        weapon = world.player.weapon
        info = HUDInfo(
            health=health,
            ammo=weapon.ammo,
            ammo_max=weapon.ammo_max,
            weapon=weapon.name,
            wave=world.wave,
            kills=world.kills,
            points=world.player.points,
            fps=app.time.fps,
            alive=world.player.state.alive,
            prompt=world.interact_prompt,
            perks=world.player_perks_string(),
            intermission_left=world.intermission_left,
            hitmarker=min(1.0, world.player.hitmarker_timer / 0.12) if world.player.hitmarker_timer > 0 else 0.0,
            blood=min(1.0, world.player.blood_timer / 0.65) if world.player.blood_timer > 0 else 0.0,
        )
        hud.render(info)

        app.swap()

    app.shutdown()


if __name__ == "__main__":
    main()
