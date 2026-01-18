# Zombie Survival 3D (Python + PyOpenGL + pygame)

Minimal 3D zombie survival game using a tiny custom engine built on top of PyOpenGL for rendering and pygame for window/input/timing.

This version includes:
- Textures (grass, concrete, zombie)
- Fog
- Day/Night cycle (toggle)
- Real shadow mapping (directional light)
- Soft shadows (Poisson PCF)
- Simple viewmodel weapon + crosshair
- Bullet tracers

## Setup

```bash
python -m venv .venv
## Windows (CMD): .venv\Scripts\activate.bat
## Windows (PowerShell): .venv\Scripts\Activate.ps1
## macOS/Linux: source .venv/bin/activate

python -m pip install -r requirements.txt
python src/main.py
```

## Controls

- **WASD**: move
- **Mouse**: look
- **LMB**: shoot
- **R**: reload
- **1 / 2**: switch weapons (pistol / shotgun)
- **Enter**: restart after death

Environment:
- **N**: toggle day/night
- **F**: toggle fog

HUD:
- A small **crosshair** is shown while alive.

## Architecture overview

```
src/
  engine/   # minimal rendering + input + math + ECS + physics
  game/     # gameplay systems: player, zombies, waves, weapons, UI
assets/
  textures/ # simple textures (png)
```

## Notes

- If you see a warning about `PyOpenGL_accelerate`, you can ignore it. This project only needs `PyOpenGL`.
- Shadow quality is controlled by `shadow_size` in `src/engine/renderer.py`.
