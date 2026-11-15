"""Register the hands-on task scene, ``handson_pick_place``, with the MuJoCo plugin.

The scene file is ``scenes/handson_pick_place.xml``: blue, red and yellow 2 cm cubes and a
square PLACE AREA on a work mat, as at the Intel AI Summit Seoul 2026 hands-on session.
Registering it (at import, in every process that imports this module) makes it a normal
plugin scene:

- ``--scene handson_pick_place`` loads it, and the viewer lists it;
- every reset (start, **Reset Scene**, ``POST /reset``) places the three cubes at random
  positions on the mat in front of the arm, clear of each other, of the PLACE AREA and of
  the arm (:func:`reset_cubes`).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from physicalai_mujoco_plugin import scene_registry
from physicalai_mujoco_plugin.spawn import (
    ARM_CLEARANCE,
    ARM_FOOTPRINT_HEIGHT,
    arm_footprint,
    object_radius,
    place_freejoint,
    sample_object_positions,
)

SCENE_ID = "handson_pick_place"
SCENE_XML = Path(__file__).resolve().parent / "scenes" / "handson_pick_place.xml"
CUBES = ("cube_red", "cube_blue", "cube_yellow")
CUBE_JOINTS = tuple(f"{cube}:joint" for cube in CUBES)
CUBE_HALF = 0.01
"""Half the cube's edge, in metres."""
PLACE_AREA = "place_area"
PLACE_AREA_XY = (0.21, -0.13)
"""World position of the PLACE AREA's centre (mat X 230, Y 150 mm); must match the XML and make_mat_texture.py."""
PLACE_AREA_HALF = 0.0265
"""Half the PLACE AREA's side, in metres."""
MAT_XY = ((0.075, 0.275), (-0.215, 0.085))
"""Where cube centres may go: the mat (world x 0.06-0.29, y -0.23 to 0.10) less 15 mm at the edges."""

# Cubes spawn on an arc in front of the arm, on the mat: x 0.12-0.27 m, y -0.06 to +0.08 m.
SPAWN_CENTER = (0.0, -0.01)
SPAWN_MIN_R, SPAWN_MAX_R = 0.13, 0.27
SPAWN_ANGLE_HALF_DEG = 20.0
CUBE_MIN_SEP = 0.08
"""Cube centres at least 8 cm apart, so the open jaws never touch a neighbour."""
PLACE_MIN_SEP = 0.10
"""Cube centres at least 10 cm from the PLACE AREA's centre, so no cube starts in it."""
SPAWN_TRIES = 50


def reset_cubes(model: object, data: object, rng: np.random.Generator, robot_roots: tuple[int, ...] = ()) -> None:
    """Put the three cubes at random, well-separated positions with random yaws.

    The plugin's own sampler keeps the last draw when a crowded layout has no clear position
    left; here a whole layout is drawn again until every cube is on the mat and clear of the
    others, of the PLACE AREA and of the arm.
    """
    import mujoco  # noqa: PLC0415

    footprint = arm_footprint(model, data, robot_roots, ARM_FOOTPRINT_HEIGHT)
    padding = object_radius(model, CUBE_JOINTS) + ARM_CLEARANCE if footprint is not None else 0.0
    positions: list[tuple[float, float]] = []
    for _ in range(SPAWN_TRIES):
        positions = sample_object_positions(
            len(CUBES),
            rng=rng,
            center=SPAWN_CENTER,
            min_r=SPAWN_MIN_R,
            max_r=SPAWN_MAX_R,
            angle_half_deg=SPAWN_ANGLE_HALF_DEG,
            target_xy=PLACE_AREA_XY,
            target_min_sep=PLACE_MIN_SEP,
            object_min_sep=CUBE_MIN_SEP,
            footprint=footprint,
            padding=padding,
        )
        p = np.asarray(positions)
        apart = all(np.hypot(*(p[a] - p[b])) >= CUBE_MIN_SEP for a in range(len(p)) for b in range(a))
        clear = all(np.hypot(*(q - PLACE_AREA_XY)) >= PLACE_MIN_SEP for q in p)
        off_arm = footprint is None or all(footprint.distance(tuple(q)) >= padding for q in p)
        (x0, x1), (y0, y1) = MAT_XY
        on_mat = bool(np.all((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)))
        if apart and clear and off_arm and on_mat:
            break
    for joint, xy in zip(CUBE_JOINTS, positions, strict=True):
        place_freejoint(model, data, joint, xy, rng, z=CUBE_HALF + 0.002)
    mujoco.mj_forward(model, data)


def _build_reset(_scene, _profile, _arms, robot_roots):  # noqa: ANN001, ANN202
    return lambda model, data, rng: reset_cubes(model, data, rng, robot_roots)


def register() -> None:
    """Add the scene to the plugin's registry; calling it again changes nothing."""
    if SCENE_ID in scene_registry._SCENES:  # noqa: SLF001
        return
    # Start from single_pick_place for its SO-101 settings (front view, viewer camera, home pose).
    base = scene_registry.get_scene("single_pick_place")
    scene_registry._SCENES[SCENE_ID] = replace(  # noqa: SLF001
        base,
        scene_id=SCENE_ID,
        display_name="Hands-on Pick & Place",
        description="Blue, red and yellow cubes and a square PLACE AREA (AI Summit Seoul 2026 hands-on)",
        # An absolute path: the plugin joins it to its own scene folder, which keeps it as it is.
        scene_xml_relpath=str(SCENE_XML),
        free_joints=CUBE_JOINTS,
        target_bodies=(PLACE_AREA,),
        spawn_center=SPAWN_CENTER,
        spawn_min_r=SPAWN_MIN_R,
        spawn_max_r=SPAWN_MAX_R,
        spawn_angle_half_deg=SPAWN_ANGLE_HALF_DEG,
        block_min_sep=CUBE_MIN_SEP,
        target_min_sep=PLACE_MIN_SEP,
        robots=("so101",),
        arm_counts=(1,),
        layout="fixed",
        profile_overview_rigs=(),
        profile_front_overrides=(),
    )
    scene_registry._RESET_FUNCTIONS[SCENE_ID] = _build_reset  # noqa: SLF001


def fix_viewer_box_textures() -> None:
    """Make the browser viewer draw box textures the way MuJoCo's cameras do.

    The plugin's viewer patch (``viewer_textures.box_uvs``) lays a box's top-face texture out
    mirrored along y compared with MuJoCo's own renderer, which the robot cameras and so the
    policy use. A tiled texture hides that; the mat's labels show it (mirrored text). Flip v on
    the faces facing +z/-z so the viewer matches the cameras. Only the viewer changes.
    """
    try:
        from physicalai_mujoco_plugin import viewer_textures  # noqa: PLC0415
    except ImportError:
        return
    plugin_box_uvs = viewer_textures.box_uvs
    if getattr(plugin_box_uvs, "_handson_fixed", False):
        return

    def box_uvs(vertices, normals, half_size, repeat, *, uniform):  # noqa: ANN001, ANN202
        uv = plugin_box_uvs(vertices, normals, half_size, repeat, uniform=uniform)
        top = np.argmax(np.abs(normals), axis=1) == 2  # noqa: PLR2004
        uv[top, 1] = repeat[1] - uv[top, 1]
        return uv

    box_uvs._handson_fixed = True  # type: ignore[attr-defined]  # noqa: SLF001
    viewer_textures.box_uvs = box_uvs


register()
fix_viewer_box_textures()
