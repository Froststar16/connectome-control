"""The body: NeuroMechFly v2 (flygym 2.x, MuJoCo) walking in an arena with a looming ball.

Everything flygym-specific lives in this file, so the rest of the control code only sees three
things: ``vision()`` (what the eyes see), ``step(command, n)`` (walk with this descending signal
for n physics steps) and a few readouts (thorax position, time).

The body
--------
* ``make_locomotion_fly`` from flygym's own examples: legs-only NeuroMechFly, position-controlled
  joints, leg adhesion, plus the two compound eyes (``add_vision``).
* ``HybridTurningController`` from the same examples: a central pattern generator (CPG, six
  coupled oscillators in a tripod gait) produces leg phases; preprogrammed steps turn phases into
  joint angles; two reflex-like rules (retract a leg that hangs in a hole, lift a leg that hits
  something) correct them. Its input is the 2-number descending signal (left, right) that the
  decoder produces. This mirrors the real division of labour: the brain sends a few descending
  commands, and the ventral nerve cord turns them into coordinated leg movements.

The arena
---------
A flat floor (grey checkerboard), a white sky, and a black ball that is moved kinematically
(a MuJoCo "mocap" body: it goes where we put it and has no collisions). A ``LoomingBall`` moves it
in a straight line at constant speed towards a target point, which makes the classic looming
stimulus used in fly escape experiments.

Timing
------
flygym's physics timestep is 0.1 ms, the same as the brain's dt, so one brain step = one physics
step and the two clocks never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoomingBall:
    """A ball that starts at ``start`` (mm, world frame) at time ``t_start`` (ms) and moves in a
    straight line towards ``target`` at ``speed`` mm/s, stopping ``stop_dist`` mm short of it."""
    start: np.ndarray
    target: np.ndarray
    speed: float = 150.0
    t_start: float = 100.0
    stop_dist: float = 3.0

    def position(self, t_ms: float) -> np.ndarray:
        start, target = np.asarray(self.start, float), np.asarray(self.target, float)
        path = target - start
        length = np.linalg.norm(path)
        travelled = np.clip(self.speed * max(t_ms - self.t_start, 0.0) / 1000.0, 0.0,
                            length - self.stop_dist)
        return start + path / length * travelled

    @classmethod
    def frontal(cls, head: np.ndarray, distance: float = 60.0, elevation_deg: float = 20.0,
                azimuth_deg: float = 0.0, **kw) -> LoomingBall:
        """Approach from ``distance`` mm away, at the given elevation and azimuth (0 = straight
        ahead, positive = to the fly's left), towards the head position."""
        e, a = np.radians(elevation_deg), np.radians(azimuth_deg)
        direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        return cls(start=np.asarray(head) + distance * direction, target=np.asarray(head), **kw)


class FlyArena:
    """NeuroMechFly + walking controller + looming ball, wrapped for the closed loop."""

    def __init__(self, ball_radius: float = 2.0, render: bool = False,
                 render_fps: int = 30, playback_speed: float = 0.1, seed: int = 0,
                 camera: str = "side"):
        import mujoco as mj
        from flygym import Simulation
        from flygym.anatomy import BodySegment, ContactBodiesPreset
        from flygym.compose import FlatGroundWorld
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import (
            HybridTurningController,
            PreprogrammedSteps,
            make_locomotion_fly,
        )

        self.seed = seed
        fly = make_locomotion_fly(name="fly", colorize=render)
        fly.add_vision()
        world = FlatGroundWorld()
        world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                      bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                      add_ground_contact_sensors=False)
        ball = world.mjcf_root.worldbody.add_body(name="looming_ball", mocap=True,
                                                  pos=[500.0, 0.0, 1.0])
        ball.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[ball_radius, 0, 0],
                      rgba=[0, 0, 0, 1], contype=0, conaffinity=0)
        cam = None
        if render and camera == "side":
            # A fixed side-view camera in the world, looking at the space in front of the fly.
            cam = world.mjcf_root.worldbody.add_camera(
                name="side_cam", pos=[5.0, -17.0, 5.5], fovy=45.0,
                xyaxes=[1, 0, 0, 0, 0.26, 0.97])
        elif render and camera == "top":
            # Looking straight down on the space the fly walks through (x 0-14 mm), x (forward)
            # to the right of the image and y (left) up: turning shows up as rotation (Phase 5).
            cam = world.mjcf_root.worldbody.add_camera(
                name="top_cam", pos=[7.0, 2.5, 28.0], fovy=45.0,
                xyaxes=[1, 0, 0, 0, 1, 0])
        elif render:
            raise ValueError(f"unknown camera {camera!r}; choose 'side' or 'top'")
        self.fly, self.world, self.ball_radius = fly, world, ball_radius
        self.sim = Simulation(world)
        self.dt_ms = self.sim.timestep * 1000.0
        if render:
            res = (360, 640) if camera == "side" else (480, 480)
            self.sim.set_renderer([cam], camera_res=res, playback_speed=playback_speed,
                                  output_fps=render_fps)
        self.render = render

        self.steps = PreprogrammedSteps()
        self.dof_order = fly.get_actuated_jointdofs_order("position")
        self.controller = HybridTurningController(timestep=self.sim.timestep,
                                                  preprogrammed_steps=self.steps,
                                                  output_dof_order=self.dof_order)
        order = fly.get_bodysegs_order()
        self._thorax = order.index(BodySegment("c_thorax"))
        self._head = order.index(BodySegment("c_head"))
        self.ball_path: LoomingBall | None = None
        self.reset()

    # -- lifecycle ------------------------------------------------------------------------------
    def reset(self, warmup_s: float = 0.05) -> None:
        """Neutral pose, settle onto the floor, ball parked far away, clock at 0."""
        from flygym_demo.complex_terrain import LocomotionAction, apply_locomotion_action

        self.sim.reset()
        self.controller.reset(seed=self.seed)
        apply_locomotion_action(self.sim, self.fly.name, LocomotionAction(
            joint_angles=self.steps.default_pose_by_dof_order(self.dof_order),
            adhesion_onoff=np.ones(6, dtype=bool)))
        self.sim.mj_data.mocap_pos[0] = [500.0, 0.0, 1.0]
        self.sim.warmup(warmup_s)
        self.t0 = self.sim.time          # our clock starts after the warm-up
        self.sim.forward()

    @property
    def t_ms(self) -> float:
        return (self.sim.time - self.t0) * 1000.0

    # -- sensing --------------------------------------------------------------------------------
    def vision(self) -> np.ndarray:
        """Ommatidia readout (2 eyes, 721 ommatidia, 2 photoreceptor types), brightness 0-1."""
        return self.sim.get_ommatidia_readouts(self.fly.name)

    def thorax(self) -> np.ndarray:
        return self.sim.get_body_positions(self.fly.name)[self._thorax].copy()

    def head(self) -> np.ndarray:
        return self.sim.get_body_positions(self.fly.name)[self._head].copy()

    def ball(self) -> np.ndarray:
        return self.sim.mj_data.mocap_pos[0].copy()

    def heading(self) -> float:
        """Yaw of the thorax in degrees (0 = facing +x, positive = turned left, i.e.
        counter-clockwise seen from above), from its orientation quaternion (w, x, y, z)."""
        w, x, y, z = self.sim.get_body_rotations(self.fly.name)[self._thorax]
        return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))

    def ommatidia_directions(self) -> np.ndarray:
        """Viewing direction of every ommatidium, (2 eyes, 721, 3) unit vectors in the HEAD frame
        (x forward, y left, z up).

        flygym renders each eye with a pinhole camera (157 deg field of view, 512 x 450 pixels),
        bends the image with a fisheye lookup table, and averages the pixels inside each
        ommatidium's hexagon. Following one pixel backwards: corrected pixel -> raw pixel (the
        lookup table) -> ray through the pinhole (camera intrinsics) -> world (camera
        orientation). An ommatidium looks along the average ray of its pixels. The head is rigid
        on this body, so the result is fixed for the whole run."""
        import mujoco as mj
        from flygym.vision.retina import Retina

        ret = Retina()
        h, w = ret.nrows, ret.ncols
        omm = ret.ommatidia_id_map.ravel().astype(np.int64) - 1     # -1 = between hexagons
        ok = (omm >= 0) & (ret._fisheye_lookup >= 0)
        row, col = np.divmod(ret._fisheye_lookup[ok], w)
        omm = omm[ok]
        head_rot = np.zeros(9)                           # the head is fused into the thorax
        mj.mju_quat2Mat(head_rot, self.sim.get_body_rotations(self.fly.name)[self._head])
        head_rot = head_rot.reshape(3, 3)
        out = np.zeros((2, ret.num_ommatidia_per_eye, 3))
        for e, cam_id in enumerate(self.sim._intern_eye_camera_ids_by_fly[self.fly.name]):
            f = (h / 2) / np.tan(np.radians(self.sim.mj_model.cam_fovy[cam_id]) / 2)
            # MuJoCo camera frame: looks along -z, image x to the right, image y up
            ray = np.stack([col + 0.5 - w / 2, h / 2 - row - 0.5, np.full(len(row), -f)], 1)
            ray /= np.linalg.norm(ray, axis=1, keepdims=True)
            acc = np.zeros((ret.num_ommatidia_per_eye, 3))
            np.add.at(acc, omm, ray)
            cam_rot = self.sim.mj_data.cam_xmat[cam_id].reshape(3, 3)
            world = acc @ cam_rot.T
            out[e] = world @ head_rot                    # world -> head frame (R^T v, row-wise)
        return out / np.linalg.norm(out, axis=-1, keepdims=True)

    # -- acting ---------------------------------------------------------------------------------
    def step(self, command: np.ndarray, n_steps: int) -> None:
        """Walk with descending signal ``command`` (left, right) for n physics steps (held
        constant: zero-order hold), moving the ball along its path on every step."""
        from flygym_demo.complex_terrain import (
            HybridControllerObservation,
            apply_locomotion_action,
        )

        command = np.asarray(command, dtype=float)
        for _ in range(n_steps):
            if self.ball_path is not None:
                self.sim.mj_data.mocap_pos[0] = self.ball_path.position(self.t_ms)
            obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
            apply_locomotion_action(self.sim, self.fly.name, self.controller.step(command, obs))
            self.sim.step()
            if self.render:
                self.sim.render_as_needed()

    def frames(self) -> list[np.ndarray]:
        """Rendered video frames so far (empty unless render=True)."""
        if not self.render or self.sim.renderer is None:
            return []
        return next(iter(self.sim.renderer.frames.values()))

    def close(self) -> None:
        """Free the OpenGL renderers (call before the interpreter exits; safe to repeat)."""
        self.sim.close()
