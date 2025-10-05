import os
import math
import pybullet as p
import pybullet_data
import numpy as np
from dataclasses import dataclass

@dataclass
class PickPlaceConfig:
    gui: bool = False
    max_steps: int = 120
    workspace_bounds = (-0.25, 0.25, -0.25, 0.25, 0.0, 0.35)  # x_min,x_max,y_min,y_max,z_min,z_max
    target_radius: float = 0.03
    action_scale: float = 0.04
    yaw_scale: float = math.radians(15)
    gripper_open_height: float = 0.18
    object_size: float = 0.025
    seed: int = 0
    render_width: int = 128
    render_height: int = 128


class SimplePickPlaceEnv:
    """
    A minimal PyBullet pick-and-place environment:
      - One block, one circular target marker.
      - Action: (dx, dy, dz, dyaw, gripper_open_flag[0 or 1])
      - Observation: RGB image + state vector (ee_pose[4] (x,y,z,yaw), gripper_open, object_xyz, goal_xy)
      - Success: object center within target_radius of goal XY & lifted above small threshold.
    """

    def __init__(self, cfg: PickPlaceConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._connect()
        self._load_world()
        self.step_count = 0
        self._cached_camera = None

    def _connect(self):
        if self.cfg.gui:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

    def close(self):
        p.disconnect(self.cid)

    def _load_world(self):
        self.plane_id = p.loadURDF("plane.urdf")
        # Simple gripper = use a sphere as a point-effector placeholder (we fake gripper logic)
        self.gripper_id = p.loadURDF("sphere2.urdf",
                                     basePosition=[0, 0, self.cfg.gripper_open_height],
                                     globalScaling=0.02)
        # Cube
        ox = self.rng.uniform(-0.15, 0.15)
        oy = self.rng.uniform(-0.15, 0.15)
        self.object_id = p.loadURDF("cube_small.urdf",
                                    basePosition=[ox, oy, self.cfg.object_size / 2],
                                    globalScaling=self.cfg.object_size / 0.05)
        # Goal (visual marker: use another small object or a visual shape)
        gx = self.rng.uniform(-0.15, 0.15)
        gy = self.rng.uniform(-0.15, 0.15)
        self.goal_xy = np.array([gx, gy], dtype=np.float32)

        vsid = p.createVisualShape(p.GEOM_CYLINDER,
                                   radius=self.cfg.target_radius,
                                   length=0.001,
                                   rgbaColor=[0, 1, 0, 0.5])
        self.goal_id = p.createMultiBody(baseVisualShapeIndex=vsid,
                                         basePosition=[gx, gy, 0.0005])

        self.gripper_open = 1.0  # 1=open, 0=closed (for later real gripper logic)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._load_world()
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        rgb = self._render()
        ee_pos, ee_yaw = self._get_gripper_pose()
        obj_pos = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
        state = np.concatenate([
            ee_pos.astype(np.float32),
            np.array([ee_yaw], dtype=np.float32),
            np.array([self.gripper_open], dtype=np.float32),
            obj_pos.astype(np.float32),
            self.goal_xy.astype(np.float32)
        ])
        return {"rgb": rgb, "state": state}

    def _get_gripper_pose(self):
        pos = np.array(p.getBasePositionAndOrientation(self.gripper_id)[0])
        # We do not rotate the sphere (yaw placeholder)
        yaw = 0.0
        return pos, yaw

    def step(self, action):
        # action = [dx, dy, dz, dyaw, gripper_open_flag]
        dx, dy, dz, dyaw, g_flag = action
        pos, yaw = self._get_gripper_pose()
        new_pos = pos + np.array([
            dx * self.cfg.action_scale,
            dy * self.cfg.action_scale,
            dz * self.cfg.action_scale
        ])
        bounds = self.cfg.workspace_bounds
        new_pos[0] = np.clip(new_pos[0], bounds[0], bounds[1])
        new_pos[1] = np.clip(new_pos[1], bounds[2], bounds[3])
        new_pos[2] = np.clip(new_pos[2], bounds[4]+0.01, bounds[5])

        self.gripper_open = 1.0 if g_flag > 0.5 else 0.0

        p.resetBasePositionAndOrientation(self.gripper_id,
                                          new_pos.tolist(),
                                          p.getQuaternionFromEuler([0, 0, yaw + dyaw * self.cfg.yaw_scale]))
        # Fake grasp: if gripper "closed" and near cube, move cube with gripper (simple heuristic)
        obj_pos, obj_ori = p.getBasePositionAndOrientation(self.object_id)
        if self.gripper_open < 0.5:
            dist = np.linalg.norm(new_pos - np.array(obj_pos))
            if dist < 0.04:
                # Attach
                new_obj_pos = new_pos.copy()
                new_obj_pos[2] = max(new_obj_pos[2] - 0.015, self.cfg.object_size / 2)
                p.resetBasePositionAndOrientation(self.object_id, new_obj_pos, obj_ori)

        p.stepSimulation()
        self.step_count += 1
        obs = self._get_obs()
        done, success = self._check_done()
        return obs, float(success), done, {"success": success}

    def _check_done(self):
        obj_pos = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
        xy_dist = np.linalg.norm(obj_pos[:2] - self.goal_xy)
        lifted = obj_pos[2] > (self.cfg.object_size * 0.75)
        success = (xy_dist < self.cfg.target_radius) and lifted
        timeout = self.step_count >= self.cfg.max_steps
        return (success or timeout), success

    def _render(self):
        if self._cached_camera is None:
            cam_target = [0, 0, 0]
            cam_distance = 0.6
            yaw, pitch, roll = 45, -60, 0
            width, height = self.cfg.render_width, self.cfg.render_height
            view = p.computeViewMatrixFromYawPitchRoll(cam_target, cam_distance, yaw, pitch, roll, 2)
            proj = p.computeProjectionMatrixFOV(fov=60,
                                                aspect=float(width)/height,
                                                nearVal=0.01,
                                                farVal=2.0)
            self._cached_camera = (view, proj)
        view, proj = self._cached_camera
        w, h = self.cfg.render_width, self.cfg.render_height
        img = p.getCameraImage(w, h, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.reshape(img[2], (h, w, 4))[:, :, :3]
        return rgb

if __name__ == "__main__":
    cfg = PickPlaceConfig(gui=True)
    env = SimplePickPlaceEnv(cfg)
    obs = env.reset()
    for _ in range(50):
        action = np.array([0, 0, -0.25, 0, 1])  # simple downward open gripper
        obs, r, d, info = env.step(action)
        if d:
            break
    env.close()