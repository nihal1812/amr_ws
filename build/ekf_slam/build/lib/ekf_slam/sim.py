import numpy as np
from models import wrap_angle

class MultiLandmarkSim:
    def __init__(
        self,
        landmarks: dict,          # {id: (x,y), ...}
        dt: float = 0.1,
        max_range: float = 8.0,
        sigma_r: float = 0.10,
        sigma_phi_deg: float = 2.0,
        seed: int = 0
    ):
        self.landmarks = {int(k): np.array(v, dtype=float) for k, v in landmarks.items()}
        self.dt = float(dt)
        self.max_range = float(max_range)
        self.sigma_r = float(sigma_r)
        self.sigma_phi = float(np.deg2rad(sigma_phi_deg))
        self.rng = np.random.default_rng(seed)

        self.gt_pose = np.array([0.0, 0.0, 0.0], dtype=float)

    def _gt_step(self, v: float, w: float):
        x, y, theta = self.gt_pose
        dt = self.dt

        if abs(w) < 1e-9:
            x = x + v * dt * np.cos(theta)
            y = y + v * dt * np.sin(theta)
            theta = theta
        else:
            theta2 = theta + w * dt
            x = x + (v / w) * (np.sin(theta2) - np.sin(theta))
            y = y + (v / w) * (-np.cos(theta2) + np.cos(theta))
            theta = theta2

        self.gt_pose = np.array([x, y, wrap_angle(theta)], dtype=float)

    def _measure_one(self, lm_xy: np.ndarray):
        x, y, theta = self.gt_pose
        lx, ly = lm_xy

        dx = lx - x
        dy = ly - y
        r_true = np.sqrt(dx*dx + dy*dy)

        if r_true > self.max_range:
            return None

        phi_true = wrap_angle(np.arctan2(dy, dx) - theta)

        r_meas = r_true + self.rng.normal(0.0, self.sigma_r)
        phi_meas = wrap_angle(phi_true + self.rng.normal(0.0, self.sigma_phi))

        return np.array([r_meas, phi_meas], dtype=float)

    def step(self, v: float, w: float):
        self._gt_step(v, w)
        u = (v, w, self.dt)

        obs = []
        for lm_id, lm_xy in self.landmarks.items():
            z = self._measure_one(lm_xy)
            if z is not None:
                obs.append((lm_id, z))
        return u, obs