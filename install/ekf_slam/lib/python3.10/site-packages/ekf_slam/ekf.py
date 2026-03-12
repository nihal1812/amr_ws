import numpy as np
from .models import h, wrap_angle, analytic_jacobian_pose, analytic_landmark_jacobian

def landmark_cols(i: int) -> int:
    return 3 + 2*i

def build_full_H(pose: np.ndarray, landmark: np.ndarray, i: int, N: int) -> np.ndarray:
    H = np.zeros((2, 3 + 2*N), dtype=float)
    Hr = analytic_jacobian_pose(pose, landmark)        # (2,3)
    Hl = analytic_landmark_jacobian(pose, landmark)    # (2,2)
    H[:, 0:3] = Hr
    j = landmark_cols(i)
    H[:, j:j+2] = Hl
    return H

def motion_model(pose: np.ndarray, u, eps_w: float = 1e-9) -> np.ndarray:
    x, y, theta = pose
    v, w, dt = u

    if abs(w) < eps_w:
        # straight-line approximation
        x_new = x + v * dt * np.cos(theta)
        y_new = y + v * dt * np.sin(theta)
        theta_new = theta
    else:
        theta2 = theta + w * dt
        x_new = x + (v / w) * (np.sin(theta2) - np.sin(theta))
        y_new = y + (v / w) * (-np.cos(theta2) + np.cos(theta))
        theta_new = theta2

    return np.array([x_new, y_new, wrap_angle(theta_new)], dtype=float)

def init_landmark_jacobians(pose: np.ndarray, z: np.ndarray):
    """
    For landmark initialization:
    l = [x + r cos(theta+phi), y + r sin(theta+phi)]
    Returns:
      lx, ly, Gx(2x3), Gz(2x2)
    """
    x, y, theta = pose
    r, phi = float(z[0]), float(z[1])
    alpha = theta + phi

    c = np.cos(alpha)
    s = np.sin(alpha)

    lx = x + r * c
    ly = y + r * s

    Gx = np.array([
        [1.0, 0.0, -r * s],
        [0.0, 1.0,  r * c],
    ], dtype=float)

    Gz = np.array([
        [c, -r * s],
        [s,  r * c],
    ], dtype=float)

    return lx, ly, Gx, Gz

def jacobian_F(pose: np.ndarray, u, eps_w: float = 1e-9) -> np.ndarray:
    x, y, theta = pose
    v, w, dt = u

    Fr = np.eye(3, dtype=float)

    if abs(w) < eps_w:
        Fr[0, 2] = -v * dt * np.sin(theta)
        Fr[1, 2] =  v * dt * np.cos(theta)
    else:
        theta2 = theta + w * dt
        Fr[0, 2] = (v / w) * (np.cos(theta2) - np.cos(theta))
        Fr[1, 2] = (v / w) * (np.sin(theta2) - np.sin(theta))

    return Fr


def jacobian_G(pose: np.ndarray, u, eps_w: float = 1e-9) -> np.ndarray:
    x, y, theta = pose
    v, w, dt = u

    Gr = np.zeros((3, 2), dtype=float)

    if abs(w) < eps_w:
        # d/dv
        Gr[0, 0] = dt * np.cos(theta)
        Gr[1, 0] = dt * np.sin(theta)
        Gr[2, 0] = 0.0
        # d/dw
        Gr[0, 1] = 0.0
        Gr[1, 1] = 0.0
        Gr[2, 1] = dt
    else:
        theta2 = theta + w * dt

        a = np.sin(theta2) - np.sin(theta)
        b = -np.cos(theta2) + np.cos(theta)

        # d/dv
        Gr[0, 0] = (1.0 / w) * a
        Gr[1, 0] = (1.0 / w) * b
        Gr[2, 0] = 0.0

        # d/dw
        Gr[0, 1] = -(v / (w*w)) * a + (v / w) * np.cos(theta2) * dt
        Gr[1, 1] = -(v / (w*w)) * b + (v / w) * np.sin(theta2) * dt
        Gr[2, 1] = dt

    return Gr

class EKFSLAM:
    def __init__(self, init_pose: np.ndarray, init_cov: np.ndarray):
        self.mu = init_pose.astype(float).copy()   # (3,)
        self.P  = init_cov.astype(float).copy()    # (3,3)
        self.id_to_index = {}
        self.N = 0

    @staticmethod
    def lm_start(i: int) -> int:
        return 3 + 2*i

    def state_size(self) -> int:
        return 3 + 2*self.N

    def pose(self) -> np.ndarray:
        return self.mu[0:3]

    def get_landmark(self, i: int) -> np.ndarray:
        j = self.lm_start(i)
        return self.mu[j:j+2]

    def set_landmark(self, i: int, value: np.ndarray) -> None:
        j = self.lm_start(i)
        self.mu[j:j+2] = value

    def initialize_landmark(self, landmark_id: int, z: np.ndarray, R: np.ndarray) -> int:
        if landmark_id in self.id_to_index:
            return self.id_to_index[landmark_id]

        pose = self.pose()
        lx, ly, Gx, Gz = init_landmark_jacobians(pose, z)

        # 1) Append landmark to mean
        self.mu = np.concatenate([self.mu, np.array([lx, ly], dtype=float)])

        # 2) Expand covariance with correct blocks
        M_old = self.P.shape[0]
        P_new = np.zeros((M_old + 2, M_old + 2), dtype=float)
        P_new[:M_old, :M_old] = self.P

        # Robot covariance
        P_rr = self.P[0:3, 0:3]  # (3,3)
        # Robot-to-all covariance
        P_rX = self.P[0:3, :M_old]  # (3,M_old)

        # New landmark covariance
        P_ll = Gx @ P_rr @ Gx.T + Gz @ R @ Gz.T  # (2,2)

        # Cross-correlation landmark <-> existing state
        P_lX = Gx @ P_rX  # (2,M_old)

        # Fill blocks
        P_new[M_old:, M_old:] = P_ll
        P_new[M_old:, :M_old] = P_lX
        P_new[:M_old, M_old:] = P_lX.T

        self.P = P_new

        # 3) Bookkeeping
        i = self.N
        self.id_to_index[landmark_id] = i
        self.N += 1
        return i

    def update_known_landmark(self, landmark_id: int, z: np.ndarray, R: np.ndarray) -> None:
        if landmark_id not in self.id_to_index:
            raise ValueError("Landmark not initialized. Call initialize_landmark first.")

        i = self.id_to_index[landmark_id]
        pose = self.pose()
        lm = self.get_landmark(i)

        z_hat = h(pose, lm)

        y = z.astype(float) - z_hat
        y[1] = wrap_angle(y[1])

        H = build_full_H(pose, lm, i, self.N)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ y
        self.mu[2] = wrap_angle(self.mu[2])

        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

    def predict(self, u, Q):
        v, w, dt = u
        x, y, thetha = self.pose()

        new_pose = motion_model(np.array([x, y, thetha]), u)
        self.mu[0:3] = new_pose
        self.mu[2] = wrap_angle(self.mu[2])

        M = self.P.shape[0]

        Fr = jacobian_F(np.array([x, y, thetha]), u)
        Gr = jacobian_G(np.array([x, y, thetha]), u)

        F = np.eye(M)
        F[0:3, 0:3] = Fr

        G = np.zeros((M, 2))
        G[0:3, :] = Gr

        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def step(self, u, Q, observations):
        self.predict(u, Q)
        for landmark_id, z, R in observations:
            if landmark_id not in self.id_to_index:
                self.initialize_landmark(landmark_id, z, R)
                self.update_known_landmark(landmark_id, z, R)  # still good
            else:
                self.update_known_landmark(landmark_id, z, R)
