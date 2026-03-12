import numpy as np

def wrap_angle(a:float)->float:
    return (a+np.pi)%(2*np.pi) - np.pi

def h(pose: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    x, y, thetha = pose
    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    r = np.sqrt(dx*dx + dy*dy)
    phi = wrap_angle(np.arctan2(dy, dx) - thetha)

    return np.array([r, phi], dtype = float)

def numeric_jacobian_pose(pose: np.ndarray, landmark: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    J = np.zeros((2, 3), dtype = float)

    for k in range(3):
        dp = np.zeros(3, dtype = float)
        dp[k] = eps

        z_plus = h(pose + dp, landmark)
        z_minus = h(pose - dp, landmark)

        dr = (z_plus[0]-z_minus[0]) / (2*eps)

        dphi = wrap_angle(z_plus[1] - z_minus[1]) / (2*eps)

        J[:, k] = np.array([dr, dphi])
    return J

def analytic_jacobian_pose(pose: np.ndarray, landmark: np.ndarray, eps_denom: float = 1e-12) -> np.ndarray:
    x, y, theta = pose
    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    q = dx*dx + dy*dy
    r = np.sqrt(q)

    r_safe = max(r, eps_denom)
    q_safe = max(q, eps_denom)

    J = np.zeros((2, 3), dtype=float)

    # dr/d[x,y,theta]
    J[0, 0] = -dx / r_safe
    J[0, 1] = -dy / r_safe
    J[0, 2] = 0.0

    # dphi/d[x,y,theta]
    J[1, 0] =  dy / q_safe
    J[1, 1] = -dx / q_safe
    J[1, 2] = -1.0

    return J

def numeric_landmark_jacobian(pose: np.ndarray, landmark: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    J = np.zeros((2, 2), dtype = float)

    for k in range(2):
        dp = np.zeros(2, dtype = float)
        dp[k] = eps

        z_plus = h(pose, landmark + dp)
        z_minus = h(pose, landmark - dp)

        dr = (z_plus[0]-z_minus[0]) / (2*eps)

        dphi = wrap_angle(z_plus[1] - z_minus[1]) / (2*eps)

        J[:, k] = np.array([dr, dphi])
    return J

def analytic_landmark_jacobian(pose: np.ndarray, landmark: np.ndarray, eps_denom: float = 1e-12) -> np.ndarray:
    x, y, theta = pose
    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    q = dx*dx + dy*dy
    r = np.sqrt(q)

    r_safe = max(r, eps_denom)
    q_safe = max(q, eps_denom)

    J = np.zeros((2, 2), dtype=float)

    # dr/d[x,y,theta]
    J[0, 0] = dx / r_safe
    J[0, 1] = dy / r_safe

    # dphi/d[x,y,theta]
    J[1, 0] = -dy / q_safe
    J[1, 1] = dx / q_safe

    return J

if __name__ == '__main__':
    pose = np.array([1.0, 2.0, 0.3])
    landmark = np.array([4.0, -1.0])

    J_num = numeric_jacobian_pose(pose, landmark, eps=1e-5)
    J_ana = analytic_jacobian_pose(pose, landmark, eps_denom=1e-12)
    J_num_lm = numeric_landmark_jacobian(pose, landmark, eps=1e-5)
    J_ana_lm = analytic_landmark_jacobian(pose, landmark, eps_denom=1e-12)

    print("case 1")
    print(J_num)
    print(J_ana)
    print ("max_abs_err:", np.max(np.abs(J_num - J_ana)))
    print(J_num_lm)
    print(J_ana_lm)
    print("max_abs_err:", np.max(np.abs(J_num_lm - J_ana_lm)))

    pose2 = np.array([0.0, 0.0, 3.13])
    landmark2 = np.array([-1.0, 0.01])

    J_num2 = numeric_jacobian_pose(pose2, landmark2, eps=1e-5)
    J_ana2 = analytic_jacobian_pose(pose2, landmark2, eps_denom=1e-12)
    J_num_lm2 = numeric_landmark_jacobian(pose2, landmark2, eps=1e-5)
    J_ana_lm2 = analytic_landmark_jacobian(pose2, landmark2, eps_denom=1e-12)

    print("\ncase 2 (wrap danger)")
    print(J_num2)
    print(J_ana2)
    print("max_abs_err:", np.max(np.abs(J_num2 - J_ana2)))
    print(J_num_lm2)
    print(J_ana_lm2)
    print("max_abs_err:", np.max(np.abs(J_num_lm2 - J_ana_lm2)))