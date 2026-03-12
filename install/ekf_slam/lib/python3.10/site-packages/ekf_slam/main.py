import numpy as np
import matplotlib.pyplot as plt
from ekf import EKFSLAM
from sim import MultiLandmarkSim

def confidence_ellipse_params(Pxy, n_std=2.0, eps=1e-12):
    """
    Returns (width, height, angle_deg) of the n-std ellipse from a 2x2 covariance.
    width/height are full lengths (diameters), angle is rotation in degrees.
    """
    # Symmetrize for numerical safety
    Pxy = 0.5 * (Pxy + Pxy.T)
    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(Pxy)
    vals = np.maximum(vals, eps)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Angle of the major axis
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # Diameter = 2 * n_std * sqrt(eigenvalue)
    width = 2.0 * n_std * np.sqrt(vals[0])
    height = 2.0 * n_std * np.sqrt(vals[1])
    return width, height, angle

def plot_cov_ellipse(ax, mean_xy, Pxy, n_std=2.0, **kwargs):
    from matplotlib.patches import Ellipse
    w, h, ang = confidence_ellipse_params(Pxy, n_std=n_std)
    e = Ellipse(xy=mean_xy, width=w, height=h, angle=ang, fill=False, **kwargs)
    ax.add_patch(e)
    return e

def main():
    # ----------------------
    # EKF-SLAM init
    # ----------------------
    slam = EKFSLAM(
        init_pose=np.array([0.0, 0.0, 0.0]),
        init_cov=np.diag([1e-3, 1e-3, 1e-3])
    )

    Q = np.diag([0.05**2, 0.02**2])  # [v, w] noise
    R = np.diag([0.10**2, (np.deg2rad(2.0))**2])  # [range, bearing] noise

    # ----------------------
    # Simulation setup
    # ----------------------
    landmarks = {
        1: (5.0, 5.0),
        2: (8.0, 0.0),
        3: (3.0, -4.0),
        4: (-2.0, 4.0),
        5: (6.0, -3.0),
    }

    sim = MultiLandmarkSim(
        landmarks=landmarks,
        dt=0.1,
        max_range=8.0,
        sigma_r=0.10,
        sigma_phi_deg=2.0,
        seed=0
    )

    # ----------------------
    # Logging containers
    # ----------------------
    gt_traj = []
    ekf_traj = []
    pose_sigma = []       # [sx, sy, stheta]
    NIS_log = []          # normalized innovation squared per observation
    innov_r = []          # range innovations
    innov_phi = []        # bearing innovations
    nis_time = []         # step index for each NIS sample
    num_obs_log = []      # how many observations each step
    t_log = []

    # Keep the last innovation stats from inside the EKF update:
    # We'll compute them outside by recomputing y, H, S right before update
    # NOTE: This slightly duplicates work but is clean for plotting.

    v_cmd, w_cmd = 1.0, 0.15
    steps = 300

    for k in range(steps):
        u, obs = sim.step(v_cmd, w_cmd)
        observations = [(lm_id, z, R) for (lm_id, z) in obs]

        # Before slam.step(), we can estimate per-observation innovations by doing:
        # predict first, then for each observation compute y, H, S using current state,
        # but your slam.step already does predict + updates.
        # So here we do: slam.predict, compute stats + update per observation like slam.step.
        slam.predict(u, Q)

        # Log observation count
        num_obs_log.append(len(observations))
        t_log.append(k * sim.dt)

        # For each observation, compute innovation + NIS, then update
        for landmark_id, z, Rk in observations:
            if landmark_id not in slam.id_to_index:
                slam.initialize_landmark(landmark_id, z, Rk)

            i = slam.id_to_index[landmark_id]
            pose = slam.pose()
            lm = slam.get_landmark(i)

            # predicted measurement
            z_hat = sim.h(pose, lm) if hasattr(sim, "h") else None
            # If sim doesn't have h(), use your models.h by importing it.
            if z_hat is None:
                from models import h, wrap_angle
                z_hat = h(pose, lm)
                y = z.astype(float) - z_hat
                y[1] = wrap_angle(y[1])
                from ekf import build_full_H
                H = build_full_H(pose, lm, i, slam.N)
                S = H @ slam.P @ H.T + Rk
            else:
                # Fallback if sim exposes h + wrap_angle similarly (unlikely)
                y = z.astype(float) - z_hat
                y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
                from ekf import build_full_H
                H = build_full_H(pose, lm, i, slam.N)
                S = H @ slam.P @ H.T + Rk

            # NIS = y^T S^-1 y (scalar)
            Sinv = np.linalg.inv(S)
            nis = float(y.T @ Sinv @ y)

            innov_r.append(float(y[0]))
            innov_phi.append(float(y[1]))
            NIS_log.append(nis)
            nis_time.append(k)

            # Do the actual EKF update
            slam.update_known_landmark(landmark_id, z, Rk)

        # Log trajectories after finishing updates
        gt_traj.append(sim.gt_pose.copy())
        ekf_traj.append(slam.pose().copy())

        # Pose sigma
        Prr = slam.P[0:3, 0:3]
        pose_sigma.append(np.sqrt(np.clip(np.diag(Prr), 0.0, np.inf)))

    gt_traj = np.array(gt_traj)
    ekf_traj = np.array(ekf_traj)
    pose_sigma = np.array(pose_sigma)
    t_log = np.array(t_log)
    NIS_log = np.array(NIS_log)
    innov_r = np.array(innov_r)
    innov_phi = np.array(innov_phi)
    nis_time = np.array(nis_time)
    num_obs_log = np.array(num_obs_log)

    # ----------------------
    # PRINT: headline numbers (good for LinkedIn post caption)
    # ----------------------
    final_pose = slam.pose()
    final_err_xy = np.linalg.norm(final_pose[:2] - sim.gt_pose[:2])
    print("\n=== EKF-SLAM Summary ===")
    print(f"Steps: {steps}, dt: {sim.dt:.3f}s")
    print(f"Final GT pose:  [{sim.gt_pose[0]: .3f}, {sim.gt_pose[1]: .3f}, {sim.gt_pose[2]: .3f}]")
    print(f"Final EKF pose: [{final_pose[0]: .3f}, {final_pose[1]: .3f}, {final_pose[2]: .3f}]")
    print(f"Final position error (m): {final_err_xy:.3f}")
    print(f"Landmarks initialized: {slam.N}")

    # Print landmark table
    print("\nLandmark estimates (GT vs EKF):")
    print("ID |   GT_x   GT_y  ||  EKF_x  EKF_y  ||  err(m)")
    for lm_id, (gx, gy) in landmarks.items():
        if lm_id in slam.id_to_index:
            lm = slam.get_landmark(slam.id_to_index[lm_id])
            err = np.linalg.norm(lm - np.array([gx, gy]))
            print(f"{lm_id:2d} | {gx:6.2f} {gy:6.2f} || {lm[0]:6.2f} {lm[1]:6.2f} || {err:6.3f}")
        else:
            print(f"{lm_id:2d} | {gx:6.2f} {gy:6.2f} ||  (not observed)")

    # ----------------------
    # FIGURE 1: Trajectory + landmarks + uncertainty ellipses
    # ----------------------
    fig1, ax1 = plt.subplots()
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], label="GT Trajectory")
    ax1.plot(ekf_traj[:, 0], ekf_traj[:, 1], label="EKF Trajectory")
    ax1.set_aspect("equal", "box")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Trajectory (GT vs EKF)")

    # Plot GT landmarks
    for lm_id, (x, y) in landmarks.items():
        ax1.scatter([x], [y])
        ax1.text(x, y, f"GT{lm_id}")

    # Plot estimated landmarks + 2-sigma ellipses
    for lm_id, i in slam.id_to_index.items():
        lm = slam.get_landmark(i)
        ax1.scatter([lm[0]], [lm[1]])
        ax1.text(lm[0], lm[1], f"EKF{lm_id}")

        j = 3 + 2 * i
        Pll = slam.P[j:j+2, j:j+2]
        plot_cov_ellipse(ax1, lm, Pll, n_std=2.0, linewidth=1.0)

    # Robot pose 2-sigma ellipse (x,y)
    Pxy = slam.P[0:2, 0:2]
    plot_cov_ellipse(ax1, slam.pose()[0:2], Pxy, n_std=2.0, linewidth=1.5)

    ax1.grid(True)
    ax1.legend()

    # ----------------------
    # FIGURE 2: Pose components vs time
    # ----------------------
    fig2, ax2 = plt.subplots()
    ax2.plot(t_log, gt_traj[:, 0], label="GT x")
    ax2.plot(t_log, ekf_traj[:, 0], label="EKF x")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("x [m]")
    ax2.set_title("x over time")
    ax2.grid(True)
    ax2.legend()

    fig3, ax3 = plt.subplots()
    ax3.plot(t_log, gt_traj[:, 1], label="GT y")
    ax3.plot(t_log, ekf_traj[:, 1], label="EKF y")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("y [m]")
    ax3.set_title("y over time")
    ax3.grid(True)
    ax3.legend()

    fig4, ax4 = plt.subplots()
    ax4.plot(t_log, gt_traj[:, 2], label="GT theta")
    ax4.plot(t_log, ekf_traj[:, 2], label="EKF theta")
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("theta [rad]")
    ax4.set_title("theta over time")
    ax4.grid(True)
    ax4.legend()

    # ----------------------
    # FIGURE 3: Pose uncertainty (sigma) vs time
    # ----------------------
    fig5, ax5 = plt.subplots()
    ax5.plot(t_log, pose_sigma[:, 0], label="sigma_x")
    ax5.plot(t_log, pose_sigma[:, 1], label="sigma_y")
    ax5.plot(t_log, pose_sigma[:, 2], label="sigma_theta")
    ax5.set_xlabel("time [s]")
    ax5.set_ylabel("std dev")
    ax5.set_title("Pose uncertainty (sqrt(diag(P_rr)))")
    ax5.grid(True)
    ax5.legend()

    # ----------------------
    # FIGURE 4: Innovations (residuals)
    # ----------------------
    fig6, ax6 = plt.subplots()
    ax6.plot(nis_time * sim.dt, innov_r, label="range innovation y_r [m]")
    ax6.set_xlabel("time [s]")
    ax6.set_ylabel("innovation [m]")
    ax6.set_title("Range innovation over time")
    ax6.grid(True)
    ax6.legend()

    fig7, ax7 = plt.subplots()
    ax7.plot(nis_time * sim.dt, innov_phi, label="bearing innovation y_phi [rad]")
    ax7.set_xlabel("time [s]")
    ax7.set_ylabel("innovation [rad]")
    ax7.set_title("Bearing innovation over time")
    ax7.grid(True)
    ax7.legend()

    # ----------------------
    # FIGURE 5: NIS (Normalized Innovation Squared)
    # ----------------------
    # For a 2D measurement, NIS ~ chi-square(df=2). Typical “okay” range is around 0..~6
    fig8, ax8 = plt.subplots()
    ax8.plot(nis_time * sim.dt, NIS_log, label="NIS")
    ax8.set_xlabel("time [s]")
    ax8.set_ylabel("NIS = y^T S^{-1} y")
    ax8.set_title("Consistency check (NIS) over time")
    ax8.grid(True)
    ax8.legend()

    # ----------------------
    # FIGURE 6: Observation count vs time (shows sensor visibility / max_range impact)
    # ----------------------
    fig9, ax9 = plt.subplots()
    ax9.plot(t_log, num_obs_log, label="#observations per step")
    ax9.set_xlabel("time [s]")
    ax9.set_ylabel("count")
    ax9.set_title("How many landmarks were observed each step")
    ax9.grid(True)
    ax9.legend()

    plt.show()

if __name__ == "__main__":
    main()
