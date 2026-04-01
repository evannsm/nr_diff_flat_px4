import numpy as np


GRAVITY = 9.8  # Match Gazebo world
CBF_GAMMA = 10.0
CBF_THRUST_MIN = 3.0
CBF_THRUST_MAX = 27.0
CLIP_THRUST_MIN = 0.5
CLIP_THRUST_MAX = 27.0


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def get_tracking_error(ref, pred):
    err = ref - pred
    err[3][0] = _wrap_to_pi(err[3][0])
    return err


def _thrust_cbf(curr_thrust, u_df_xyz, sigmad2_xyz, mass):
    """Control Barrier Function for thrust in flat-input space."""
    h = -(5.0 / 72.0) * (curr_thrust - CBF_THRUST_MIN) * (curr_thrust - CBF_THRUST_MAX)
    dhdtau = -(5.0 / 72.0) * (2.0 * curr_thrust - (CBF_THRUST_MIN + CBF_THRUST_MAX))

    a3 = np.array([[0.0, 0.0, -1.0]]).T
    grav_vector = GRAVITY * a3
    accels = sigmad2_xyz + grav_vector
    x_norm = max(np.linalg.norm(accels), 1e-8)
    xT_over_normx = accels / x_norm
    phi = dhdtau * mass * xT_over_normx

    d = float((-phi.T @ u_df_xyz - CBF_GAMMA * h).squeeze())
    if d <= 0.0:
        return np.zeros((4, 1))

    phi_norm_sq = max(np.linalg.norm(phi) ** 2, 1e-12)
    v_xyz = (d / phi_norm_sq) * phi
    return np.vstack([v_xyz, np.zeros((1, 1))])


def nr_diff_flat_px4_numpy(
    state,
    last_input,
    x_df,
    u_df_ff,
    use_feedforward,
    ref,
    error_integral,
    T_lookahead,
    step,
    mass,
    ROT,
    yaw_dot,
    alpha,
    integral_gain,
    integral_limit,
    num_iterations,
    iteration_damping,
    use_thrust_cbf,
):
    """Differential-flatness based Newton-Raphson controller (NumPy version)."""
    curr_x, curr_y, curr_z = state[0][0], state[1][0], state[2][0]
    curr_vx, curr_vy, curr_vz = state[3][0], state[4][0], state[5][0]
    curr_yaw = state[8][0]
    curr_yawdot = last_input[3][0]
    curr_thrust = last_input[0][0]

    z1 = np.array([[curr_x, curr_y, curr_z, curr_yaw]]).T
    z2 = np.array([[curr_vx, curr_vy, curr_vz, curr_yawdot]]).T
    candidate_z3 = np.array(x_df[8:12], dtype=np.float64)
    dgdu_inv = (2.0 / T_lookahead**2) * np.eye(4)
    clipped_integral = np.clip(error_integral, -integral_limit, integral_limit).reshape(4, 1)

    corrected_u_df = np.zeros((4, 1))
    cbf_term = np.zeros((4, 1))
    for _ in range(int(num_iterations)):
        pred = z1 + z2 * T_lookahead + 0.5 * candidate_z3 * T_lookahead**2
        error = get_tracking_error(ref, pred) + integral_gain.reshape(4, 1) * clipped_integral
        nr_step = dgdu_inv @ error
        nominal_u_df = u_df_ff if use_feedforward else np.zeros_like(u_df_ff)
        raw_u_df = nominal_u_df + alpha.reshape(4, 1) * nr_step

        if use_thrust_cbf:
            cbf_term = _thrust_cbf(curr_thrust, raw_u_df[0:3], candidate_z3[0:3], mass)
        else:
            cbf_term = np.zeros((4, 1))

        corrected_u_df = raw_u_df + cbf_term
        candidate_z3 = candidate_z3 + iteration_damping * corrected_u_df * step

    a_df = np.block([
        [np.zeros((4, 4)), np.eye(4), np.zeros((4, 4))],
        [np.zeros((4, 4)), np.zeros((4, 4)), np.eye(4)],
        [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))],
    ])
    b_df = np.block([
        [np.zeros((4, 4))],
        [np.zeros((4, 4))],
        [np.eye(4)],
    ])

    x_dot_df = a_df @ x_df + b_df @ corrected_u_df
    x_df = x_df + x_dot_df * step

    sigmad2 = x_df[8:12]
    sigmad3 = corrected_u_df

    a1 = np.array([[1.0, 0.0, 0.0]]).T
    a2 = np.array([[0.0, 1.0, 0.0]]).T
    a3 = np.array([[0.0, 0.0, -1.0]]).T

    accels = sigmad2[0:3] + GRAVITY * a3
    thrust = mass * max(np.linalg.norm(accels), 1e-8)
    if use_thrust_cbf:
        clipped_thrust = thrust
    else:
        clipped_thrust = np.clip(thrust, CLIP_THRUST_MIN, CLIP_THRUST_MAX)

    b3 = accels / max(np.linalg.norm(accels), 1e-8)
    e1 = np.cos(curr_yaw) * a1 + np.sin(curr_yaw) * a2
    val2 = np.cross(b3, e1, axis=0)
    b2 = val2 / max(np.linalg.norm(val2), 1e-8)
    b1 = np.cross(b2, b3, axis=0)

    jerk = sigmad3[0:3]
    val3 = jerk - (jerk.T @ b3) * b3
    p = (b2.T @ ((mass / -max(clipped_thrust, 1e-8)) * val3)).item()
    q = (b1.T @ ((mass / -max(clipped_thrust, 1e-8)) * val3)).item()

    a_mat = np.hstack((e1, b2, a3))
    b_mat = ROT
    x_known = np.array([yaw_dot])
    y_known = np.array([p, q])

    a1_mat = a_mat[:, :2]
    a2_mat = a_mat[:, 2:]
    b1_mat = b_mat[:, 2:]
    b2_mat = b_mat[:, :2]
    rhs = (b2_mat @ y_known) - (a2_mat @ x_known)
    m_mat = np.hstack((a1_mat, -b1_mat))
    _, _, r = np.linalg.solve(m_mat, rhs)
    r = -r

    max_rate = 0.8
    p = np.clip(p, -max_rate, max_rate)
    q = np.clip(q, -max_rate, max_rate)
    r = np.clip(r, -max_rate, max_rate)

    u = np.array([[clipped_thrust, p, q, r]]).T
    return u, x_df, cbf_term
