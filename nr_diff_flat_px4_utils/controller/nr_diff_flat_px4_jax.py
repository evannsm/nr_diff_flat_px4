import jax.numpy as jnp
from jax import lax

from nr_diff_flat_px4_utils.jax_utils import jit

GRAVITY = 9.8  # Match Gazebo world
CBF_GAMMA = 10.0
CBF_THRUST_MIN = 3.0
CBF_THRUST_MAX = 27.0
CLIP_THRUST_MIN = 0.5
CLIP_THRUST_MAX = 27.0


def _wrap_to_pi(angle):
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def get_tracking_error(ref, pred):
    err = ref - pred
    return err.at[3].set(_wrap_to_pi(err[3]))


def _thrust_cbf_jax(curr_thrust, u_df_xyz, sigmad2_xyz, mass):
    """Return the thrust CBF correction in flat-input space."""
    a3 = jnp.array([[0.0, 0.0, -1.0]]).T
    grav_vector = GRAVITY * a3

    tau = curr_thrust
    h = -(5.0 / 72.0) * (tau - CBF_THRUST_MIN) * (tau - CBF_THRUST_MAX)
    dhdtau = -(5.0 / 72.0) * (2.0 * tau - (CBF_THRUST_MIN + CBF_THRUST_MAX))

    accels = sigmad2_xyz + grav_vector
    accels_norm = jnp.maximum(jnp.linalg.norm(accels), 1e-8)
    xT_over_normx = accels / accels_norm
    phi = dhdtau * mass * xT_over_normx

    d = (-phi.T @ u_df_xyz - CBF_GAMMA * h).squeeze()
    phi_norm_sq = jnp.maximum(jnp.sum(phi**2), 1e-12)
    v_xyz = (d / phi_norm_sq) * phi
    v_xyz = jnp.where(d <= 0.0, jnp.zeros_like(v_xyz), v_xyz)
    return jnp.vstack([v_xyz, jnp.zeros((1, 1))])


def _flat_dynamics_mats():
    a_df = jnp.block([
        [jnp.zeros((4, 4)), jnp.eye(4), jnp.zeros((4, 4))],
        [jnp.zeros((4, 4)), jnp.zeros((4, 4)), jnp.eye(4)],
        [jnp.zeros((4, 4)), jnp.zeros((4, 4)), jnp.zeros((4, 4))],
    ])
    b_df = jnp.block([
        [jnp.zeros((4, 4))],
        [jnp.zeros((4, 4))],
        [jnp.eye(4)],
    ])
    return a_df, b_df


@jit
def NR_tracker_flat(
    STATE,
    INPUT,
    x_df,
    ref,
    error_integral,
    T_LOOKAHEAD,
    step,
    MASS,
    ROT,
    yaw_dot,
    alpha,
    integral_gain,
    integral_limit,
    num_iterations,
    iteration_damping,
    use_thrust_cbf,
):
    """Differential-flatness based Newton-Raphson controller (JAX version)."""
    curr_x, curr_y, curr_z, curr_yaw = STATE[0][0], STATE[1][0], STATE[2][0], STATE[-1][0]
    curr_vx, curr_vy, curr_vz = STATE[3][0], STATE[4][0], STATE[5][0]
    curr_yawdot = INPUT[3][0]
    curr_thrust = INPUT[0][0]

    z1 = jnp.array([[curr_x, curr_y, curr_z, curr_yaw]]).T
    z2 = jnp.array([[curr_vx, curr_vy, curr_vz, curr_yawdot]]).T
    z3_initial = x_df[8:12]
    dgdu_inv = (2.0 / T_LOOKAHEAD**2) * jnp.eye(4)
    clipped_integral = jnp.clip(error_integral, -integral_limit, integral_limit).reshape(4, 1)

    def nr_iteration(_, carry):
        candidate_z3, corrected_u_df, cbf_term = carry
        pred = z1 + z2 * T_LOOKAHEAD + 0.5 * candidate_z3 * T_LOOKAHEAD**2
        error = get_tracking_error(ref, pred) + integral_gain * clipped_integral
        nr_step = dgdu_inv @ error
        raw_u_df = alpha * nr_step

        cbf_term = lax.cond(
            use_thrust_cbf,
            lambda _: _thrust_cbf_jax(curr_thrust, raw_u_df[0:3], candidate_z3[0:3], MASS),
            lambda _: jnp.zeros_like(raw_u_df),
            operand=None,
        )
        corrected_u_df = raw_u_df + cbf_term
        updated_z3 = candidate_z3 + iteration_damping * corrected_u_df * step
        return updated_z3, corrected_u_df, cbf_term

    _, corrected_u_df, cbf_term = lax.fori_loop(
        0,
        num_iterations,
        nr_iteration,
        (z3_initial, jnp.zeros((4, 1)), jnp.zeros((4, 1))),
    )

    a_df, b_df = _flat_dynamics_mats()
    x_dot_df = a_df @ x_df + b_df @ corrected_u_df
    x_df = x_df + x_dot_df * step

    sigmad2 = x_df[8:12]
    sigmad3 = corrected_u_df

    a1 = jnp.array([[1.0, 0.0, 0.0]]).T
    a2 = jnp.array([[0.0, 1.0, 0.0]]).T
    a3 = jnp.array([[0.0, 0.0, -1.0]]).T

    accels = sigmad2[0:3] + GRAVITY * a3
    thrust = MASS * jnp.maximum(jnp.linalg.norm(accels), 1e-8)
    clipped_thrust = lax.cond(
        use_thrust_cbf,
        lambda _: thrust,
        lambda _: jnp.clip(thrust, CLIP_THRUST_MIN, CLIP_THRUST_MAX),
        operand=None,
    )

    b3 = accels / jnp.maximum(jnp.linalg.norm(accels), 1e-8)
    e1 = jnp.cos(curr_yaw) * a1 + jnp.sin(curr_yaw) * a2
    val2 = jnp.cross(b3, e1, axis=0)
    b2 = val2 / jnp.maximum(jnp.linalg.norm(val2), 1e-8)
    b1 = jnp.cross(b2, b3, axis=0)

    jerk = sigmad3[0:3]
    val3 = jerk - (jerk.T @ b3) * b3
    p = b2.T @ ((MASS / -jnp.maximum(clipped_thrust, 1e-8)) * val3)
    q = b1.T @ ((MASS / -jnp.maximum(clipped_thrust, 1e-8)) * val3)

    a_mat = jnp.hstack((e1, b2, a3))
    b_mat = ROT
    x_known = jnp.array([yaw_dot])
    y_known = jnp.array([p[0][0], q[0][0]])

    a1_mat = a_mat[:, :2]
    a2_mat = a_mat[:, 2:]
    b1_mat = b_mat[:, 2:]
    b2_mat = b_mat[:, :2]
    rhs = (b2_mat @ y_known) - (a2_mat @ x_known)
    m_mat = jnp.hstack((a1_mat, -b1_mat))
    unknowns = jnp.linalg.solve(m_mat, rhs)
    _, _, r = unknowns
    r = -r

    max_rate = 0.8
    p = jnp.clip(p, -max_rate, max_rate)
    q = jnp.clip(q, -max_rate, max_rate)
    r = jnp.clip(r, -max_rate, max_rate)

    u = jnp.vstack(
        [
            clipped_thrust.reshape((1, 1)),
            p.reshape((1, 1)),
            q.reshape((1, 1)),
            r.reshape((1, 1)),
        ]
    )
    return u, x_df, cbf_term
