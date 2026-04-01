# Flatness-based prediction with Newton-Raphson Flow for PX4-ROS2 Deployment

A ROS 2 differential-flatness based Newton-Raphson controller for quadrotors. Exploits the differential flatness property of quadrotor dynamics — where flat outputs `[x, y, z, yaw]` fully determine the state and inputs — to compute thrust and body rate commands directly without iterative Jacobian inversion.

## Approach

Quadrotors are differentially flat systems: given a trajectory in flat output space and its derivatives, the full state and control inputs can be recovered algebraically. This controller:

1. Takes position-level references from `quad_trajectories`
2. Computes derivatives via JAX autodiff to build the flat state `[sigma, sigma_dot, sigma_ddot]` (12D)
3. Maps the flat state directly to thrust and body rates through differential geometry
4. Applies either a **thrust CBF** or **simple clipping** to enforce actuator limits

## Key Features

- **Baseline and workshop profiles** — repeatable diff-flat NR tuning presets for comparison runs
- **Differential-flatness operating point** — optional `--ff` mode injects nominal flat-state acceleration and jerk from the shared trajectory model
- **Thrust Control Barrier Function** — quadratic barrier for smooth thrust constraint enforcement (default)
- **Dual implementations** — JAX (JIT-compiled, default) and NumPy (reference)
- **PX4 integration** — publishes attitude setpoints and offboard commands via `px4_msgs`
- **Structured logging** — optional CSV logging via ROS2Logger

## Control Profiles

| Profile | Lookahead | `alpha` | Integral gain | Integral limit | Iterations | Damping |
|---------|-----------|---------|---------------|----------------|------------|---------|
| `baseline` | `0.8 s` | `[20, 30, 30, 30]` | `[0, 0, 0, 0]` | `[0, 0, 0, 0]` | `1` | `1.0` |
| `workshop` | `0.5 s` | `[24, 34, 34, 24]` | `[0.30, 0.30, 0.45, 0.10]` | `[0.75, 0.75, 0.50, 0.30]` | `2` | `0.65` |

Both profiles use the thrust CBF (`CBF_GAMMA = 10.0`, thrust bounds `3.0 N` to `27.0 N`) by default.

## Feedforward Operating Point (`--ff`)

When `--ff` is enabled, the node builds a JIT-compiled flatness helper from the selected trajectory in `quad_trajectories` and evaluates:

- `x_df_ff = [sigma, sigma_dot, sigma_ddot]`
- `u_df_ff = sigma_dddot`

at the current trajectory time. The controller then keeps a deviation state `x_df_dev` and runs the NR update around the moving operating point:

- effective flat state: `x_df = x_df_ff + x_df_dev`
- effective flat input: `u_df = u_df_ff + alpha * nr_step + cbf_term`

This keeps the existing diff-flat controller structure intact while letting experiments compare `baseline+noff`, `baseline+ff`, `workshop+noff`, and `workshop+ff` without hand-editing code.

## Usage

```bash
source install/setup.bash

# Fly a helix in simulation (JAX backend, default)
ros2 run nr_diff_flat_px4 run_node --platform sim --trajectory helix

# Use NumPy backend
ros2 run nr_diff_flat_px4 run_node --platform sim --trajectory circle_horz --ctrl-type numpy

# Run the workshop profile
ros2 run nr_diff_flat_px4 run_node --platform sim --trajectory fig8_contraction --nr-profile workshop

# Run the baseline profile with the flatness feedforward operating point
ros2 run nr_diff_flat_px4 run_node --platform sim --trajectory fig8_contraction --ff

# Hardware flight with logging
ros2 run nr_diff_flat_px4 run_node --platform hw --trajectory fig8_horz --log
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--platform {sim,hw}` | Target platform (required) |
| `--trajectory {hover,yaw_only,circle_horz,...}` | Trajectory type (required) |
| `--ctrl-type {jax,numpy}` | Controller backend (default: `jax`) |
| `--nr-profile {baseline,workshop}` | Diff-flat NR profile |
| `--hover-mode {1..8}` | Hover sub-mode (1-4 for hardware) |
| `--ff` | Enable the flatness feedforward operating point |
| `--log` | Enable CSV data logging |
| `--log-file NAME` | Custom log filename |
| `--double-speed` | 2x trajectory speed |
| `--short` | Short variant (fig8_vert) |
| `--spin` | Enable yaw rotation |
| `--flight-period SEC` | Custom flight duration |

## Validation Snapshot

The integrated feedforward path was validated in headless SITL on `fig8_contraction` on April 1, 2026:

- `baseline+noff`: `1.3242 m` position RMSE, `0.6693 ms` mean compute time
- `baseline+ff`: `1.3496 m` position RMSE, `0.6683 ms` mean compute time
- `workshop+noff`: `0.5388 m` position RMSE, `0.6787 ms` mean compute time
- `workshop+ff`: `0.5535 m` position RMSE, `0.6743 ms` mean compute time

So this slice is integration-complete and analysis-visible. On this first checked trajectory, the workshop profile delivered the real gain, while the feedforward operating point did **not** improve either the baseline or workshop profile.

## Dependencies

- [quad_trajectories](https://github.com/evannsm/quad_trajectories) — trajectory definitions
- [quad_platforms](https://github.com/evannsm/quad_platforms) — platform abstraction
- [ROS2Logger](https://github.com/evannsm/ROS2Logger) — experiment logging
- [px4_msgs](https://github.com/PX4/px4_msgs) — PX4 ROS 2 message definitions
- JAX / jaxlib

## Package Structure

```
nr_diff_flat_px4/
├── nr_diff_flat_px4/
│   ├── run_node.py              # CLI entry point and argument parsing
│   └── ros2px4_node.py          # ROS 2 node (subscriptions, publishers, control loop)
└── nr_diff_flat_px4_utils/
    ├── controller/
    │   ├── nr_diff_flat_px4_jax.py  # JAX implementation (default)
    │   └── nr_diff_flat_px4_numpy.py# NumPy reference implementation
    ├── px4_utils/               # PX4 interface and flight phase management
    ├── transformations/         # Yaw adjustment utilities
    ├── main_utils.py            # Helper functions
    └── jax_utils.py             # JAX configuration
```

## Installation

```bash
# Inside a ROS 2 workspace src/ directory
git clone git@github.com:evannsm/nr_diff_flat_px4.git
cd .. && colcon build --symlink-install
```

## License

MIT
