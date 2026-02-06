# nr_diff_flat

A ROS 2 differential-flatness based Newton-Raphson controller for quadrotors. Exploits the differential flatness property of quadrotor dynamics — where flat outputs `[x, y, z, yaw]` fully determine the state and inputs — to compute thrust and body rate commands directly without iterative Jacobian inversion.

## Approach

Quadrotors are differentially flat systems: given a trajectory in flat output space and its derivatives, the full state and control inputs can be recovered algebraically. This controller:

1. Takes position-level references from `quad_trajectories`
2. Computes derivatives via JAX autodiff to build the flat state `[sigma, sigma_dot, sigma_ddot]` (12D)
3. Maps the flat state directly to thrust and body rates through differential geometry
4. Applies either a **thrust CBF** or **simple clipping** to enforce actuator limits

## Key Features

- **Differential flatness** — direct algebraic mapping from flat outputs to control inputs
- **Thrust Control Barrier Function** — quadratic barrier for smooth thrust constraint enforcement (default)
- **Dual implementations** — JAX (JIT-compiled, default) and NumPy (reference)
- **PX4 integration** — publishes attitude setpoints and offboard commands via `px4_msgs`
- **Structured logging** — optional CSV logging via ROS2Logger

## Control Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ALPHA` | `[20, 30, 30, 30]` | Control gains `[x, y, z, yaw]` |
| `USE_THRUST_CBF` | `True` | Use CBF for thrust limits (else clipping) |
| `CBF_GAMMA` | `10.0` | CBF convergence rate |
| `CBF_THRUST_MIN` | `3.0 N` | Minimum thrust (CBF) |
| `CBF_THRUST_MAX` | `27.0 N` | Maximum thrust (CBF) |

## Usage

```bash
source install/setup.bash

# Fly a helix in simulation (JAX backend, default)
ros2 run nr_diff_flat run_node --platform sim --trajectory helix

# Use NumPy backend
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --ctrl-type numpy

# Hardware flight with logging
ros2 run nr_diff_flat run_node --platform hw --trajectory fig8_horz --log
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--platform {sim,hw}` | Target platform (required) |
| `--trajectory {hover,yaw_only,circle_horz,...}` | Trajectory type (required) |
| `--ctrl-type {jax,numpy}` | Controller backend (default: `jax`) |
| `--hover-mode {1..8}` | Hover sub-mode (1-4 for hardware) |
| `--log` | Enable CSV data logging |
| `--log-file NAME` | Custom log filename |
| `--double-speed` | 2x trajectory speed |
| `--short` | Short variant (fig8_vert) |
| `--spin` | Enable yaw rotation |
| `--flight-period SEC` | Custom flight duration |

## Dependencies

- [quad_trajectories](https://github.com/evannsm/quad_trajectories) — trajectory definitions
- [quad_platforms](https://github.com/evannsm/quad_platforms) — platform abstraction
- [ROS2Logger](https://github.com/evannsm/ROS2Logger) — experiment logging
- [px4_msgs](https://github.com/PX4/px4_msgs) — PX4 ROS 2 message definitions
- JAX / jaxlib

## Package Structure

```
nr_diff_flat/
├── nr_diff_flat/
│   ├── run_node.py              # CLI entry point and argument parsing
│   └── ros2px4_node.py          # ROS 2 node (subscriptions, publishers, control loop)
└── nr_diff_flat_utils/
    ├── controller/
    │   ├── nr_diff_flat_jax.py  # JAX implementation (default)
    │   └── nr_diff_flat_numpy.py# NumPy reference implementation
    ├── px4_utils/               # PX4 interface and flight phase management
    ├── transformations/         # Yaw adjustment utilities
    ├── main_utils.py            # Helper functions
    └── jax_utils.py             # JAX configuration
```

## Installation

```bash
# Inside a ROS 2 workspace src/ directory
git clone git@github.com:evannsm/nr_diff_flat.git
cd .. && colcon build --symlink-install
```

## License

MIT
