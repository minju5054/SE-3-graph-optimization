from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionConfig:
    dt: float = 0.1
    observation_step: int = 8
    inference_latency_steps: int = 3
    commit_horizon_steps: int = 1
    optimization_window_poses: int = 7
    inference_behavior: str = "continue_old"

    def __post_init__(self):
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.observation_step < 0:
            raise ValueError("observation_step must be non-negative.")
        if self.inference_latency_steps < 0:
            raise ValueError("inference_latency_steps must be non-negative.")
        if self.commit_horizon_steps < 0:
            raise ValueError("commit_horizon_steps must be non-negative.")
        if self.optimization_window_poses < 2:
            raise ValueError("optimization_window_poses must be at least two.")
        if self.inference_behavior not in {"continue_old", "hold_pose"}:
            raise ValueError(
                "inference_behavior must be 'continue_old' or 'hold_pose'."
            )

    @property
    def new_ready_step(self):
        return self.observation_step + self.inference_latency_steps

    @property
    def modification_step(self):
        return self.new_ready_step + self.commit_horizon_steps


def new_local_index(global_step, config):
    """Convert a wall-clock step to the observation-relative NEW index."""
    return int(global_step) - config.observation_step


def _old_pose_at(old, global_step):
    if global_step < 0:
        raise ValueError("global_step must be non-negative.")
    return old[min(global_step, len(old) - 1)].copy()


def executed_pose_before_modification(old, global_step, config):
    """Return the immutable policy result at one step up to modification."""
    old = np.asarray(old, dtype=float)
    if old.ndim != 2 or old.shape[1] != 3 or len(old) == 0:
        raise ValueError("Expected old shape [N, 3].")
    if global_step > config.modification_step:
        raise ValueError("This function only defines the committed prefix.")

    if (
        config.inference_behavior == "hold_pose"
        and global_step > config.observation_step
    ):
        return _old_pose_at(old, config.observation_step)
    return _old_pose_at(old, global_step)


def build_committed_prefix(old, config):
    """Build poses 0..modification_step, including the fixed transition anchor."""
    if config.modification_step >= len(old):
        raise ValueError("OLD chunk does not reach the modification step.")
    return np.vstack(
        [
            executed_pose_before_modification(old, step, config)
            for step in range(config.modification_step + 1)
        ]
    )


def local_transition_windows(old, new, config):
    """Extract time-aligned OLD/NEW windows at the modification point."""
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)
    valid_old = old.ndim == 2 and old.shape[1:] == (3,)
    valid_new = new.ndim == 2 and new.shape[1:] == (3,)
    if not valid_old or not valid_new:
        raise ValueError("Expected old/new shapes [N, 3].")

    start_global = config.modification_step
    stop_global = start_global + config.optimization_window_poses
    start_new = new_local_index(start_global, config)
    stop_new = start_new + config.optimization_window_poses
    if stop_global > len(old):
        raise ValueError("OLD chunk is too short for the transition window.")
    if start_new < 0 or stop_new > len(new):
        raise ValueError(
            "NEW chunk is too short for the aligned transition window."
        )

    old_window = old[start_global:stop_global].copy()
    old_window[0] = executed_pose_before_modification(
        old, start_global, config
    )
    new_window = new[start_new:stop_new].copy()
    return old_window, new_window


def assemble_local_transition(old, new, transition, config):
    """Preserve the prefix, insert a local window, then execute raw NEW suffix."""
    transition = np.asarray(transition, dtype=float)
    expected_shape = (config.optimization_window_poses, 3)
    if transition.shape != expected_shape:
        raise ValueError(f"Expected transition shape {expected_shape}.")

    prefix = build_committed_prefix(old, config)
    start_new = new_local_index(config.modification_step, config)
    suffix_start = start_new + config.optimization_window_poses
    return np.vstack([prefix[:-1], transition, new[suffix_start:]])


def assemble_hard_switch(old, new, config):
    """Switch directly to the wall-clock-aligned NEW pose at modification."""
    prefix = build_committed_prefix(old, config)
    start_new = new_local_index(config.modification_step, config)
    return np.vstack([prefix[:-1], new[start_new:]])


def assemble_continue_old(old, new, config):
    """Execute OLD for the comparison horizon, holding its terminal pose if needed."""
    total_poses = config.observation_step + len(new)
    return np.vstack([_old_pose_at(old, step) for step in range(total_poses)])


def aligned_new_reference(new, config, total_poses=None):
    """Place NEW local pose zero at the observation wall-clock step."""
    if total_poses is None:
        total_poses = config.observation_step + len(new)
    reference = np.full((total_poses, 3), np.nan, dtype=float)
    available = min(len(new), total_poses - config.observation_step)
    if available > 0:
        reference[
            config.observation_step : config.observation_step + available
        ] = new[:available]
    return reference
