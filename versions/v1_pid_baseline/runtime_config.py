"""Central runtime backend switch for auto-drive-1 demos.

Edit BACKEND here to switch modes:

- "cpu"       : Pure NumPy PPO expert + CPU rendering (default, fastest for driving demo)
- "torch_gpu" : Torch expert on CUDA + CUDA image pipeline (for GPU training / obs on GPU)

Note: "cpu" is NOT torch running on CPU. It bypasses torch entirely and uses numpy_expert.
"""

from __future__ import annotations

import numpy as np

BACKEND = "cpu"

_ACTIVE_EXPERT = None


def get_backend_label():
    if BACKEND == "torch_gpu":
        return "torch_gpu"
    if BACKEND != "cpu":
        raise ValueError(f"Unknown BACKEND={BACKEND!r}. Use 'cpu' or 'torch_gpu'.")
    return "cpu"


def _load_base_expert():
    """Load the raw expert implementation for the selected backend."""
    if get_backend_label() == "torch_gpu":
        from metadrive.examples.ppo_expert.torch_expert import torch_expert

        return torch_expert

    from metadrive.examples.ppo_expert import numpy_expert

    return numpy_expert.expert


def _wrap_cruise_expert(base_expert, target_speed_kmh):
    """Bias throttle toward a target cruise speed."""

    def expert(vehicle, deterministic=False, need_obs=False):
        result = base_expert(vehicle, deterministic=deterministic, need_obs=need_obs)
        if need_obs:
            action, obs = result
        else:
            action, obs = result, None

        action = np.array(action, dtype=np.float32)
        speed = vehicle.speed_km_h
        if speed < target_speed_kmh - 4.0:
            action[1] = max(float(action[1]), 0.82)
        elif speed > target_speed_kmh + 6.0:
            action[1] = min(float(action[1]), 0.15)

        if need_obs:
            return action, obs
        return action

    return expert


def get_expert_fn():
    """Return the active expert policy, after apply_runtime_expert_patch()."""
    if _ACTIVE_EXPERT is not None:
        return _ACTIVE_EXPERT
    return _load_base_expert()


def apply_runtime_expert_patch(cruise_target_kmh=None):
    """Patch MetaDrive so every expert entry point uses runtime_config."""
    global _ACTIVE_EXPERT

    base_expert = _load_base_expert()
    if cruise_target_kmh is not None:
        _ACTIVE_EXPERT = _wrap_cruise_expert(base_expert, cruise_target_kmh)
    else:
        _ACTIVE_EXPERT = base_expert

    import metadrive.examples as md_examples
    import metadrive.examples.ppo_expert as ppo_expert
    import metadrive.policy.expert_policy as expert_policy
    import metadrive.policy.manual_control_policy as manual_control_policy

    ppo_expert.expert = _ACTIVE_EXPERT
    md_examples.expert = _ACTIVE_EXPERT
    manual_control_policy.expert = _ACTIVE_EXPERT
    expert_policy.expert = _ACTIVE_EXPERT

    from metadrive.engine.logger import get_logger

    label = get_backend_label()
    get_logger().info(
        "Runtime expert backend: %s (runtime_config.BACKEND=%r)",
        label,
        BACKEND,
    )


def get_render_config():
    """MetaDrive render / observation device settings for the selected backend."""
    if get_backend_label() == "torch_gpu":
        return dict(
            image_on_cuda=True,
            multi_thread_render=True,
            render_pipeline=False,
        )
    return dict(
        image_on_cuda=False,
        multi_thread_render=False,
        render_pipeline=False,
    )
