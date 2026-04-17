from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as locomotion_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_velocity_toward_goal(
    env: ManagerBasedRLEnv,
    goal_x: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward world-frame velocity toward a goal line located at +x in each env frame."""
    asset = env.scene[asset_cfg.name]
    goal_dir_w = torch.zeros_like(asset.data.root_lin_vel_w[:, :2])
    goal_dir_w[:, 0] = 1.0
    return torch.sum(asset.data.root_lin_vel_w[:, :2] * goal_dir_w, dim=1)


def goal_distance_progress(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Backward-compatible alias for incremental goal progress shaping."""
    return goal_progress_delta(env=env, goal_x=goal_x, start_x=start_x, asset_cfg=asset_cfg)


def goal_progress_delta(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    normalize_by_goal: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward per-step progress toward the goal line in the env x-frame.

    This term is stateful on the environment instance so the reward is based on
    *incremental* progress instead of absolute x-position. It can be annealed to
    zero once the policy reliably reaches the goal. By default the progress is
    normalized by the goal distance; the baseline can opt into raw meter deltas.
    """
    asset = env.scene[asset_cfg.name]
    if start_x is None:
        progress_start_x = env.scene.env_origins[:, 0]
    else:
        progress_start_x = torch.full_like(asset.data.root_pos_w[:, 0], float(start_x))
    current_progress = asset.data.root_pos_w[:, 0] - progress_start_x
    if normalize_by_goal:
        current_progress = current_progress / max(goal_x, 1e-6)
        current_progress = torch.clamp(current_progress, min=0.0, max=1.5)

    prev_progress = getattr(env, "_goal_progress_prev", None)
    if prev_progress is None or prev_progress.shape != current_progress.shape:
        prev_progress = current_progress.clone()

    delta_progress = current_progress - prev_progress
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        delta_progress = torch.where(reset_mask, torch.zeros_like(delta_progress), delta_progress)

    env._goal_progress_prev = current_progress.clone()
    if normalize_by_goal:
        return torch.clamp(delta_progress, min=-0.25, max=0.25)
    return torch.clamp(delta_progress, min=-0.05, max=0.05)


def gated_goal_progress_delta(
    env: ManagerBasedRLEnv,
    goal_x: float,
    min_height: float,
    safe_height: float,
    start_x: float | None = None,
    normalize_by_goal: bool = True,
    min_upright: float = 0.55,
    safe_upright: float = 0.9,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Reward progress only while the robot remains upright and out of torso contact.

    This blocks the common exploit where the policy pitches the torso forward to
    generate transient +x root velocity or displacement during a fall.
    """
    progress = goal_progress_delta(
        env=env,
        goal_x=goal_x,
        start_x=start_x,
        normalize_by_goal=normalize_by_goal,
        asset_cfg=asset_cfg,
    )
    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    return progress * alive_gate


def gated_feet_air_time_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.55,
    safe_upright: float = 0.9,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
    torso_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Bootstrap alternating steps without paying for collapse-driven motion.

    This stays intentionally small. It should only help the policy discover
    stepping, not become the main objective.
    """
    air_time = locomotion_mdp.feet_air_time_positive_biped(
        env=env,
        command_name=command_name,
        threshold=threshold,
        sensor_cfg=sensor_cfg,
    )
    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=torso_sensor_cfg,
    )
    return air_time * alive_gate


def gated_feet_step_transition_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.35,
    safe_upright: float = 0.75,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
    torso_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Reward early foot lift-and-touchdown transitions instead of full single-stance gait.

    Unlike the stricter biped air-time reward, this turns on as soon as a foot
    leaves the ground long enough and touches back down. Negative values are
    clamped away so the term only bootstraps discovered steps.
    """
    step_transition = locomotion_mdp.feet_air_time(
        env=env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        threshold=threshold,
    )
    step_transition = torch.clamp(step_transition, min=0.0)
    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=torso_sensor_cfg,
    )
    return step_transition * alive_gate


def gated_single_swing_step_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_air_time: float,
    target_air_time: float,
    min_stance_time: float,
    min_forward_speed: float,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.35,
    safe_upright: float = 0.75,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
    torso_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Reward a clean single-foot swing instead of tiny two-foot shuffles.

    The term only turns on when exactly one foot is airborne, the stance foot has
    remained planted long enough, and the robot is moving forward while upright.
    That makes in-place rocking much less profitable than a genuine step.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    detached = air_time > 0.0
    single_swing = torch.sum(detached.int(), dim=1) == 1

    swing_air_time = torch.sum(torch.where(detached, air_time, 0.0), dim=1)
    stance_contact_time = torch.max(torch.where(~detached, contact_time, 0.0), dim=1).values

    swing_progress = torch.clamp(
        (swing_air_time - min_air_time) / max(target_air_time - min_air_time, 1.0e-6), min=0.0, max=1.0
    )
    stance_progress = torch.clamp(stance_contact_time / max(min_stance_time, 1.0e-6), min=0.0, max=1.0)
    forward_progress = torch.clamp(
        asset.data.root_lin_vel_b[:, 0] / max(min_forward_speed, 1.0e-6), min=0.0, max=1.0
    )

    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=torso_sensor_cfg,
    )
    moving_command = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return single_swing.float() * swing_progress * stance_progress * forward_progress * alive_gate * moving_command


def gated_early_step_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_air_time: float,
    target_air_time: float,
    min_stance_time: float,
    min_forward_speed: float,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.35,
    safe_upright: float = 0.75,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
    torso_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Reward the first clean foot-unload event before a full swing gait exists.

    This is intentionally easier than `gated_single_swing_step_reward`: it pays as
    soon as one foot lifts briefly while the other has remained planted and the
    robot is still moving forward in an upright posture.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    unload_progress_per_foot = torch.clamp(
        (air_time - min_air_time) / max(target_air_time - min_air_time, 1.0e-6), min=0.0, max=1.0
    )
    stance_progress_per_foot = torch.clamp(contact_time / max(min_stance_time, 1.0e-6), min=0.0, max=1.0)

    unload_progress, unload_idx = torch.max(unload_progress_per_foot, dim=1)
    other_idx = 1 - unload_idx
    stance_progress = stance_progress_per_foot.gather(1, other_idx.unsqueeze(1)).squeeze(1)
    unload_asymmetry = torch.abs(unload_progress_per_foot[:, 0] - unload_progress_per_foot[:, 1])
    forward_progress = torch.clamp(
        asset.data.root_lin_vel_b[:, 0] / max(min_forward_speed, 1.0e-6), min=0.0, max=1.0
    )

    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=torso_sensor_cfg,
    )
    moving_command = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05).float()
    return unload_progress * stance_progress * unload_asymmetry * forward_progress * alive_gate * moving_command


def gated_forward_speed(
    env: ManagerBasedRLEnv,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.35,
    safe_upright: float = 0.8,
    contact_force_threshold: float = 1.0,
    speed_cap: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Small bootstrap reward for upright forward motion in the robot frame.

    Using body-frame forward speed avoids paying for world-frame fall kinematics.
    The alive gate ensures it is not earned during torso-collapse.
    """
    asset = env.scene[asset_cfg.name]
    forward_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=speed_cap)
    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    return forward_speed * alive_gate


def gated_track_lin_vel_xy_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.3,
    safe_upright: float = 0.75,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Small command-tracking bootstrap gated on being alive and upright."""
    track_reward = locomotion_mdp.track_lin_vel_xy_yaw_frame_exp(
        env=env,
        command_name=command_name,
        std=std,
        asset_cfg=asset_cfg,
    )
    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    return track_reward * alive_gate


def stand_up_height_reward(
    env: ManagerBasedRLEnv,
    min_height: float,
    target_height: float,
    min_upright: float = 0.3,
    safe_upright: float = 0.75,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Reward recovering into a walking-height posture without paying for collapse.

    Unlike the low-height penalty, this is a positive shaped reward that helps the
    policy stand back up before a full gait exists.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    height_progress = torch.clamp(
        (asset.data.root_pos_w[:, 2] - min_height) / max(target_height - min_height, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    uprightness = -asset.data.projected_gravity_b[:, 2]
    upright_progress = torch.clamp(
        (uprightness - min_upright) / max(safe_upright - min_upright, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    net_contact_forces = contact_sensor.data.net_forces_w_history
    torso_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    torso_contact = torch.any(torso_contact > contact_force_threshold, dim=1)
    contact_gate = (~torso_contact).float()

    return height_progress * upright_progress * contact_gate


def stance_leg_extension_reward(
    env: ManagerBasedRLEnv,
    target_hip_pitch: float,
    target_knee: float,
    target_ankle: float,
    posture_sigma: float,
    stance_contact_time: float,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.3,
    safe_upright: float = 0.75,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hip_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_hip_pitch"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_knee"),
    ankle_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_ankle"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
    torso_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Reward keeping at least one planted leg close to the nominal standing posture.

    The curriculum already found stepping. The remaining failure mode is a crouched
    low-height posture that survives without torso contact. This term biases the
    supporting leg back toward the default H1 stance without making both legs stay
    frozen at the target during swing.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    hip_pos = asset.data.joint_pos[:, hip_cfg.joint_ids]
    knee_pos = asset.data.joint_pos[:, knee_cfg.joint_ids]
    ankle_pos = asset.data.joint_pos[:, ankle_cfg.joint_ids]

    hip_error = torch.square(hip_pos - target_hip_pitch)
    knee_error = torch.square(knee_pos - target_knee)
    ankle_error = torch.square(ankle_pos - target_ankle)
    per_leg_error = hip_error + knee_error + ankle_error
    per_leg_score = torch.exp(-per_leg_error / max(posture_sigma * posture_sigma, 1.0e-6))

    contact_progress = torch.clamp(
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] / max(stance_contact_time, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    supporting_leg_score = torch.max(per_leg_score * contact_progress, dim=1).values

    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=torso_sensor_cfg,
    )
    return supporting_leg_score * alive_gate


def upright_survival_reward(
    env: ManagerBasedRLEnv,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.3,
    safe_upright: float = 0.75,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Tiny alive bonus so early training prefers staying upright over instant collapse."""
    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    step_dt = float(getattr(env, "step_dt", 1.0))
    return alive_gate * step_dt


def upright_alive_gate(
    env: ManagerBasedRLEnv,
    min_height: float,
    safe_height: float,
    min_upright: float = 0.55,
    safe_upright: float = 0.9,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Soft gate for dense rewards while the robot is upright and not in torso contact."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    base_height = asset.data.root_pos_w[:, 2]
    height_span = max(safe_height - min_height, 1.0e-6)
    height_gate = torch.clamp((base_height - min_height) / height_span, min=0.0, max=1.0)

    uprightness = -asset.data.projected_gravity_b[:, 2]
    upright_span = max(safe_upright - min_upright, 1.0e-6)
    upright_gate = torch.clamp((uprightness - min_upright) / upright_span, min=0.0, max=1.0)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    torso_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    torso_contact = torch.any(torso_contact > contact_force_threshold, dim=1)
    contact_gate = (~torso_contact).float()

    return height_gate * upright_gate * contact_gate


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    bonus: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-step bonus whenever robot has crossed the goal x-line."""
    asset = env.scene[asset_cfg.name]
    if start_x is None:
        goal_line_x = env.scene.env_origins[:, 0] + goal_x
    else:
        goal_line_x = torch.full_like(asset.data.root_pos_w[:, 0], float(start_x + goal_x))
    reached = asset.data.root_pos_w[:, 0] >= goal_line_x
    return reached.float() * bonus


def base_height_penalty(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize low base height before terminal fall happens."""
    asset = env.scene[asset_cfg.name]
    deficit = torch.clamp(min_height - asset.data.root_pos_w[:, 2], min=0.0)
    return deficit


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the control-step duration so a unit negative weight becomes a time cost."""
    step_dt = float(getattr(env, "step_dt", 1.0))
    return torch.full((env.num_envs,), step_dt, device=env.device, dtype=torch.float32)


def completion_time_metric(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Zero-reward metric: records elapsed seconds for each env that crosses the goal this step.

    Writes a fresh list to env.extras["completion_times_s"] every step so the play script
    can read and accumulate per-episode completion times without reward-manager normalization.
    Must use a non-zero weight so the reward manager does not skip the call.
    """
    asset = env.scene[asset_cfg.name]
    if start_x is None:
        goal_line_x = env.scene.env_origins[:, 0] + goal_x
    else:
        goal_line_x = torch.full_like(asset.data.root_pos_w[:, 0], float(start_x + goal_x))

    reached = asset.data.root_pos_w[:, 0] >= goal_line_x
    step_dt = float(getattr(env, "step_dt", 1.0))
    # Reset every step so stale values from previous steps are never re-read.
    env.extras["completion_times_s"] = (
        (env.episode_length_buf[reached].float() * step_dt).tolist() if reached.any() else []
    )
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def time_remaining_goal_bonus(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    base_bonus: float = 200.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-step bonus on crossing the goal line, scaled by remaining episode time fraction.

    Pays base_bonus at step 1 and 0 at the last step, creating a direct gradient toward
    reaching the goal faster rather than only penalizing slow attempts via time_cost.
    """
    asset = env.scene[asset_cfg.name]
    if start_x is None:
        goal_line_x = env.scene.env_origins[:, 0] + goal_x
    else:
        goal_line_x = torch.full_like(asset.data.root_pos_w[:, 0], float(start_x + goal_x))
    reached = asset.data.root_pos_w[:, 0] >= goal_line_x

    max_steps = float(getattr(env, "max_episode_length", 1))
    current_steps = env.episode_length_buf.float()
    time_remaining_fraction = torch.clamp(1.0 - current_steps / max_steps, min=0.0, max=1.0)
    return reached.float() * base_bonus * time_remaining_fraction


def speed_gated_goal_progress_delta(
    env: ManagerBasedRLEnv,
    goal_x: float,
    min_forward_speed: float,
    start_x: float | None = None,
    normalize_by_goal: bool = True,
    min_height: float = 0.42,
    safe_height: float = 0.70,
    min_upright: float = 0.30,
    safe_upright: float = 0.80,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Goal progress gated on uprightness AND minimum forward body-frame speed.

    Blocks both fall-forward (not upright) and slow-shuffle (below min_forward_speed)
    exploits, pushing the policy toward genuinely fast upright locomotion.
    """
    progress = goal_progress_delta(
        env=env,
        goal_x=goal_x,
        start_x=start_x,
        normalize_by_goal=normalize_by_goal,
        asset_cfg=asset_cfg,
    )
    alive_gate = upright_alive_gate(
        env=env,
        min_height=min_height,
        safe_height=safe_height,
        min_upright=min_upright,
        safe_upright=safe_upright,
        contact_force_threshold=contact_force_threshold,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    asset = env.scene[asset_cfg.name]
    speed_gate = torch.clamp(
        asset.data.root_lin_vel_b[:, 0] / max(min_forward_speed, 1.0e-6), min=0.0, max=1.0
    )
    return progress * alive_gate * speed_gate


def zone_crossing_bonus(
    env: ManagerBasedRLEnv,
    zone_positions: tuple[float, ...],
    sigma: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense local bonus near obstacle-zone lines in x-direction."""
    asset = env.scene[asset_cfg.name]
    env_x = env.scene.env_origins[:, 0].unsqueeze(1)
    robot_x = asset.data.root_pos_w[:, 0].unsqueeze(1)
    thresholds = env_x + torch.tensor(zone_positions, device=robot_x.device).unsqueeze(0)
    return torch.exp(-((robot_x - thresholds) ** 2) / (2.0 * sigma**2)).sum(dim=1)
