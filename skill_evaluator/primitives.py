"""
Layer 2: Success Criteria Primitives for ManiSkill/SAPIEN.

These are the atomic building blocks for evaluating robot skill success.
Each function uses god-view sim data (actor poses, contacts) to answer
a simple True/False question about the physical state.

All functions operate on the ManiSkill env's unwrapped interface.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Primitive 1: object_grasped
# ---------------------------------------------------------------------------

def object_grasped(env, obj_name: str, max_finger_dist: float = 0.085) -> bool:
    """Is the robot's gripper currently grasping the named object?

    Checks two conditions:
    1. Gripper fingers are partially closed (not fully open)
    2. Object is between the finger pads (close to both)

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the actor in the scene
        max_finger_dist: Max distance from object to finger pad center (m)

    Returns:
        True if object appears to be grasped
    """
    actor = _get_actor(env, obj_name)
    if actor is None:
        return False

    robot = env.agent.robot
    obj_pos = actor.pose.p[0].cpu().numpy()

    # Get finger pad positions
    left_pad = robot.links_map.get("left_inner_finger_pad")
    right_pad = robot.links_map.get("right_inner_finger_pad")
    if left_pad is None or right_pad is None:
        return False

    left_pos = left_pad.pose.p[0].cpu().numpy()
    right_pos = right_pad.pose.p[0].cpu().numpy()

    # Check gripper is not fully open (qpos > threshold)
    qpos = robot.get_qpos()[0].cpu().numpy()
    gripper_q = float(qpos[10])  # left_outer_knuckle_joint
    if gripper_q < 0.01:  # fully open
        return False

    # Check object is close to both finger pads
    dist_left = np.linalg.norm(obj_pos - left_pos)
    dist_right = np.linalg.norm(obj_pos - right_pos)

    return dist_left < max_finger_dist and dist_right < max_finger_dist


# ---------------------------------------------------------------------------
# Primitive 2: object_lifted
# ---------------------------------------------------------------------------

def object_lifted(env, obj_name: str, min_height: float = 0.05,
                  reference: str = "initial") -> bool:
    """Has the object been lifted above a reference height?

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the actor
        min_height: Minimum lift distance (m) above reference
        reference: "initial" uses the object's reset z, "absolute" uses min_height as z

    Returns:
        True if object z is above reference + min_height
    """
    actor = _get_actor(env, obj_name)
    if actor is None:
        return False

    obj_z = float(actor.pose.p[0, 2].cpu())

    if reference == "absolute":
        return obj_z > min_height
    else:
        # Use initial pose stored in object_actors
        initial_z = _get_initial_z(env, obj_name)
        if initial_z is None:
            # No initial z recorded — cannot determine if lifted
            return False
        return obj_z > initial_z + min_height


# ---------------------------------------------------------------------------
# Primitive 3: object_on_target
# ---------------------------------------------------------------------------

def object_on_target(env, obj_name: str, target_name: str,
                     xy_threshold: float = 0.15,
                     z_threshold: float = 0.15) -> bool:
    """Is the object positioned on top of the target?

    Checks:
    1. XY distance between object and target is small
    2. Object z is above target z (sitting on top, not below)
    3. Object z is not too far above target (not floating)

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the object actor
        target_name: Name of the target actor (receptacle/surface)
        xy_threshold: Max horizontal distance (m)
        z_threshold: Max vertical distance above target (m)

    Returns:
        True if object is on the target
    """
    obj_actor = _get_actor(env, obj_name)
    tgt_actor = _get_actor(env, target_name)
    if obj_actor is None or tgt_actor is None:
        return False

    obj_pos = obj_actor.pose.p[0].cpu().numpy()
    tgt_pos = tgt_actor.pose.p[0].cpu().numpy()

    xy_dist = np.linalg.norm(obj_pos[:2] - tgt_pos[:2])
    z_diff = obj_pos[2] - tgt_pos[2]

    return xy_dist < xy_threshold and 0 < z_diff < z_threshold


# ---------------------------------------------------------------------------
# Primitive 4: gripper_released
# ---------------------------------------------------------------------------

def gripper_released(env, open_threshold: float = 0.05) -> bool:
    """Is the gripper in an open/released state?

    Args:
        env: Unwrapped ManiSkill env
        open_threshold: Max gripper joint value to consider "open" (0=fully open)

    Returns:
        True if gripper is open
    """
    robot = env.agent.robot
    qpos = robot.get_qpos()[0].cpu().numpy()
    gripper_q = float(qpos[10])  # left_outer_knuckle_joint
    return gripper_q < open_threshold





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_actor(env, name: str):
    """Find an actor by name in scene.actors or object_actors."""
    # Try scene actors first
    scene = env.scene
    if name in scene.actors:
        return scene.actors[name]

    # Try object_actors (manipulable objects)
    if hasattr(env, 'object_actors'):
        for scene_objs in env.object_actors:
            if name in scene_objs:
                return scene_objs[name].get('actor')

    # Fuzzy match: look for actors containing the name
    for actor_name, actor in scene.actors.items():
        if name in actor_name:
            return actor

    return None


def _get_initial_z(env, obj_name: str) -> Optional[float]:
    """Get the initial z position of an object from object_actors."""
    if hasattr(env, 'object_actors'):
        for scene_objs in env.object_actors:
            if obj_name in scene_objs:
                pose = scene_objs[obj_name].get('pose')
                if pose is not None:
                    return float(pose.p[2]) if hasattr(pose, 'p') else None
    return None





# ---------------------------------------------------------------------------
# Compound helpers (useful combinations)
# ---------------------------------------------------------------------------

def object_picked_up(env, obj_name: str, min_lift: float = 0.05) -> bool:
    """Compound: object is both grasped AND lifted."""
    return object_grasped(env, obj_name) and object_lifted(env, obj_name, min_lift)


def object_placed(env, obj_name: str, target_name: str) -> bool:
    """Compound: object is on target AND gripper is released."""
    return object_on_target(env, obj_name, target_name) and gripper_released(env)


def gripper_far_from(env, obj_name: str, min_dist: float = 0.25) -> bool:
    """Is the gripper EEF far from the object? (post-place check)"""
    actor = _get_actor(env, obj_name)
    if actor is None:
        return True  # can't find object = consider it far

    robot = env.agent.robot
    eef = robot.links_map["eef"]
    eef_pos = eef.pose.p[0].cpu().numpy()
    obj_pos = actor.pose.p[0].cpu().numpy()

    return np.linalg.norm(eef_pos - obj_pos) > min_dist
