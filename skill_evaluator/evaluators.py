"""
Layer 3: Skill-Specific Evaluators for ManiSkill.

Each evaluator maps to a published Tidybot-Skills skill and returns a dict
with 'success' (bool) plus sub-criteria for debugging.

Usage:
    from skill_evaluator import evaluate_pick_up_object
    result = evaluate_pick_up_object(env.unwrapped, "cup")
    # {'success': True, 'grasped': True, 'lifted': True}
"""

from .primitives import (
    object_grasped, object_lifted, object_on_target,
    object_in_camera_center, object_visible,
    gripper_released, gripper_far_from,
    object_picked_up, object_placed,
)


# ---------------------------------------------------------------------------
# Skill 1: tb-center-object
# ---------------------------------------------------------------------------

def evaluate_center_object(env, obj_name: str,
                           camera_uid: str = "wrist_camera",
                           tolerance: float = 0.15) -> dict:
    """Evaluate tb-center-object skill.

    Success: target object is centered in the wrist camera view.

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the object to center
        camera_uid: Camera to check centering in
        tolerance: Fraction of image for center tolerance (0.15 = 15%)

    Returns:
        dict with 'success' and sub-criteria
    """
    visible = object_visible(env, obj_name, camera_uid)
    centered = object_in_camera_center(env, obj_name, camera_uid, tolerance)

    return {
        "success": centered,
        "visible": visible,
        "centered": centered,
    }


# ---------------------------------------------------------------------------
# Skill 2: tb-pick-up-object
# ---------------------------------------------------------------------------

def evaluate_pick_up_object(env, obj_name: str,
                            min_lift: float = 0.05) -> dict:
    """Evaluate tb-pick-up-object skill.

    Success: object is grasped by the gripper AND lifted above its initial height.

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the object to pick up
        min_lift: Minimum lift height above initial position (m)

    Returns:
        dict with 'success' and sub-criteria
    """
    grasped = object_grasped(env, obj_name)
    lifted = object_lifted(env, obj_name, min_lift)

    return {
        "success": grasped and lifted,
        "grasped": grasped,
        "lifted": lifted,
    }


# ---------------------------------------------------------------------------
# Skill 3: tb-place-object
# ---------------------------------------------------------------------------

def evaluate_place_object(env, obj_name: str, target_name: str,
                          xy_threshold: float = 0.15) -> dict:
    """Evaluate tb-place-object skill.

    Success: object is on the target AND gripper has released it AND
    gripper is away from the object.

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the object being placed
        target_name: Name of the target receptacle/surface
        xy_threshold: Max horizontal distance for "on target" (m)

    Returns:
        dict with 'success' and sub-criteria
    """
    on_target = object_on_target(env, obj_name, target_name, xy_threshold)
    released = gripper_released(env)
    far = gripper_far_from(env, obj_name, min_dist=0.15)

    return {
        "success": on_target and released and far,
        "on_target": on_target,
        "released": released,
        "gripper_far": far,
    }


# ---------------------------------------------------------------------------
# Skill 4: tb-pick-and-place
# ---------------------------------------------------------------------------

def evaluate_pick_and_place(env, pick_obj: str, place_target: str,
                            xy_threshold: float = 0.15) -> dict:
    """Evaluate tb-pick-and-place skill.

    Success: object has been moved from its initial position to the target,
    gripper has released, and gripper is away.

    Args:
        env: Unwrapped ManiSkill env
        pick_obj: Name of the object that was picked
        place_target: Name of the target receptacle
        xy_threshold: Max horizontal distance for "on target"

    Returns:
        dict with 'success' and sub-criteria
    """
    on_target = object_on_target(env, pick_obj, place_target, xy_threshold)
    released = gripper_released(env)
    far = gripper_far_from(env, pick_obj, min_dist=0.15)

    return {
        "success": on_target and released and far,
        "on_target": on_target,
        "released": released,
        "gripper_far": far,
    }


# ---------------------------------------------------------------------------
# Skill 5: tb-find-and-pick-up
# ---------------------------------------------------------------------------

def evaluate_find_and_pick_up(env, obj_name: str,
                               min_lift: float = 0.05) -> dict:
    """Evaluate tb-find-and-pick-up skill.

    Success: same as pick-up-object (object grasped + lifted).
    The "find" part is implicit — if the object is grasped, it was found.

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the object
        min_lift: Minimum lift height (m)

    Returns:
        dict with 'success' and sub-criteria
    """
    grasped = object_grasped(env, obj_name)
    lifted = object_lifted(env, obj_name, min_lift)

    return {
        "success": grasped and lifted,
        "found": True,  # if grasped, was found; if not, still might have been found
        "grasped": grasped,
        "lifted": lifted,
    }


# ---------------------------------------------------------------------------
# Skill 6: tb-look-forward
# ---------------------------------------------------------------------------

def evaluate_look_forward(env, obj_name: str,
                          camera_uid: str = "wrist_camera") -> dict:
    """Evaluate tb-look-forward skill.

    Success: target object is visible in the wrist camera.
    This is a perception-only skill — success means the camera can see the object.

    Args:
        env: Unwrapped ManiSkill env
        obj_name: Name of the object to detect
        camera_uid: Camera to check visibility in

    Returns:
        dict with 'success' and sub-criteria
    """
    visible = object_visible(env, obj_name, camera_uid)
    centered = object_in_camera_center(env, obj_name, camera_uid, tolerance_ratio=0.4)

    return {
        "success": visible,
        "visible": visible,
        "roughly_centered": centered,
    }


# ---------------------------------------------------------------------------
# Registry: skill name → evaluator function
# ---------------------------------------------------------------------------

SKILL_EVALUATORS = {
    "tb-center-object": evaluate_center_object,
    "tb-pick-up-object": evaluate_pick_up_object,
    "tb-place-object": evaluate_place_object,
    "tb-pick-and-place": evaluate_pick_and_place,
    "tb-find-and-pick-up": evaluate_find_and_pick_up,
    "tb-look-forward": evaluate_look_forward,
}


def get_evaluator(skill_name: str):
    """Get the evaluator function for a skill by name."""
    return SKILL_EVALUATORS.get(skill_name)


def list_skills():
    """List all skills with evaluators."""
    return list(SKILL_EVALUATORS.keys())
