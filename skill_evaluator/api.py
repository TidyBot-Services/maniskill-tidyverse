"""
API integration: exposes skill evaluation via the ManiSkill server's command socket.

Adds two commands:
  - "evaluate_skill": Run a skill evaluator and return results
  - "get_objects":    List all scene actors with positions (god-view)

These are called by the agent_server or test scripts via ZMQ CMD socket.
"""

import numpy as np
from typing import Optional

from .evaluators import SKILL_EVALUATORS, list_skills
from . import primitives


def handle_evaluate_skill(server, skill_name: str, **kwargs) -> dict:
    """Run a skill evaluator on the current sim state.

    Args:
        server: ManiskillServer instance
        skill_name: e.g. "tb-pick-up-object"
        **kwargs: Arguments for the evaluator (obj_name, target_name, etc.)

    Returns:
        dict with 'success' and sub-criteria, or error
    """
    evaluator = SKILL_EVALUATORS.get(skill_name)
    if evaluator is None:
        return {"error": f"Unknown skill: {skill_name}",
                "available": list_skills()}

    env = server.env
    if env is None:
        return {"error": "No environment loaded"}

    uw = env.unwrapped
    try:
        result = evaluator(uw, **kwargs)
        return result
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def handle_get_objects(server) -> dict:
    """List all scene actors with their positions.

    Returns:
        dict with 'objects' list and 'robot' info
    """
    env = server.env
    if env is None:
        return {"error": "No environment loaded"}

    uw = env.unwrapped
    scene = uw.scene

    objects = []
    for name, actor in scene.actors.items():
        pos = actor.pose.p[0].cpu().numpy()
        objects.append({
            "name": name,
            "position": [round(float(x), 4) for x in pos],
        })

    # Add manipulable objects
    if hasattr(uw, 'object_actors'):
        for scene_idx, scene_objs in enumerate(uw.object_actors):
            for name, data in scene_objs.items():
                actor = data.get('actor')
                if actor is not None:
                    pos = actor.pose.p[0].cpu().numpy()
                    objects.append({
                        "name": name,
                        "position": [round(float(x), 4) for x in pos],
                        "manipulable": True,
                    })

    # Robot info
    robot = uw.agent.robot
    eef = robot.links_map["eef"]
    eef_pos = eef.pose.p[0].cpu().numpy()

    return {
        "objects": objects,
        "robot": {
            "eef_position": [round(float(x), 4) for x in eef_pos],
        },
        "total": len(objects),
    }
