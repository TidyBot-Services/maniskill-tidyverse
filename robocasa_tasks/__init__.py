"""
RoboCasa v0.2 tasks ported to ManiSkill.

94 tasks (86 multi-stage + 8 single-stage) from RoboCasa v0.2,
each registered as a gymnasium environment.

Usage:
    from maniskill_tidyverse import robocasa_tasks  # auto-registers all envs
    env = gymnasium.make("RoboCasa-Prepare-Coffee-v0", robot_uids="tidyverse")
"""

from . import single_stage
from . import multi_stage
