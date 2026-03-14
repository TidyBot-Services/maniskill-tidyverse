# ManiSkill TidyVerse Robot

TidyVerse robot agent for [ManiSkill3](https://github.com/haosulab/ManiSkill) — a Franka Panda arm on a TidyBot mobile base with a Robotiq 85 gripper. Matches the real [TidyBot](https://tidybot.cs.princeton.edu/) hardware.

## Robot Specs

- **Arm:** Franka Panda 7-DOF
- **Gripper:** Robotiq 2F-85 (parallel jaw)
- **Base:** 3-DOF mobile base (x, y, yaw)
- **Total active joints:** 16 (3 base + 7 arm + 6 gripper)
- **EE link:** `eef`

## Install

```bash
pip install mani_skill==3.0.0b22 mplib==0.2.1 pycollada
```

## Setup

```bash
git clone https://github.com/shaoyifei96/maniskill-tidyverse
cd maniskill-tidyverse

# Create symlinks (required — URDF mesh paths are relative)
ln -sf $(python3 -c "import mani_skill; print(mani_skill.__path__[0])")/assets/robots/panda/franka_description franka_description
ln -sf ~/.maniskill/data/robots/robotiq_2f/meshes robotiq_meshes
```

## Quick Start

```python
import sys
sys.path.insert(0, '/path/to/maniskill-tidyverse')
import tidyverse_agent  # registers 'tidyverse' robot via @register_agent()
import mani_skill.envs
import gymnasium as gym

# With GUI
env = gym.make('PickCube-v1', render_mode='human', num_envs=1,
               robot_uids='tidyverse', control_mode='pd_ee_delta_pose')
obs, info = env.reset(seed=0)
```

Works with any ManiSkill3 environment (`PickCube-v1`, `RoboCasaKitchen-v1`, etc.).

## Control Modes

All modes use action order: **`[arm, gripper, base]`**

| Mode | Dims | Action Format |
|------|------|---------------|
| `pd_ee_delta_pose` | 10 | `[dx,dy,dz, dax,day,daz, gripper(1), base_vx,base_vy,base_vyaw]` |
| `pd_ee_pose` | 10 | `[x,y,z, ax,ay,az, gripper(1), base_vx,base_vy,base_vyaw]` |
| `pd_joint_pos` | 11 | `[arm_j1-j7(7), gripper(1), base_vx,base_vy,base_vyaw(3)]` |
| `pd_joint_delta_pos` | 11 | `[Δarm_j1-Δj7(7), Δgripper(1), base_vx,base_vy,base_vyaw(3)]` |
| **`whole_body`** | 11 | `[arm_j1-j7(7), gripper(1), base_x,base_y,base_yaw(3)]` |

> **Note:** In `pd_joint_pos` and `pd_joint_delta_pos`, the base uses **velocity** control. In `whole_body`, the base uses **position** control — required for motion planning.

## Motion Planning (mplib 0.2.1)

Full whole-body 10-DOF motion planning (3 base + 7 arm) with sub-millimeter accuracy.

### Whole-Body Planning (base + arm)

```python
import mplib
from mplib.pymp import Pose as MPPose

# Use whole_body mode — base is position-controlled
env = gym.make('PickCube-v1', render_mode='human', num_envs=1,
               robot_uids='tidyverse', control_mode='whole_body')
obs, info = env.reset(seed=0)
agent = env.unwrapped.agent
robot = agent.robot

# Planning URDF: box collisions, no visual meshes (fast loading)
planner = mplib.Planner(
    urdf='tidyverse_bare.urdf',
    srdf='tidyverse_bare_mplib.srdf',  # auto-generated on first run
    move_group='eef',
)

# CRITICAL: sync mplib frame with ManiSkill robot placement
root_p = robot.pose.p[0].cpu().numpy()
root_q = robot.pose.q[0].cpu().numpy()
planner.robot.set_base_pose(MPPose(p=root_p, q=root_q))

# Plan
qpos = robot.get_qpos().cpu().numpy()[0]
target_pose = sapien.Pose(p=[0.5, 0.3, 1.0], q=[1, 0, 0, 0])
result = planner.plan_pose(target_pose, qpos, time_step=env.unwrapped.control_timestep)

if result['status'] == 'Success':
    traj = result['position']  # (N, 10): [base_x, base_y, base_yaw, j1-j7]
    gripper_val = qpos[10]
    for i in range(traj.shape[0]):
        # Map: mplib [base(3), arm(7)] → action [arm(7), gripper(1), base(3)]
        action = np.concatenate([traj[i, 3:10], [gripper_val], traj[i, 0:3]])
        env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
```

### Arm-Only Planning (fixed base)

```python
# Use the arm-only planning URDF (base joints fixed)
planner = mplib.Planner(
    urdf='tidyverse_arm_planning.urdf',
    srdf='tidyverse_arm_planning_mplib.srdf',
    move_group='eef',
)

# Set base pose to base_link world position
links = {l.get_name(): l for l in robot.get_links()}
planner.robot.set_base_pose(MPPose(
    p=links['base_link'].pose.p[0].cpu().numpy(),
    q=links['base_link'].pose.q[0].cpu().numpy()))

# Planner expects [arm(7), gripper(6)] = 13 joints
arm_qpos = qpos[3:10]
gripper_qpos = qpos[10:]
planner_qpos = np.concatenate([arm_qpos, gripper_qpos])

result = planner.plan_pose(target_pose, planner_qpos, time_step=env.unwrapped.control_timestep)
# traj shape: (N, 7) — arm joints only
# Action: [arm(7), gripper(1), base_vel=0,0,0]
```

### Accuracy Results

| Mode | Error |
|------|-------|
| Arm-only | < 1mm |
| Whole-body (small reach) | 0.0mm |
| Whole-body (50cm reach) | 0.1mm |
| Whole-body (return) | 0.2mm |

## Planning URDFs

The main `tidyverse.urdf` uses DAE meshes that cause mplib to hang. Use these planning-specific URDFs instead:

| File | Use Case |
|------|----------|
| `tidyverse_bare.urdf` | Whole-body planning — box collisions, base joints active |
| `tidyverse_arm_planning.urdf` | Arm-only planning — base joints fixed |
| `tidyverse_bare_mplib.srdf` | Auto-generated SRDF for whole-body |
| `tidyverse_arm_planning_mplib.srdf` | Auto-generated SRDF for arm-only |

## RoboCasa Kitchens

120 kitchen configurations (10 layouts × 12 styles) via `RoboCasaKitchen-v1`:

```python
env = gym.make('RoboCasaKitchen-v1', num_envs=1,
               robot_uids='tidyverse', control_mode='whole_body')
obs, info = env.reset(seed=42)
```

153 object categories available. See [object spawning example](https://github.com/shaoyifei96/maniskill-tidyverse/wiki) for placing objects on counter surfaces.

## Known Limitations

- `RoboCasaKitchen-v1` is a scene viewer — no task definitions or `_check_success()`
- Fixture interaction (`is_open()`/`set_door_state()`) are stubs
- ManiSkill warns `"tidyverse is not in the task's list of supported robots"` — safe to ignore
- Whole-body planner may choose excessive base yaw rotations (cosmetic — TCP accuracy unaffected)
- mplib hangs on DAE collision meshes — always use `tidyverse_bare.urdf` for planning

## File Structure

```
maniskill-tidyverse/
├── tidyverse_agent.py              # Agent class, registered as 'tidyverse'
├── tidyverse.urdf                   # Full URDF (for ManiSkill rendering)
├── tidyverse_bare.urdf              # Planning URDF (box collisions, whole-body)
├── tidyverse_arm_planning.urdf      # Planning URDF (base fixed, arm-only)
├── tidyverse_bare_mplib.srdf        # Auto-generated SRDF
├── tidyverse_arm_planning_mplib.srdf
├── tidyverse_base/                  # Base meshes
├── franka_description/              # Symlink → Panda meshes
├── robotiq_meshes/                  # Symlink → Robotiq meshes
└── README.md
```

## License

MIT
