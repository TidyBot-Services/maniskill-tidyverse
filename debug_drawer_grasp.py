#!/usr/bin/env python3
"""Debug: teleport to drawer, try each grasp orientation, save wrist camera images."""
import sys, os, signal, time
import numpy as np
import torch, sapien, cv2
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tidyverse_agent, mani_skill.envs
try:
    import robocasa_tasks
except ImportError:
    pass
import planning_utils  # monkey-patch

from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib import Pose as MPPose
from mani_skill.utils import common
from scipy.spatial.transform import Rotation as Rot
from transforms3d.quaternions import qmult

from perception import find_handle_targets, perceive_by_seg_id
from planning_utils import add_fixture_boxes_to_planner, build_kitchen_acm, sync_planner
from execution import (
    ARM_HOME, GRIPPER_OPEN, MASK_ARM_ONLY, MASK_WHOLE_BODY,
    IK_TIMEOUT, PLANNING_TIMEOUT, Q_FLIP_Z,
    make_action, get_robot_qpos, wait_until_stable, execute_trajectory,
)

signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("timeout")))

OUT = os.path.join(os.path.dirname(__file__), 'debug_drawer_images')
os.makedirs(OUT, exist_ok=True)


def save_cam_image(obs, cam_name, filename, label=""):
    rgb = common.to_numpy(obs["sensor_data"][cam_name]["rgb"][0])
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if label:
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    path = os.path.join(OUT, filename)
    cv2.imwrite(path, img)
    print(f"  Saved: {path}")


def compute_front_grasp(fixture_yaw):
    rot = Rot.from_euler('yz', [np.pi / 2, fixture_yaw])
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


def compute_front_vert(fixture_yaw):
    rot = Rot.from_euler('yz', [np.pi / 2, fixture_yaw]) * Rot.from_euler('z', np.pi / 2)
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def main():
    env = gym.make('RoboCasaKitchen-v1', num_envs=1, robot_uids='tidyverse',
                   control_mode='whole_body', obs_mode='rgb+depth+segmentation',
                   render_mode='rgb_array', sensor_configs=dict(shader_pack="default"))
    obs, _ = env.reset(seed=0)
    robot = env.unwrapped.agent.robot
    scene = env.unwrapped.scene.sub_scenes[0]
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']

    def step_fn(action):
        return env.step(action)

    # Stabilize
    hold = make_action(ARM_HOME, GRIPPER_OPEN, get_robot_qpos(robot)[:3])
    wait_until_stable(lambda a: env.step(a), hold, robot)

    # Find drawers
    handle_targets = find_handle_targets(fixtures, env.unwrapped)
    print(f"Found {len(handle_targets)} handles")

    # Register seg ids
    try:
        del env.unwrapped.__dict__['segmentation_id_map']
    except KeyError:
        pass
    seg_map = env.unwrapped.segmentation_id_map
    for ht in handle_targets:
        seg_map[int(ht['link'].per_scene_id)] = ht['link']

    # Pick closest drawer to robot
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()

    # Pick stack_2_right_group_3 (mid-height, z~0.57)
    best = None
    for ht in handle_targets:
        if ht['fixture_name'] == 'stack_2_right_group_3':
            best = ht
            break
    if best is None:
        # Fallback: pick highest right-wall drawer
        for ht in sorted(handle_targets,
                         key=lambda h: h['link'].pose.p[0].cpu().numpy()[2],
                         reverse=True):
            if 'right' in ht['fixture_name']:
                best = ht
                break
    if best is None:
        print("No suitable drawers!")
        env.close()
        return

    ht = best
    fxt_yaw = ht['fixture_yaw']
    front_dir = ht['front_dir']
    print(f"\nTarget: {ht['fixture_name']} yaw={np.degrees(fxt_yaw):.0f}deg "
          f"front_dir={front_dir}")

    # Teleport base
    link_pos = ht['link'].pose.p[0].cpu().numpy()
    target_xy = link_pos[:2] + front_dir[:2] * 0.55
    target_yaw = np.arctan2(-front_dir[1], -front_dir[0])

    root_p = robot.pose.p[0].cpu().numpy()
    root_q = robot.pose.q[0].cpu().numpy()
    from scipy.spatial.transform import Rotation as R_root
    root_yaw = R_root.from_quat([root_q[1], root_q[2], root_q[3], root_q[0]]).as_euler('xyz')[2]

    dx = target_xy[0] - root_p[0]
    dy = target_xy[1] - root_p[1]
    cos_ry, sin_ry = np.cos(-root_yaw), np.sin(-root_yaw)

    qpos = get_robot_qpos(robot)
    qpos[0] = cos_ry * dx - sin_ry * dy
    qpos[1] = sin_ry * dx + cos_ry * dy
    qpos[2] = target_yaw - root_yaw
    qpos[3:10] = ARM_HOME
    qpos[10:] = 0.0
    robot.set_qpos(torch.tensor(qpos, dtype=torch.float32).unsqueeze(0))

    hold = make_action(ARM_HOME, GRIPPER_OPEN, qpos[:3])
    for _ in range(30):
        env.step(hold)
    wait_until_stable(lambda a: env.step(a), hold, robot, max_steps=60)

    print(f"Base at target_xy=[{target_xy[0]:.2f}, {target_xy[1]:.2f}], "
          f"yaw={np.degrees(target_yaw):.0f}deg")

    # Save initial view
    obs, _, _, _, _ = env.step(hold)
    save_cam_image(obs, 'base_camera', '00_base_initial.png', 'Base cam - initial')
    save_cam_image(obs, 'wrist_camera', '00_wrist_initial.png', 'Wrist cam - initial')

    # Re-perceive handle
    sid = int(ht['link'].per_scene_id)
    perc = perceive_by_seg_id(obs, sid, camera_name='base_camera')
    if perc:
        handle_pos = perc.center_3d
        print(f"Handle perceived at {handle_pos}")
    else:
        handle_pos = link_pos
        print(f"Handle NOT visible, using link pos {link_pos}")

    # Setup planner
    pw = SapienPlanningWorld(scene, [robot._objs[0]])
    eef = next(n for n in pw.get_planned_articulations()[0]
               .get_pinocchio_model().get_link_names() if 'eef' in n)
    planner = SapienPlanner(pw, move_group=eef)

    # Add fixture boxes, skip all drawers in the same stack
    stack_prefix = '_'.join(ht['fixture_name'].split('_')[:2])  # e.g. 'stack_2'
    skip_fxts = {fn for fn in fixtures if fn.startswith(stack_prefix)}
    print(f"Skipping fixture boxes for: {skip_fxts}")
    add_fixture_boxes_to_planner(pw, scene, fixtures, skip_fixtures=skip_fxts)
    build_kitchen_acm(pw, planner, set(), mode='strict',
                      robot_pos=arm_base, target_positions=[handle_pos])

    # Relax collision with target drawer
    acm = pw.get_allowed_collision_matrix()
    robot_links = planner.pinocchio_model.get_link_names()
    for al in ht['articulation'].get_links():
        for rl in robot_links:
            acm.set_entry(rl, al.get_name(), True)

    # Try each grasp orientation
    orientations = {
        'Front': compute_front_grasp(fxt_yaw),
        'FrontVert': compute_front_vert(fxt_yaw),
        'Front_flip': qmult(compute_front_grasp(fxt_yaw), Q_FLIP_Z),
        'FrontVert_flip': qmult(compute_front_vert(fxt_yaw), Q_FLIP_Z),
    }

    PRE_OFFSET = 0.14
    GRASP_OFFSET = 0.06

    for name, q_wxyz in orientations.items():
        print(f"\n--- {name} ---")
        pre_pos = handle_pos + front_dir * PRE_OFFSET
        grasp_pos = handle_pos + front_dir * GRASP_OFFSET

        # Reset arm to home first
        cq = get_robot_qpos(robot)
        cq[3:10] = ARM_HOME
        robot.set_qpos(torch.tensor(cq, dtype=torch.float32).unsqueeze(0))
        hold = make_action(ARM_HOME, GRIPPER_OPEN, cq[:3])
        for _ in range(20):
            env.step(hold)

        # IK for pre-grasp
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        pre_pose = MPPose(p=pre_pos, q=q_wxyz)
        pre_base = planner._transform_goal_to_wrt_base(pre_pose)

        sol = None
        for mask_name, mask in [("arm-only", MASK_ARM_ONLY), ("whole-body", MASK_WHOLE_BODY)]:
            signal.alarm(IK_TIMEOUT)
            try:
                status, solutions = planner.IK(pre_base, cq, mask=mask,
                                                n_init_qpos=40, return_closest=True)
            except TimeoutError:
                continue
            finally:
                signal.alarm(0)
            if solutions is not None:
                sol = solutions
                print(f"  Pre-grasp IK ({mask_name}): OK")
                break
            else:
                print(f"  Pre-grasp IK ({mask_name}): no solution")

        if sol is None:
            print(f"  SKIP — no IK solution")
            save_cam_image(obs, 'wrist_camera', f'{name}_SKIP.png', f'{name} - NO IK')
            continue

        # Plan and execute to pre-grasp
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        signal.alarm(PLANNING_TIMEOUT)
        try:
            result = planner.plan_qpos([sol], cq, planning_time=5.0)
        except TimeoutError:
            result = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)

        if result['status'] != 'Success':
            print(f"  Pre-grasp path: FAILED — {result['status']}")
            continue

        print(f"  Pre-grasp path: OK ({result['position'].shape[0]} wp)")
        execute_trajectory(result['position'], lambda a: env.step(a),
                           GRIPPER_OPEN, lock_base=True, robot=robot)

        # Settle and capture
        cq = get_robot_qpos(robot)
        hold = make_action(cq[3:10], GRIPPER_OPEN, cq[:3])
        for _ in range(20):
            env.step(hold)
        obs, _, _, _, _ = env.step(hold)

        save_cam_image(obs, 'wrist_camera', f'{name}_pregrasp_wrist.png',
                       f'{name} - pre-grasp (wrist)')
        save_cam_image(obs, 'base_camera', f'{name}_pregrasp_base.png',
                       f'{name} - pre-grasp (base)')

        # Also try approach with plan_screw
        grasp_pose = MPPose(p=grasp_pos, q=q_wxyz)
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_app = planner.plan_screw(grasp_pose, cq, time_step=0.05)
        except TimeoutError:
            r_app = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)
        print(f"  Approach (screw): {r_app['status']}")

        if r_app['status'] != 'Success':
            # Check collision at approach target
            sync_planner(planner)
            cq = get_robot_qpos(robot)
            grasp_base = planner._transform_goal_to_wrt_base(grasp_pose)
            signal.alarm(IK_TIMEOUT)
            try:
                s2, sol2 = planner.IK(grasp_base, cq, mask=MASK_ARM_ONLY,
                                       n_init_qpos=40, return_closest=True)
            except TimeoutError:
                sol2 = None
            finally:
                signal.alarm(0)
            if sol2 is not None:
                print(f"  Approach IK (arm-only): OK — path planning is the issue")
                # Check collision
                pw_collisions = pw.check_collision(sol2)
                print(f"  Collision check at approach: {pw_collisions}")
            else:
                signal.alarm(IK_TIMEOUT)
                try:
                    s3, sol3 = planner.IK(grasp_base, cq, mask=MASK_WHOLE_BODY,
                                           n_init_qpos=40, return_closest=True)
                except TimeoutError:
                    sol3 = None
                finally:
                    signal.alarm(0)
                if sol3 is not None:
                    print(f"  Approach IK (whole-body): OK — arm-only can't reach")
                else:
                    print(f"  Approach IK: NO SOLUTION for any mask")

        # Print EEF position for debugging
        eef_link = next(l for l in robot.get_links() if 'eef' in l.get_name())
        eef_pos = eef_link.pose.p[0].cpu().numpy()
        print(f"  EEF at pre-grasp: {eef_pos}")
        print(f"  Target grasp pos: {grasp_pos}")
        print(f"  Distance to grasp: {np.linalg.norm(eef_pos - grasp_pos):.4f}m")

    env.close()
    print(f"\nImages saved to {OUT}/")


if __name__ == '__main__':
    main()
