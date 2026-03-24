#!/usr/bin/env python3
"""Perception-based drawer opening: detect drawer handles via camera,
grasp from the front, and pull open.

Usage:
    # GUI
    python test_drawer_open.py --render human --seed 0

    # Headless — save video
    python test_drawer_open.py --render rgb_array --seed 0
"""
import sys, os, signal, argparse, time
import numpy as np
import torch, sapien, cv2
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tidyverse_agent   # noqa: F401 — registers 'tidyverse'
import mani_skill.envs    # noqa: F401 — registers envs
try:
    import robocasa_tasks  # noqa: registers RoboCasa single-stage tasks
except ImportError:
    pass

# Apply monkey-patch for Robotiq scaled meshes
import planning_utils  # noqa: F401

from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib import Pose as MPPose
from mani_skill.utils import common
from scipy.spatial.transform import Rotation as Rot
from transforms3d.quaternions import qmult

from perception import find_handle_targets, perceive_by_seg_id, save_perception_debug
from planning_utils import (
    add_fixture_boxes_to_planner, build_kitchen_acm, sync_planner,
)
from execution import (
    ARM_HOME, GRIPPER_OPEN, GRIPPER_CLOSED,
    MASK_ARM_ONLY, MASK_WHOLE_BODY,
    PLANNING_TIMEOUT, IK_TIMEOUT, Q_FLIP_Z,
    make_action, get_robot_qpos, wait_until_stable,
    execute_trajectory, actuate_gripper,
)

# ─── Constants ────────────────────────────────────────────────────────────────

HANDLE_APPROACH_OFFSET = 0.0    # target at perceived center (handle bar)
PULL_DISTANCE = 0.15           # 15cm pull to open drawer


# ─── Video Writer ─────────────────────────────────────────────────────────────

class VideoWriter:
    def __init__(self, path, fps=30):
        self.path, self.fps, self.writer, self.frame_count = path, fps, None, 0

    def add_frame(self, frame):
        if frame.ndim == 4: frame = frame[0]
        h, w = frame.shape[:2]
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.path,
                                          cv2.VideoWriter_fourcc(*'mp4v'),
                                          self.fps, (w, h))
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self):
        if self.writer:
            self.writer.release()
            print(f"Video saved: {self.path} ({self.frame_count} frames)")


# ─── Grasp orientation for front-facing handle ───────────────────────────────

def _read_drawer_state(handle_info):
    """Read normalized drawer open state (0=closed, 1=fully open)."""
    art = handle_info['articulation']
    try:
        # Active joints only — find slide joint by iterating active DOFs
        active_joints = art.get_active_joints()
        qpos_art = art.get_qpos()
        for qi, j in enumerate(active_joints):
            if 'slide' in j.name.lower():
                q_val = float(qpos_art[0, qi])
                max_disp = getattr(handle_info['fixture'], 'max_displacement', 0.4)
                return q_val / max_disp
        return -1
    except Exception as e:
        print(f"    Could not read drawer state: {e}")
        return -1


def _euler_grasp(yaw):
    """Front grasp at given yaw: Ry(90°) + Rz(yaw). Returns wxyz."""
    q = Rot.from_euler('yz', [np.pi / 2, yaw]).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])



# ─── Attempt to open a single drawer ─────────────────────────────────────────

def attempt_drawer_open(handle_pos, handle_info, surface_normal, robot,
                        planner, pw, step_fn, timings, idx, total, env=None):
    """Open a drawer by grasping its handle and pulling.

    Pipeline:
      1. Solve grasp IK (whole-body moves base + arm)
      2. Plan & execute to grasp pose
      3. Close gripper
      4. Pull: retract along surface normal to open
      5. Release & return home
    """
    tag = f"[{idx+1}/{total}] {handle_info['fixture_name']}"
    # Pull direction = surface normal (away from drawer face)
    pull_dir = surface_normal / np.linalg.norm(surface_normal)

    # Check drawer state before
    state_before = _read_drawer_state(handle_info)
    print(f"  Drawer state BEFORE: {state_before:.2f}")

    # The whole-body planner will move the base as needed
    sync_planner(planner)

    # Relax collision between robot and target drawer + nearby objects
    acm = pw.get_allowed_collision_matrix()
    robot_link_names = planner.pinocchio_model.get_link_names()
    art = handle_info['articulation']
    for art_link in art.get_links():
        aln = art_link.get_name()
        for rl in robot_link_names:
            acm.set_entry(rl, aln, True)
    # Also relax objects near the handle (robot will drive through this area)
    for obj_name in pw.get_object_names():
        if obj_name in robot_link_names:
            continue
        obj_pose = pw.get_object(obj_name).pose
        obj_pos = np.array([obj_pose.p[0], obj_pose.p[1]])
        dist = np.linalg.norm(obj_pos - handle_pos[:2])
        if dist < 1.5:
            for rl in robot_link_names:
                acm.set_entry(rl, obj_name, True)
    print(f"  Relaxed collisions near handle")

    # Build grasp candidates from perceived surface normal
    approach = -surface_normal / np.linalg.norm(surface_normal)
    yaw = np.arctan2(approach[1], approach[0])
    q_front = _euler_grasp(yaw)
    q_90 = _euler_grasp(yaw + np.pi / 2)
    strategies = [("Front", q_front), ("Front90", q_90)]
    print(f"  Surface normal: {surface_normal}")
    print(f"  Pull direction: {pull_dir}")

    for strategy_name, target_q in strategies:
        print(f"\n  --- {tag} ({strategy_name}) ---")
        print(f"    Handle pos: {handle_pos}")

        grasp_pos = handle_pos + pull_dir * HANDLE_APPROACH_OFFSET

        # 1. Solve grasp IK (try original, then 180°-flipped)
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        grasp_sols = None
        grasp_mask = None
        used_q = target_q

        for flip_label, q_used in [("", target_q),
                                    ("flip180", qmult(target_q, Q_FLIP_Z))]:
            if grasp_sols is not None:
                break
            grasp_pose = MPPose(p=grasp_pos, q=q_used)
            grasp_base = planner._transform_goal_to_wrt_base(grasp_pose)
            for mask_name, mask, n_ik in [("arm-only", MASK_ARM_ONLY, 40),
                                          ("whole-body", MASK_WHOLE_BODY, 200)]:
                suffix = f"+{flip_label}" if flip_label else ""
                t0 = time.time()
                signal.alarm(IK_TIMEOUT)
                try:
                    status, solutions = planner.IK(
                        grasp_base, cq, mask=mask, n_init_qpos=n_ik,
                        return_closest=True)
                except TimeoutError:
                    dt = time.time() - t0
                    print(f"    Grasp IK ({mask_name}{suffix}): TIMEOUT  [{dt:.2f}s]")
                    timings['ik'] += dt
                    continue
                finally:
                    signal.alarm(0)
                dt = time.time() - t0
                timings['ik'] += dt
                if solutions is not None:
                    grasp_sols = solutions
                    grasp_mask = mask
                    used_q = q_used
                    print(f"    Grasp IK ({mask_name}{suffix}): OK  [{dt:.2f}s]")
                    break
                else:
                    print(f"    Grasp IK ({mask_name}{suffix}): no solution  [{dt:.2f}s]")

        if grasp_sols is None:
            print(f"    Grasp IK: FAILED for {strategy_name}")
            continue

        # 2. Plan path to grasp pose
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            result = planner.plan_qpos([grasp_sols], cq, planning_time=5.0)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Grasp path: TIMEOUT  [{dt:.2f}s]")
            timings['planning'] += dt
            continue
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if result['status'] != 'Success':
            print(f"    Grasp path: FAILED — {result['status']}  [{dt:.2f}s]")
            continue
        print(f"    Grasp path: OK ({result['position'].shape[0]} wp)  [{dt:.2f}s]")

        used_arm_only = bool(isinstance(grasp_mask, np.ndarray) and grasp_mask[0])
        motion_mask = MASK_ARM_ONLY if used_arm_only else MASK_WHOLE_BODY

        # Execute to grasp pose
        t0 = time.time()
        execute_trajectory(result['position'], step_fn, GRIPPER_OPEN,
                           lock_base=used_arm_only, robot=robot)
        timings['exec'] += time.time() - t0

        # 4. Close gripper on handle
        t0 = time.time()
        actuate_gripper(step_fn, robot, GRIPPER_CLOSED, n_steps=40)
        qpos = get_robot_qpos(robot)
        hold_closed = make_action(qpos[3:10], GRIPPER_CLOSED, qpos[:3])
        for _ in range(20):
            step_fn(hold_closed)
        timings['gripper'] += time.time() - t0
        gripper_q = get_robot_qpos(robot)[10:]
        eef_link = next(l for l in robot.get_links() if 'eef' in l.get_name())
        eef_pos = eef_link.pose.p[0].cpu().numpy()
        actual_qpos = get_robot_qpos(robot)
        planned_final = result['position'][-1]
        base_err = np.linalg.norm(actual_qpos[0:3] - planned_final[0:3])
        arm_err = np.max(np.abs(actual_qpos[3:10] - planned_final[3:10]))
        print(f"    Gripper closed: eef={eef_pos}")
        print(f"    Tracking: base_err={base_err:.4f}m arm_err={arm_err:.4f}rad")
        print(f"    Target was: {grasp_pos}")


        # 5. Pull: retract along surface normal to open the drawer
        # (collision already relaxed at start of attempt)
        pull_target_pos = grasp_pos + pull_dir * PULL_DISTANCE
        pull_pose = MPPose(p=pull_target_pos, q=used_q)
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_pull = planner.plan_screw(pull_pose, cq, time_step=0.05)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Pull (screw): TIMEOUT  [{dt:.2f}s]")
            timings['planning'] += dt
            r_pull = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_pull['status'] != 'Success':
            # Fallback: plan to a partially retracted arm config (no orientation constraint)
            print(f"    Pull (screw): {r_pull['status']}, trying qpos retract fallback...")
            sync_planner(planner)
            cq = get_robot_qpos(robot)
            retract_qpos = cq.copy()
            # Retract arm toward home while keeping base fixed
            alpha = 0.4  # blend 40% toward home
            retract_qpos[3:10] = cq[3:10] * (1 - alpha) + ARM_HOME * alpha
            t0 = time.time()
            signal.alarm(PLANNING_TIMEOUT)
            try:
                r_pull = planner.plan_qpos([retract_qpos], cq, planning_time=5.0)
            except TimeoutError:
                dt = time.time() - t0
                timings['planning'] += dt
                r_pull = {'status': 'TIMEOUT'}
            finally:
                signal.alarm(0)
            dt = time.time() - t0
            timings['planning'] += dt

        if r_pull['status'] == 'Success':
            print(f"    Pull: OK ({r_pull['position'].shape[0]} wp)  [{dt:.2f}s]")
            t0 = time.time()
            execute_trajectory(r_pull['position'], step_fn, GRIPPER_CLOSED,
                               lock_base=used_arm_only, robot=robot)
            timings['exec'] += time.time() - t0
        else:
            print(f"    Pull: FAILED — {r_pull['status']}  [{dt:.2f}s]")

        # Hold to let physics settle
        qpos = get_robot_qpos(robot)
        hold = make_action(qpos[3:10], GRIPPER_CLOSED, qpos[:3])
        for _ in range(30):
            step_fn(hold)

        # Check drawer state by reading joint qpos directly
        drawer_open = _read_drawer_state(handle_info)
        print(f"    Drawer state AFTER: {drawer_open:.2f}")

        # 6. Release gripper
        t0 = time.time()
        actuate_gripper(step_fn, robot, GRIPPER_OPEN, n_steps=30)
        timings['gripper'] += time.time() - t0

        # 7. Retract to clear handle
        retract_pos = pull_target_pos + pull_dir * 0.10
        retract_pose = MPPose(p=retract_pos, q=used_q)
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_ret = planner.plan_pose(retract_pose, cq, mask=motion_mask,
                                       planning_time=3.0)
        except TimeoutError:
            r_ret = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)
        if r_ret['status'] == 'Success':
            execute_trajectory(r_ret['position'], step_fn, GRIPPER_OPEN,
                               lock_base=used_arm_only, robot=robot)

        # 8. Return home
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        home_qpos = cq.copy()
        home_qpos[3:10] = ARM_HOME
        home_qpos[10:] = 0.0
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_home = planner.plan_qpos([home_qpos], cq, planning_time=5.0)
        except TimeoutError:
            r_home = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)

        if r_home['status'] == 'Success':
            execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN,
                               robot=robot)
        else:
            print(f"    Return home: FAILED, staying in place")

        wait_until_stable(step_fn,
                          make_action(ARM_HOME, GRIPPER_OPEN,
                                      get_robot_qpos(robot)[:3]),
                          robot, max_steps=100)

        if drawer_open > 0.3:
            return 'success'
        elif drawer_open > 0.05:
            return 'partial'
        else:
            return 'unreachable'

    return 'unreachable'


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Perception-based drawer opening")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', default='human', choices=['human', 'rgb_array'])
    parser.add_argument('--task', default='RoboCasaKitchen-v1')
    parser.add_argument('--max-drawers', type=int, default=None)
    parser.add_argument('--acm', default='strict', choices=['relaxed', 'strict'])
    parser.add_argument('--debug-dir', default=None)
    args = parser.parse_args()

    t_total = time.time()

    # --- Create environment ---
    print(f"Creating env: {args.task}...")
    t0 = time.time()
    env = gym.make(
        args.task,
        num_envs=1,
        robot_uids='tidyverse',
        control_mode='whole_body',
        obs_mode='rgb+depth+segmentation',
        render_mode=args.render,
        sensor_configs=dict(shader_pack="default"),
    )
    obs, info = env.reset(seed=args.seed)
    t_env = time.time() - t0
    print(f"  env setup: {t_env:.2f}s")

    robot = env.unwrapped.agent.robot
    scene = env.unwrapped.scene.sub_scenes[0]
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']
    is_human = (args.render == 'human')

    # Reposition render camera
    from mani_skill.utils import sapien_utils as _su
    _rpos = robot.pose.p[0].cpu().numpy()
    _cam_eye = [_rpos[0], _rpos[1] - 3.5, 3.5]
    _cam_target = [_rpos[0], _rpos[1] + 1.0, 0.8]
    _cam_pose = _su.look_at(_cam_eye, _cam_target)
    _p = _cam_pose.raw_pose[0].cpu().numpy()
    _sapien_pose = sapien.Pose(p=_p[:3], q=_p[3:])
    for cam in env.unwrapped._human_render_cameras.values():
        cam.camera.set_local_pose(_sapien_pose)

    # Video writer
    video_dir = os.path.expanduser('~/tidyverse_videos')
    os.makedirs(video_dir, exist_ok=True)
    video_writer = None
    if args.render == 'rgb_array':
        base = f'drawer_open_seed{args.seed}_acm{args.acm}'
        run = 0
        while os.path.exists(os.path.join(video_dir, f'{base}_run{run}.mp4')):
            run += 1
        video_path = os.path.join(video_dir, f'{base}_run{run}.mp4')
        video_writer = VideoWriter(video_path, fps=30)

    step_label = ["idle"]

    def _burn_label(frame, text):
        h = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, h / 600)
        thick = max(1, int(h / 300))
        (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(frame, (0, 0), (tw + 20, th_ + 16), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, th_ + 8), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    def step_fn(action):
        obs_step, _, _, _, _ = env.step(action)
        if is_human:
            env.render()
        elif video_writer is not None:
            frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            frame = frame.astype(np.uint8).copy()
            _burn_label(frame, step_label[0])
            video_writer.add_frame(frame)

    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    print(f"Robot arm base at {arm_base}")

    # --- Stabilize ---
    base_cmd = get_robot_qpos(robot)[:3].copy()
    hold = make_action(ARM_HOME, GRIPPER_OPEN, base_cmd)
    step_label[0] = "Stabilizing"
    print("\nStabilizing robot...")
    wait_until_stable(step_fn, hold, robot)

    # --- Discover drawer handles ---
    print("\nDiscovering drawer handles from fixtures...")
    handle_targets = find_handle_targets(fixtures, env.unwrapped)
    print(f"  Found {len(handle_targets)} handle links:")
    for ht in handle_targets:
        # Get link world position for sorting
        link_pos = ht['link'].pose.p[0].cpu().numpy()
        ht['link_world_pos'] = link_pos
        print(f"    {ht['fixture_name']} ({ht['fixture_type']}) — "
              f"link: {ht['link_name']}, pos: [{link_pos[0]:.2f}, {link_pos[1]:.2f}, {link_pos[2]:.2f}], "
              f"yaw: {np.degrees(ht['fixture_yaw']):.0f}deg")

    if not handle_targets:
        print("\nNo drawer handles found!")
        env.close()
        return

    # --- Perceive each handle via camera (per seg_id) ---
    # Force seg map refresh and register handle links
    try:
        del env.unwrapped.__dict__['segmentation_id_map']
    except KeyError:
        pass
    seg_map = env.unwrapped.segmentation_id_map
    for ht in handle_targets:
        sid = int(ht['link'].per_scene_id)
        seg_map[sid] = ht['link']

    obs, _, _, _, _ = env.step(hold)
    step_label[0] = "Perceiving handles"
    print("\nPerceiving handles via camera...")

    perceived_handles = []
    for ht in handle_targets:
        seg_id = int(ht['link'].per_scene_id)
        # Try base camera first, then wrist
        perc = None
        for cam_name in ['base_camera', 'wrist_camera']:
            perc = perceive_by_seg_id(obs, seg_id, camera_name=cam_name)
            if perc is not None:
                break
        if perc is not None and perc.surface_normal is not None:
            handle_pos = perc.center_3d
            normal = perc.surface_normal
            dist = np.linalg.norm(handle_pos - arm_base)
            print(f"    {ht['fixture_name']:30s} seg_id={seg_id:3d}  "
                  f"pos=[{handle_pos[0]:.3f}, {handle_pos[1]:.3f}, {handle_pos[2]:.3f}]  "
                  f"normal=[{normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f}]  "
                  f"dist={dist:.2f}m  pixels={perc.mask_pixels}")
            perceived_handles.append((ht, handle_pos, normal))
        else:
            # Skip — can't determine approach direction without perception
            link_pos = ht['link_world_pos']
            print(f"    {ht['fixture_name']:30s} NOT VISIBLE — skipping")

    # Sort by distance (tuple is now (ht, pos, normal))
    perceived_handles.sort(key=lambda x: np.linalg.norm(x[1] - arm_base))

    # Deduplicate: keep only the closest drawer per orientation group
    # (quantize yaw to nearest 90° so same-wall drawers are grouped)
    seen_yaw_buckets = {}
    unique_handles = []
    for ht, pos, normal in perceived_handles:
        approach = -normal / np.linalg.norm(normal)
        yaw = np.arctan2(approach[1], approach[0])
        bucket = round(yaw / (np.pi / 2))  # quantize to 0, ±1, ±2
        if bucket not in seen_yaw_buckets:
            seen_yaw_buckets[bucket] = ht['fixture_name']
            unique_handles.append((ht, pos, normal))
    perceived_handles = unique_handles
    print(f"\n  Unique orientations: {len(perceived_handles)} "
          f"(buckets: {seen_yaw_buckets})")

    if args.max_drawers is not None:
        perceived_handles = perceived_handles[:args.max_drawers]

    print(f"  Will attempt {len(perceived_handles)} drawers")

    # --- Setup planner ---
    print("\nSetting up SapienPlanner...")
    t0 = time.time()
    signal.alarm(30)
    try:
        pw = SapienPlanningWorld(scene, [robot._objs[0]])
        eef = next(n for n in pw.get_planned_articulations()[0]
                   .get_pinocchio_model().get_link_names() if 'eef' in n)
        planner = SapienPlanner(pw, move_group=eef)
    except TimeoutError:
        print("FATAL: planner setup timed out")
        env.close()
        return
    finally:
        signal.alarm(0)
    t_planner = time.time() - t0
    print(f"  planner setup: {t_planner:.2f}s")

    # Add fixture boxes — skip entire stacks containing target drawers
    skip_set = set()
    for ht, _, _ in perceived_handles:
        # Skip all fixtures in the same stack (e.g. stack_2_right_group_*)
        stack_prefix = '_'.join(ht['fixture_name'].split('_')[:2])
        for fn in fixtures:
            if fn.startswith(stack_prefix):
                skip_set.add(fn)
    print(f"  Skipping fixture boxes for drawer stacks: {skip_set}")
    fixture_box_names = add_fixture_boxes_to_planner(pw, scene, fixtures,
                                                      skip_fixtures=skip_set)
    print(f"  Added {len(fixture_box_names)} fixture boxes")

    # ACM
    target_names = {ht['link_name'] for ht, _, _ in perceived_handles}
    target_positions = [pos for _, pos, _ in perceived_handles]
    build_kitchen_acm(pw, planner, target_names, mode=args.acm,
                      robot_pos=arm_base, target_positions=target_positions)

    # --- Open drawers ---
    timings = {'ik': 0.0, 'planning': 0.0, 'exec': 0.0,
               'gripper': 0.0, 'settle': 0.0}
    results = {'success': 0, 'partial': 0, 'unreachable': 0, 'error': 0}

    for ci, (ht, handle_pos, normal) in enumerate(perceived_handles):
        dist = np.linalg.norm(handle_pos - arm_base)
        print(f"\n{'='*60}")
        print(f"[{ci+1}/{len(perceived_handles)}] {ht['fixture_name']} ({ht['fixture_type']}) "
              f"— link: {ht['link_name']}  dist={dist:.2f}m")

        step_label[0] = f"drawer {ci+1}/{len(perceived_handles)} {ht['fixture_name']}"

        try:
            outcome = attempt_drawer_open(handle_pos, ht, normal, robot,
                                          planner, pw, step_fn, timings, ci,
                                          len(perceived_handles), env=env)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            outcome = 'error'

        results[outcome] = results.get(outcome, 0) + 1
        print(f"  => {outcome.upper()}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k:12s}: {v}/{len(perceived_handles)}")

    t_total_elapsed = time.time() - t_total
    print(f"\nTIMING:")
    print(f"  env setup:     {t_env:7.2f}s")
    print(f"  planner setup: {t_planner:7.2f}s")
    print(f"  IK:            {timings['ik']:7.2f}s")
    print(f"  planning:      {timings['planning']:7.2f}s")
    print(f"  execution:     {timings['exec']:7.2f}s")
    print(f"  gripper:       {timings['gripper']:7.2f}s")
    print(f"  TOTAL:         {t_total_elapsed:7.2f}s")

    if video_writer:
        video_writer.close()

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
