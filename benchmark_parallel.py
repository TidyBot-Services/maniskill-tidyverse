#!/usr/bin/env python3
"""Benchmark parallel environments with different sensor configurations.

Tests how many parallel envs can run with sensors on the current GPU.
Each config runs in a separate subprocess to avoid GPU PhysX re-init issues.

Usage:
    conda activate maniskill
    python benchmark_parallel.py
    python benchmark_parallel.py --env PushCube-v1
    python benchmark_parallel.py --robocasa
"""
import sys, os, json, argparse, subprocess, time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Worker script that runs a single benchmark in isolation
WORKER_CODE = r'''
import sys, os, time, json, gc
sys.path.insert(0, os.environ["BENCH_DIR"])

import numpy as np
import torch
import sapien
import gymnasium as gym
from mani_skill.sensors.camera import CameraConfig

# Import agent (registers 'tidyverse')
import tidyverse_agent as _ta
import mani_skill.envs

# Parse args from env vars
env_id = os.environ["BENCH_ENV"]
num_envs = int(os.environ["BENCH_NUM_ENVS"])
obs_mode = os.environ["BENCH_OBS_MODE"]
n_cameras = int(os.environ["BENCH_N_CAMERAS"])
resolution = int(os.environ["BENCH_RESOLUTION"])
n_steps = int(os.environ["BENCH_STEPS"])

# Build sensor configs and patch agent
def make_sensors():
    configs = []
    if n_cameras >= 1:
        configs.append(CameraConfig(
            uid="wrist_camera",
            pose=sapien.Pose(p=[0, 0, 0.05], q=[1, 0, 0, 0]),
            width=resolution, height=resolution,
            fov=np.pi / 2, near=0.01, far=100,
            entity_uid="eef",
        ))
    if n_cameras >= 2:
        configs.append(CameraConfig(
            uid="overhead_camera",
            pose=sapien.Pose(p=[0, 0, 0.5], q=[0.707, 0.707, 0, 0]),
            width=resolution, height=resolution,
            fov=np.pi / 2, near=0.01, far=100,
            entity_uid="panda_link0",
        ))
    return configs

sensor_cfgs = make_sensors()
_ta.TidyVerse._sensor_configs = property(lambda self: sensor_cfgs)


def get_vram():
    try:
        r = __import__("subprocess").run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return float(r.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0

result = {}
try:
    vram_before = get_vram()
    t0 = time.time()
    env = gym.make(env_id, num_envs=num_envs, robot_uids="tidyverse",
                   control_mode="pd_joint_pos", obs_mode=obs_mode, render_mode=None)
    obs, info = env.reset(seed=0)
    t_setup = time.time() - t0

    has_images = isinstance(obs, dict) and "sensor_data" in obs

    action_space = env.action_space
    t_start = time.time()
    for _ in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(action_space.sample())
    t_elapsed = time.time() - t_start

    vram_after = get_vram()
    fps = (n_steps * num_envs) / t_elapsed

    env.close()

    result = {
        "env_id": env_id, "num_envs": num_envs,
        "obs_mode": obs_mode, "n_cameras": n_cameras,
        "resolution": resolution, "setup_time": round(t_setup, 2),
        "fps": round(fps, 1), "vram_before_mb": vram_before,
        "vram_after_mb": vram_after,
        "vram_delta_mb": round(vram_after - vram_before, 1),
        "has_images": has_images, "status": "ok",
    }
except Exception as e:
    err = str(e)
    is_oom = "out of memory" in err.lower() or "cuda" in err.lower() or "vk" in err.lower()
    result = {
        "env_id": env_id, "num_envs": num_envs,
        "obs_mode": obs_mode, "n_cameras": n_cameras,
        "resolution": resolution,
        "status": "OOM" if is_oom else f"ERROR: {err[:120]}",
    }

print("BENCH_RESULT:" + json.dumps(result))
'''


def run_one(env_id, num_envs, obs_mode, n_cameras, resolution, n_steps):
    """Run a single benchmark in a subprocess."""
    env = os.environ.copy()
    env.update({
        "BENCH_DIR": SCRIPT_DIR,
        "BENCH_ENV": env_id,
        "BENCH_NUM_ENVS": str(num_envs),
        "BENCH_OBS_MODE": obs_mode,
        "BENCH_N_CAMERAS": str(n_cameras),
        "BENCH_RESOLUTION": str(resolution),
        "BENCH_STEPS": str(n_steps),
    })

    try:
        proc = subprocess.run(
            [sys.executable, "-c", WORKER_CODE],
            capture_output=True, text=True, timeout=300, env=env,
        )
        # Find result line
        for line in proc.stdout.split("\n"):
            if line.startswith("BENCH_RESULT:"):
                return json.loads(line[len("BENCH_RESULT:"):])

        # No result found — check stderr
        stderr = proc.stderr[-200:] if proc.stderr else ""
        return {
            "env_id": env_id, "num_envs": num_envs,
            "obs_mode": obs_mode, "n_cameras": n_cameras,
            "resolution": resolution,
            "status": f"ERROR: no result. stderr: {stderr}",
        }
    except subprocess.TimeoutExpired:
        return {
            "env_id": env_id, "num_envs": num_envs,
            "obs_mode": obs_mode, "n_cameras": n_cameras,
            "resolution": resolution, "status": "TIMEOUT",
        }


def print_results(results):
    print(f"\n{'='*105}")
    print(f"{'Env':25s} | {'Sensors':20s} | {'num_envs':>8s} | {'FPS':>10s} | "
          f"{'VRAM(MB)':>9s} | {'Setup(s)':>8s} | {'Status':10s}")
    print(f"{'-'*105}")
    for r in results:
        sensor_desc = (f"{r.get('n_cameras', 0)}x {r.get('resolution', '-')} "
                       f"{r.get('obs_mode', '-')}")
        if r["status"] == "ok":
            print(f"{r['env_id']:25s} | {sensor_desc:20s} | {r['num_envs']:8d} | "
                  f"{r['fps']:10.1f} | {r['vram_after_mb']:9.0f} | "
                  f"{r['setup_time']:8.2f} | {r['status']:10s}")
        else:
            print(f"{r['env_id']:25s} | {sensor_desc:20s} | {r['num_envs']:8d} | "
                  f"{'---':>10s} | {'---':>9s} | {'---':>8s} | "
                  f"{r['status'][:30]:30s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PushCube-v1")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--robocasa", action="store_true",
                        help="Also benchmark RoboCasaKitchen-v1")
    args = parser.parse_args()

    num_envs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    sensor_configs = [
        ("state", 0, 0),       # no sensor baseline
        ("rgbd", 1, 128),      # 1x 128x128 RGBD
        ("rgbd", 2, 128),      # 2x 128x128 RGBD
        ("rgbd", 1, 256),      # 1x 256x256 RGBD
    ]

    envs_to_test = [args.env]
    if args.robocasa:
        envs_to_test.append("RoboCasaKitchen-v1")

    all_results = []

    for env_id in envs_to_test:
        print(f"\n{'#'*60}")
        print(f"# Benchmarking: {env_id}")
        print(f"{'#'*60}")

        for obs_mode, n_cameras, resolution in sensor_configs:
            label = (f"{n_cameras}x{resolution} {obs_mode}"
                     if n_cameras > 0 else "no sensor (state)")
            print(f"\n--- Sensor config: {label} ---")

            for n in num_envs_list:
                print(f"  num_envs={n:4d} ... ", end="", flush=True)
                result = run_one(env_id, n, obs_mode, n_cameras, resolution,
                                 args.steps)
                all_results.append(result)

                if result["status"] == "ok":
                    print(f"FPS={result['fps']:.1f}  "
                          f"VRAM={result['vram_after_mb']:.0f}MB  "
                          f"setup={result['setup_time']:.1f}s")
                else:
                    print(f"{result['status']}")
                    if "OOM" in result["status"] or "TIMEOUT" in result["status"]:
                        print(f"  -> Stopping this config at num_envs={n}")
                        break

    print_results(all_results)

    out_path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
