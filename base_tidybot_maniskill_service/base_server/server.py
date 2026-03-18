"""Base bridge for ManiSkill — exposes mobile base control via RPC.

Protocol: multiprocessing.managers.BaseManager on port 50000 (same as MuJoCo version).
"""

import threading
import time
from multiprocessing.managers import BaseManager

import numpy as np

from .config import BASE_RPC_AUTHKEY, BASE_RPC_PORT


class SimBase:
    """Simulated base object exposed via RPC."""

    def __init__(self, server):
        self._server = server
        self._cmd_vel = [0.0, 0.0, 0.0]
        self._cmd_time = 0.0
        self._is_velocity_mode = False

        state = self._server.get_state()
        self._origin_x = state.base_x
        self._origin_y = state.base_y
        self._origin_theta = state.base_theta
        print(f"[base-bridge] Origin: x={self._origin_x:.3f}, "
              f"y={self._origin_y:.3f}, theta={self._origin_theta:.3f}")

    def _world_to_sdk(self, x, y, theta):
        dx = x - self._origin_x
        dy = y - self._origin_y
        cos_o = np.cos(-self._origin_theta)
        sin_o = np.sin(-self._origin_theta)
        sdk_x = cos_o * dx - sin_o * dy
        sdk_y = sin_o * dx + cos_o * dy
        sdk_theta = self._angle_wrap(theta - self._origin_theta)
        return sdk_x, sdk_y, sdk_theta

    def _sdk_to_world(self, sdk_x, sdk_y, sdk_theta):
        cos_o = np.cos(self._origin_theta)
        sin_o = np.sin(self._origin_theta)
        x = self._origin_x + cos_o * sdk_x - sin_o * sdk_y
        y = self._origin_y + sin_o * sdk_x + cos_o * sdk_y
        theta = self._angle_wrap(sdk_theta + self._origin_theta)
        return x, y, theta

    @staticmethod
    def _angle_wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def ensure_initialized(self):
        pass

    def get_full_state(self):
        state = self._server.get_state()
        sdk_x, sdk_y, sdk_theta = self._world_to_sdk(
            state.base_x, state.base_y, state.base_theta)
        return {
            "base_pose": np.array([sdk_x, sdk_y, sdk_theta]),
            "base_velocity": np.array([state.base_vx, state.base_vy, state.base_wz]),
        }

    def execute_action(self, target):
        pose = target["base_pose"]
        sdk_x, sdk_y, sdk_theta = float(pose[0]), float(pose[1]), float(pose[2])
        world_x, world_y, world_theta = self._sdk_to_world(sdk_x, sdk_y, sdk_theta)
        self._is_velocity_mode = False
        self._cmd_vel = [0.0, 0.0, 0.0]
        self._server.set_base_action([world_x, world_y, world_theta])

    def set_target_velocity(self, vel, frame="global"):
        self._cmd_vel = [float(vel[0]), float(vel[1]), float(vel[2])]
        self._cmd_time = time.time()
        self._is_velocity_mode = True

    def stop(self):
        self._is_velocity_mode = False
        self._cmd_vel = [0.0, 0.0, 0.0]

    def reset(self):
        pass

    def get_battery_voltage(self):
        return 25.2

    def get_command_state(self):
        return {
            "is_velocity_mode": self._is_velocity_mode,
            "cmd_vel": list(self._cmd_vel),
            "cmd_time": self._cmd_time,
        }


_sim_base_instance = None


class _SimBaseProxy:
    def __init__(self, real):
        self._real = real

    def ensure_initialized(self):
        return self._real.ensure_initialized()

    def get_full_state(self):
        return self._real.get_full_state()

    def execute_action(self, target):
        return self._real.execute_action(target)

    def set_target_velocity(self, vel, frame="global"):
        return self._real.set_target_velocity(vel, frame=frame)

    def stop(self):
        return self._real.stop()

    def reset(self):
        return self._real.reset()

    def get_battery_voltage(self):
        return self._real.get_battery_voltage()

    def get_command_state(self):
        return self._real.get_command_state()


def _base_factory():
    return _SimBaseProxy(_sim_base_instance)


class _BaseBridgeManager(BaseManager):
    pass


class BaseBridge:
    """Protocol bridge: multiprocessing.managers RPC on port 50000."""

    def __init__(self, server, port=BASE_RPC_PORT, authkey=BASE_RPC_AUTHKEY):
        self._server = server
        self._port = port
        self._authkey = authkey
        self._thread = None
        self._manager = None
        self._running = False

    def start(self):
        global _sim_base_instance
        _sim_base_instance = SimBase(self._server)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="base-bridge")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._manager is not None:
            try:
                server = self._manager.get_server()
                server.stop_event = True
            except Exception:
                pass
            self._manager = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        print("[base-bridge] Stopped")

    def _run(self):
        _BaseBridgeManager.register("Base", callable=_base_factory)
        self._manager = _BaseBridgeManager(
            address=("0.0.0.0", self._port),
            authkey=self._authkey,
        )
        server = self._manager.get_server()
        print(f"[base-bridge] RPC server listening on port {self._port}")
        try:
            server.serve_forever()
        except Exception as e:
            if self._running:
                print(f"[base-bridge] Server error: {e}")
