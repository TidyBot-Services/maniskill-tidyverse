"""Camera bridge for ManiSkill — streams camera images via WebSocket.

Protocol: WebSocket on port 5580 (same as MuJoCo version).
  JSON text commands, binary frames with JSON header.
"""

import asyncio
import io
import json
import struct
import threading
import time

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from .config import CAMERA_WS_PORT, DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH, SIM_CAMERAS

# Message types (same as MuJoCo camera bridge)
GET_STATE = 1
STATE = 2
SUBSCRIBE = 3
UNSUBSCRIBE = 4
ACK = 5
ERROR = 6
GET_INTRINSICS = 7
INTRINSICS = 8


class CameraBridge:
    """Protocol bridge: WebSocket for camera streaming."""

    def __init__(self, server, port=CAMERA_WS_PORT):
        self._server = server
        self._port = port
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run_ws_server, daemon=True, name="camera-bridge")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        print("[camera-bridge] Stopped")

    def _run_ws_server(self):
        try:
            import websockets
            import websockets.sync.server as ws_sync
        except ImportError:
            print("[camera-bridge] ERROR: websockets not installed")
            return

        print(f"[camera-bridge] WebSocket on port {self._port}")

        def handler(websocket):
            self._handle_client_sync(websocket)

        with ws_sync.serve(handler, "0.0.0.0", self._port) as server:
            while self._running:
                server.socket.settimeout(0.5)
                try:
                    server.serve_forever()
                except Exception:
                    break

    def _handle_client_sync(self, ws):
        """Handle a single WebSocket client connection."""
        streaming = False
        fps = DEFAULT_FPS
        quality = 80

        try:
            while self._running:
                # Check for incoming messages (non-blocking)
                try:
                    raw = ws.recv(timeout=0.01)
                    msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())

                    # Support both {"action": ...} (client) and {"type": ...} formats
                    action = msg.get("action", "")
                    msg_type = msg.get("type", -1)

                    if action == "get_state" or msg_type == GET_STATE:
                        ws.send(json.dumps(self._build_state()))
                    elif action == "subscribe" or msg_type == SUBSCRIBE:
                        streaming = True
                        fps = msg.get("fps", DEFAULT_FPS)
                        quality = msg.get("quality", 80)
                        ws.send(json.dumps({"type": ACK, "action": "ack"}))
                    elif action == "unsubscribe" or msg_type == UNSUBSCRIBE:
                        streaming = False
                        ws.send(json.dumps({"type": ACK, "action": "ack"}))
                    elif action == "get_intrinsics" or msg_type == GET_INTRINSICS:
                        ws.send(json.dumps(self._build_intrinsics()))
                except TimeoutError:
                    pass
                except Exception:
                    break

                if streaming:
                    state = self._server.get_state()
                    # Send RGB frame
                    if state.camera_rgb is not None:
                        frame_data = self._encode_jpeg(state.camera_rgb, quality)
                        header = {
                            "camera_id": "maniskill_001",
                            "stream": "color",
                            "timestamp": time.time(),
                            "width": state.camera_rgb.shape[1],
                            "height": state.camera_rgb.shape[0],
                            "format": "jpeg",
                        }
                        ws.send(self._pack_frame(header, frame_data))

                    # Send depth frame
                    if state.camera_depth is not None:
                        frame_data = self._encode_depth_png(state.camera_depth)
                        header = {
                            "camera_id": "maniskill_001",
                            "stream": "depth",
                            "timestamp": time.time(),
                            "width": state.camera_depth.shape[1],
                            "height": state.camera_depth.shape[0],
                            "format": "png",
                        }
                        ws.send(self._pack_frame(header, frame_data))

                    time.sleep(1.0 / fps)
                else:
                    time.sleep(0.1)

        except Exception:
            pass

    @staticmethod
    def _pack_frame(header, data):
        """Pack header + data into binary frame: [4B header_len][JSON header][data]."""
        header_bytes = json.dumps(header).encode()
        return struct.pack(">I", len(header_bytes)) + header_bytes + data

    @staticmethod
    def _encode_jpeg(rgb_array, quality=80):
        """Encode HxWx3 uint8 array to JPEG bytes."""
        if Image is not None:
            img = Image.fromarray(rgb_array.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
        # Fallback: raw bytes
        return rgb_array.tobytes()

    @staticmethod
    def _encode_depth_png(depth_array):
        """Encode HxWx1 depth array to PNG bytes (16-bit depth in mm)."""
        if Image is not None:
            # Convert to mm (ManiSkill depth is in meters as int16)
            depth_2d = depth_array.squeeze()
            if depth_2d.dtype != np.uint16:
                depth_2d = np.clip(depth_2d * 1000, 0, 65535).astype(np.uint16)
            img = Image.fromarray(depth_2d, mode="I;16")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        return depth_array.tobytes()

    @staticmethod
    def _build_state():
        return {
            "type": STATE,
            "data": {
                "timestamp": time.time(),
                "cameras": [
                    {
                        "device_id": cam["serial"],
                        "name": cam["name"],
                        "camera_type": "maniskill",
                        "serial_number": cam["serial"],
                        "width": DEFAULT_WIDTH,
                        "height": DEFAULT_HEIGHT,
                        "fps": DEFAULT_FPS,
                        "streams": ["color", "depth"],
                        "firmware_version": "",
                    }
                    for cam in SIM_CAMERAS.values()
                ],
                "active_streams": {},
                "is_streaming": True,
                "error": "",
            },
        }

    @staticmethod
    def _build_intrinsics():
        fov = np.pi / 2
        fy = DEFAULT_HEIGHT / (2 * np.tan(fov / 2))
        fx = fy
        return {
            "type": INTRINSICS,
            "data": {
                "fx": fx, "fy": fy,
                "ppx": DEFAULT_WIDTH / 2.0,
                "ppy": DEFAULT_HEIGHT / 2.0,
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "depth_scale": 0.001,
            },
        }
