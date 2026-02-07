#!/usr/bin/env python3
"""pvp.deploy.py

PVP4Real-side deployment/inference loop (learning container).

This script loads a trained PVP4Real policy and continuously publishes the
novice/policy command to:

  /pvp/novice_cmd_vel   geometry_msgs/Twist

The Stretch3 container runs the authority runtime (`stretch3.deploy.py`) which
arbitrates human teleop vs. this policy command and publishes the authoritative
`/stretch/cmd_vel`.

Inputs (read):
  /camera/camera/color/image_raw                      sensor_msgs/Image
  /camera/camera/aligned_depth_to_color/image_raw     sensor_msgs/Image

Output (write):
  /pvp/novice_cmd_vel                          geometry_msgs/Twist
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2

from pvp.pvp_td3 import PVPTD3


def _now() -> float:
    return time.monotonic()


def _vw_to_twist(vw: np.ndarray) -> Twist:
    t = Twist()
    t.linear.x = float(vw[0])
    t.angular.z = float(vw[1])
    return t


@dataclass
class ObsCfg:
    resize_hw: Tuple[int, int] = (84, 84)
    stack_n: int = 5
    depth_max_m: float = 5.0


class DeployCache(Node):
    def __init__(self, obs_cfg: ObsCfg):
        super().__init__("pvp_deploy_cache")
        self.obs_cfg = obs_cfg
        self.bridge = CvBridge()

        self._rgb_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)
        self._depth_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_default = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, "/camera/camera/color/image_raw", self._on_rgb, qos_sensor)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self._on_depth, qos_sensor)

        self.novice_pub = self.create_publisher(Twist, "/pvp/novice_cmd_vel", qos_default)

        self.get_logger().info(
            f"Subscribed RGB/Depth. Will stack {obs_cfg.stack_n} frames @ {obs_cfg.resize_hw}. "
            "Publishing /pvp/novice_cmd_vel"
        )

    def _on_rgb(self, msg: Image) -> None:
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.obs_cfg.resize_hw[1], self.obs_cfg.resize_hw[0]), interpolation=cv2.INTER_AREA)
        self._rgb_stack.append(img_rgb)

    def _on_depth(self, msg: Image) -> None:
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        d = np.array(d)
        if d.dtype == np.uint16:
            depth_m = d.astype(np.float32) / 1000.0
        else:
            depth_m = d.astype(np.float32)
        depth_m = cv2.resize(depth_m, (self.obs_cfg.resize_hw[1], self.obs_cfg.resize_hw[0]), interpolation=cv2.INTER_NEAREST)
        self._depth_stack.append(depth_m)

    def stacks_ready(self) -> bool:
        return (len(self._rgb_stack) == self.obs_cfg.stack_n) and (len(self._depth_stack) == self.obs_cfg.stack_n)

    def build_obs_uint8(self) -> np.ndarray:
        if not self.stacks_ready():
            raise RuntimeError("Stacks not ready")

        rgb_list = list(self._rgb_stack)
        depth_list = list(self._depth_stack)

        depth_max = float(self.obs_cfg.depth_max_m)
        depth_u8_list = []
        for d in depth_list:
            d_clip = np.clip(d, 0.0, depth_max)
            d_u8 = (d_clip / depth_max * 255.0).astype(np.uint8)
            depth_u8_list.append(d_u8)

        rgb_cat = np.concatenate(rgb_list, axis=2)
        depth_cat = np.stack(depth_u8_list, axis=2)
        obs = np.concatenate([rgb_cat, depth_cat], axis=2)
        return obs.astype(np.uint8, copy=False)

    def publish_novice_vw(self, vw: np.ndarray) -> None:
        self.novice_pub.publish(_vw_to_twist(vw))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PVP4Real deploy (publishes /pvp/novice_cmd_vel)")
    p.add_argument("--model_path", type=str, required=True, help="Path to a saved .zip model checkpoint")
    p.add_argument("--hz", type=float, default=10.0, help="Inference publish frequency")
    p.add_argument("--max_lin", type=float, default=0.4, help="Max linear velocity (m/s)")
    p.add_argument("--max_ang", type=float, default=1.2, help="Max angular velocity (rad/s)")
    p.add_argument("--depth_max_m", type=float, default=5.0)
    p.add_argument("--stack_n", type=int, default=5)
    p.add_argument("--resize", type=int, nargs=2, default=[84, 84], metavar=("H", "W"))
    p.add_argument("--device", type=str, default="auto", help="Torch device (cuda/cpu/auto)")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic policy output")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    rclpy.init()
    node: Optional[DeployCache] = None
    try:
        obs_cfg = ObsCfg(
            resize_hw=(int(args.resize[0]), int(args.resize[1])),
            stack_n=int(args.stack_n),
            depth_max_m=float(args.depth_max_m),
        )
        node = DeployCache(obs_cfg)

        # Load model
        model = PVPTD3.load(str(model_path), device=str(args.device))

        dt = 1.0 / float(args.hz)
        last_t = _now()

        # Wait until obs ready
        start = _now()
        while rclpy.ok() and (not node.stacks_ready()):
            rclpy.spin_once(node, timeout_sec=0.05)
            if _now() - start > 10.0:
                raise TimeoutError("Timed out waiting for RGB-D stacks")

        node.get_logger().info(
            f"Starting deploy loop @ {args.hz}Hz (dt={dt:.3f}s). Model: {model_path}"
        )

        while rclpy.ok():
            # Wall-clock pacing
            elapsed = _now() - last_t
            if elapsed < dt:
                end_t = _now() + (dt - elapsed)
                while rclpy.ok() and _now() < end_t:
                    rclpy.spin_once(node, timeout_sec=0.01)
            last_t = _now()

            obs = node.build_obs_uint8()

            # SB3-style predict: returns (action, state)
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)

            vw = np.array([action[0] * float(args.max_lin), action[1] * float(args.max_ang)], dtype=np.float32)
            node.publish_novice_vw(vw)

            # keep callbacks flowing
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
