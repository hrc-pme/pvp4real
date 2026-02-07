#!/usr/bin/env python3
"""stretch3.deploy.py

Stretch3-side deploy runtime (robot container).

This is the deployment counterpart of `stretch3.hitl.py` and is also
*algorithm-agnostic*. It applies the same authority logic:

  - If `/stretch/is_teleop` is True, forward `/stretch/cmd_vel_teleop`
  - Else, forward policy command from `/pvp/novice_cmd_vel`

Use this for evaluation/deployment runs where the learning container only
performs inference and publishes `/pvp/novice_cmd_vel`.

ROS2 inputs (provided by Stretch3 bringup / standby):
  /stretch/is_teleop        std_msgs/Bool
  /stretch/cmd_vel_teleop   geometry_msgs/Twist
  /pvp/novice_cmd_vel       geometry_msgs/Twist

ROS2 output:
  /stretch/cmd_vel          geometry_msgs/Twist
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


def _now() -> float:
    return time.monotonic()


def _zero_twist() -> Twist:
    t = Twist()
    t.linear.x = 0.0
    t.angular.z = 0.0
    return t


@dataclass
class RuntimeCfg:
    hz: float = 10.0
    policy_stale_s: float = 0.5
    zero_on_stale: bool = True


class Stretch3DeployAuthority(Node):
    def __init__(self, cfg: RuntimeCfg):
        super().__init__("stretch3_deploy_authority")
        self.cfg = cfg

        self._is_teleop: bool = False
        self._teleop_cmd: Twist = _zero_twist()
        self._policy_cmd: Twist = _zero_twist()
        self._last_policy_t: float = -1.0

        qos_default = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Bool, "/stretch/is_teleop", self._on_is_teleop, qos_default)
        self.create_subscription(Twist, "/stretch/cmd_vel_teleop", self._on_teleop_cmd, qos_default)
        self.create_subscription(Twist, "/pvp/novice_cmd_vel", self._on_policy_cmd, qos_default)

        self._cmd_pub = self.create_publisher(Twist, "/stretch/cmd_vel", qos_default)

        self.create_timer(1.0 / float(cfg.hz), self._tick)

        self.get_logger().info(
            f"Deploy authority runtime started @ {cfg.hz}Hz. "
            f"Policy stale threshold: {cfg.policy_stale_s:.2f}s. "
            f"Policy topic: /pvp/novice_cmd_vel"
        )

    def _on_is_teleop(self, msg: Bool) -> None:
        self._is_teleop = bool(msg.data)

    def _on_teleop_cmd(self, msg: Twist) -> None:
        self._teleop_cmd = msg

    def _on_policy_cmd(self, msg: Twist) -> None:
        self._policy_cmd = msg
        self._last_policy_t = _now()

    def _policy_is_stale(self) -> bool:
        if self._last_policy_t < 0:
            return True
        return (_now() - self._last_policy_t) > float(self.cfg.policy_stale_s)

    def _tick(self) -> None:
        if self._is_teleop:
            cmd = self._teleop_cmd
        else:
            if self.cfg.zero_on_stale and self._policy_is_stale():
                cmd = _zero_twist()
            else:
                cmd = self._policy_cmd
        self._cmd_pub.publish(cmd)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stretch3 deploy runtime (authority/arbitration only)")
    p.add_argument("--hz", type=float, default=10.0, help="Control loop frequency.")
    p.add_argument("--policy_stale_s", type=float, default=0.5, help="Policy cmd stale threshold in seconds.")
    p.add_argument("--no_zero_on_stale", action="store_true", help="Do not zero cmd when policy is stale.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = RuntimeCfg(
        hz=float(args.hz),
        policy_stale_s=float(args.policy_stale_s),
        zero_on_stale=(not bool(args.no_zero_on_stale)),
    )

    rclpy.init()
    node: Optional[Stretch3DeployAuthority] = None
    try:
        node = Stretch3DeployAuthority(cfg)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
