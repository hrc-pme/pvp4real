#!/usr/bin/env python3
"""stretch3.hitl.py

Stretch3-side HITL runtime (robot container).

This script is intentionally *algorithm-agnostic* (no PVP/SB3/Gym/Torch deps).
It runs in the **Stretch3 container** and provides:

  - Human takeover arbitration via /stretch/is_teleop
  - Final command publication to /stretch/cmd_vel
  - Safety watchdog when policy commands stop arriving

Inter-container contract
------------------------
The learning container (pvp4) publishes the novice/policy command as a Twist:

  /pvp/novice_cmd_vel   geometry_msgs/Twist   (policy cmd in robot units)

This runtime chooses the authoritative command each tick:

  if is_teleop: publish /stretch/cmd_vel_teleop
  else:         publish /pvp/novice_cmd_vel (or zero if stale)

ROS2 inputs (provided by Stretch3 bringup / standby):
  /stretch/is_teleop        std_msgs/Bool
  /stretch/cmd_vel_teleop   geometry_msgs/Twist

ROS2 outputs:
  /stretch/cmd_vel          geometry_msgs/Twist

Notes
-----
* This script assumes the robot is already brought up (standby tested OK).
* Keep this script in the repo so it can be invoked from the mounted workspace,
  but run it inside the Stretch3 container.
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
    hz: float = 5.0
    # If we don't receive policy command for this long, output zero cmd.
    policy_stale_s: float = 0.5
    # If True, publish zeros when teleop is False and policy is stale.
    zero_on_stale: bool = True


class Stretch3Authority(Node):
    """Selects authoritative cmd_vel based on teleop flag."""

    def __init__(self, cfg: RuntimeCfg):
        super().__init__("stretch3_authority")
        self.cfg = cfg

        self._is_teleop: bool = False
        self._teleop_cmd: Twist = _zero_twist()
        self._policy_cmd: Twist = _zero_twist()
        self._last_policy_t: float = -1.0

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

        # Inputs
        self.create_subscription(Bool, "/stretch/is_teleop", self._on_is_teleop, qos_default)
        self.create_subscription(Twist, "/stretch/cmd_vel_teleop", self._on_teleop_cmd, qos_default)
        # Policy output from pvp container
        self.create_subscription(Twist, "/pvp/novice_cmd_vel", self._on_policy_cmd, qos_default)

        # Output
        self._cmd_pub = self.create_publisher(Twist, "/stretch/cmd_vel", qos_default)

        # Timer loop
        period = 1.0 / float(cfg.hz)
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f"Authority runtime started @ {cfg.hz}Hz. "
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
        # Select authoritative command
        if self._is_teleop:
            cmd = self._teleop_cmd
        else:
            if self.cfg.zero_on_stale and self._policy_is_stale():
                cmd = _zero_twist()
            else:
                cmd = self._policy_cmd
        self._cmd_pub.publish(cmd)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stretch3 HITL runtime (authority/arbitration only)")
    p.add_argument("--hz", type=float, default=5.0, help="Control loop frequency.")
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
    node: Optional[Stretch3Authority] = None
    try:
        node = Stretch3Authority(cfg)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
