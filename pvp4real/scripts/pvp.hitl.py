#!/usr/bin/env python3
"""pvp.hitl.py

PVP4Real-side HITL online training loop (learning container).

This script runs inside the **pvp4real container** and performs learning while
remaining agnostic to the robot container's control authority.

Key contract
------------
The Stretch3 container is responsible for publishing the final authoritative
command to `/stretch/cmd_vel` (it arbitrates human vs policy).  This script:
  - Publishes the policy/novice command to `/pvp/novice_cmd_vel`
  - Observes the executed behavior via `/stretch/cmd_vel`
  - Uses `/stretch/is_teleop` + `/stretch/cmd_vel_teleop` for HITL signals

ROS2 topics
-----------
Inputs (read):
  RGB:     /camera/camera/color/image_raw                      sensor_msgs/Image
  Depth:   /camera/camera/aligned_depth_to_color/image_raw     sensor_msgs/Image
  I(s):    /stretch/is_teleop                           std_msgs/Bool
  a_h:     /stretch/cmd_vel_teleop                      geometry_msgs/Twist
  a_exec:  /stretch/cmd_vel                             geometry_msgs/Twist

Output (write):
  a_n:     /pvp/novice_cmd_vel                          geometry_msgs/Twist

Observation encoding (paper-aligned default):
  - Resize RGB + Depth to HxW (default 84x84)
  - Stack last N frames (default 5)
  - Channel-last uint8: (H, W, N*(3+1))
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import cv2

# Gymnasium is used by the repo's SB3 fork (some modules import gymnasium.spaces)
import gymnasium as gym

from pvp.pvp_td3 import PVPTD3


def _now() -> float:
    return time.monotonic()


def _twist_to_vw(msg: Twist) -> np.ndarray:
    return np.array([float(msg.linear.x), float(msg.angular.z)], dtype=np.float32)


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


class HITLCache(Node):
    """Caches latest ROS messages and maintains stacked RGB-D frames."""

    def __init__(self, obs_cfg: ObsCfg):
        super().__init__("pvp_hitl_cache")
        self.obs_cfg = obs_cfg
        self.bridge = CvBridge()

        # Latest signals
        self._is_teleop: bool = False
        self._teleop_twist: Twist = Twist()
        self._exec_twist: Twist = Twist()

        # Frame stacks
        self._rgb_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)
        self._depth_stack: Deque[np.ndarray] = deque(maxlen=obs_cfg.stack_n)

        # takeover rising edge
        self._prev_takeover: bool = False

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
        self.create_subscription(Image, "/camera/camera/color/image_raw", self._on_rgb, qos_sensor)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self._on_depth, qos_sensor)
        self.create_subscription(Bool, "/stretch/is_teleop", self._on_is_teleop, qos_default)
        self.create_subscription(Twist, "/stretch/cmd_vel_teleop", self._on_teleop_twist, qos_default)
        self.create_subscription(Twist, "/stretch/cmd_vel", self._on_exec_twist, qos_default)

        # Output: policy/novice command (robot units)
        self.novice_pub = self.create_publisher(Twist, "/pvp/novice_cmd_vel", qos_default)

        self.get_logger().info(
            f"Subscribed RGB/Depth + HITL signals. Will stack {obs_cfg.stack_n} frames @ {obs_cfg.resize_hw}."
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

    def _on_is_teleop(self, msg: Bool) -> None:
        self._is_teleop = bool(msg.data)

    def _on_teleop_twist(self, msg: Twist) -> None:
        self._teleop_twist = msg

    def _on_exec_twist(self, msg: Twist) -> None:
        self._exec_twist = msg

    def stacks_ready(self) -> bool:
        return (len(self._rgb_stack) == self.obs_cfg.stack_n) and (len(self._depth_stack) == self.obs_cfg.stack_n)

    def get_takeover_and_start(self) -> Tuple[bool, bool]:
        takeover = self._is_teleop
        takeover_start = takeover and (not self._prev_takeover)
        self._prev_takeover = takeover
        return takeover, takeover_start

    def get_human_vw(self) -> np.ndarray:
        return _twist_to_vw(self._teleop_twist)

    def get_exec_vw(self) -> np.ndarray:
        return _twist_to_vw(self._exec_twist)

    def publish_novice_vw(self, vw: np.ndarray) -> None:
        self.novice_pub.publish(_vw_to_twist(vw))

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

        rgb_cat = np.concatenate(rgb_list, axis=2)              # HxWx(3N)
        depth_cat = np.stack(depth_u8_list, axis=2)             # HxWxN
        obs = np.concatenate([rgb_cat, depth_cat], axis=2)      # HxWx(4N)
        return obs.astype(np.uint8, copy=False)
    
    def publish_is_teleop(self, is_teleop: bool) -> None:
        """Publish is_teleop state to switch between gamepad/navigation mode."""
        # Create publisher for is_teleop (on-demand)
        if not hasattr(self, 'is_teleop_pub'):
            qos_default = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            self.is_teleop_pub = self.create_publisher(Bool, "/stretch/is_teleop", qos_default)
        
        msg = Bool()
        msg.data = is_teleop
        self.is_teleop_pub.publish(msg)
        self.get_logger().info(f"Mode switched: {'Gamepad (Teleop)' if is_teleop else 'Navigation (Policy)'}")


class HITLControlGUI:
    """Simple GUI for controlling HITL training process."""
    
    def __init__(self, node: HITLCache, total_steps: int, model_dir: str, save_every: int):
        self.node = node
        self.total_steps = total_steps
        self.model_dir = model_dir
        self.save_every = save_every
        self.current_steps = 0
        self.quit_requested = False
        self.is_teleop = False  # Start in navigation mode
        
        # Create GUI window
        self.root = tk.Tk()
        self.root.title("PVP HITL Training Control")
        self.root.geometry("500x350")
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="PVP4Real HITL Training", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Training status section
        status_frame = ttk.LabelFrame(main_frame, text="Training Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Steps display
        ttk.Label(status_frame, text="Current Steps:").grid(row=0, column=0, sticky=tk.W)
        self.steps_label = ttk.Label(status_frame, text="0", font=("Arial", 12, "bold"))
        self.steps_label.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))
        
        ttk.Label(status_frame, text="Total Steps:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(status_frame, text=f"{total_steps}", font=("Arial", 12)).grid(row=1, column=1, sticky=tk.E, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(status_frame, text="Save Every:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(status_frame, text=f"{save_every} steps", font=("Arial", 12)).grid(row=2, column=1, sticky=tk.E, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(status_frame, text="Model Directory:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        dir_label = ttk.Label(status_frame, text=f"{model_dir}", font=("Arial", 9), wraplength=350)
        dir_label.grid(row=3, column=1, sticky=tk.E, padx=(10, 0), pady=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, length=400, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=2, pady=(10, 5), sticky=(tk.W, tk.E))
        self.progress['maximum'] = total_steps
        
        # Progress percentage
        self.progress_label = ttk.Label(status_frame, text="0.0%", font=("Arial", 10))
        self.progress_label.grid(row=5, column=0, columnspan=2)
        
        # Control buttons section
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Mode switch button
        self.mode_button = ttk.Button(control_frame, text="Switch to Gamepad Mode", 
                                      command=self._on_mode_switch, width=25)
        self.mode_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Mode status label
        self.mode_label = ttk.Label(control_frame, text="Current: Navigation (Policy)", 
                                    font=("Arial", 10, "bold"), foreground="green")
        self.mode_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Quit button
        quit_button = ttk.Button(control_frame, text="Save & Quit Training", 
                                command=self._on_quit, width=25)
        quit_button.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(1, weight=1)
        
    def _on_mode_switch(self):
        """Toggle between gamepad and navigation mode."""
        self.is_teleop = not self.is_teleop
        
        # Publish the new state
        self.node.publish_is_teleop(self.is_teleop)
        
        # Update GUI
        if self.is_teleop:
            self.mode_button.config(text="Switch to Navigation Mode")
            self.mode_label.config(text="Current: Gamepad (Teleop)", foreground="orange")
        else:
            self.mode_button.config(text="Switch to Gamepad Mode")
            self.mode_label.config(text="Current: Navigation (Policy)", foreground="green")
    
    def _on_quit(self):
        """Request graceful shutdown."""
        self.quit_requested = True
        self.root.quit()
    
    def update_steps(self, steps: int):
        """Update the current step count in GUI."""
        self.current_steps = steps
        self.steps_label.config(text=f"{steps}")
        self.progress['value'] = steps
        progress_pct = (steps / self.total_steps) * 100 if self.total_steps > 0 else 0
        self.progress_label.config(text=f"{progress_pct:.1f}%")
    
    def run(self):
        """Start the GUI main loop in a separate thread."""
        self.root.mainloop()
    
    def start_thread(self):
        """Start GUI in a separate thread."""
        gui_thread = threading.Thread(target=self.run, daemon=True)
        gui_thread.start()
        return gui_thread


class StretchHITLEnv(gym.Env):
    """Gymnasium env used for learning; publishes novice cmd, observes executed cmd."""

    def __init__(self, node: HITLCache, dt: float, max_lin: float, max_ang: float):
        super().__init__()
        self.node = node
        self.dt = float(dt)
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self._last_step_t = _now()

        h, w = node.obs_cfg.resize_hw
        c = node.obs_cfg.stack_n * (3 + 1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _action_to_vw(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a.astype(np.float32), -1.0, 1.0)
        return np.array([a[0] * self.max_lin, a[1] * self.max_ang], dtype=np.float32)

    def _vw_to_action(self, vw: np.ndarray) -> np.ndarray:
        return np.array([vw[0] / self.max_lin, vw[1] / self.max_ang], dtype=np.float32).clip(-1.0, 1.0)

    def seed(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]
  
    def reset(self, *, seed: Optional[int] = None, options=None):  # gymnasium API
        if seed is not None:
            np.random.seed(seed)
        start = _now()
        while rclpy.ok() and (not self.node.stacks_ready()):
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if _now() - start > 10.0:
                raise TimeoutError("Timed out waiting for RGB-D stacks")
        obs = self.node.build_obs_uint8()
        return obs, {}

    def step(self, action: np.ndarray):  # gymnasium API
        # Wall-clock pacing
        elapsed = _now() - self._last_step_t
        if elapsed < self.dt:
            end_t = _now() + (self.dt - elapsed)
            while rclpy.ok() and _now() < end_t:
                rclpy.spin_once(self.node, timeout_sec=0.01)
        self._last_step_t = _now()

        obs = self.node.build_obs_uint8()

        takeover, takeover_start = self.node.get_takeover_and_start()
        human_vw = self.node.get_human_vw()
        exec_vw = self.node.get_exec_vw()

        novice_action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        novice_vw = self._action_to_vw(novice_action)
        # Publish novice command (Stretch3 container arbitrates authority)
        self.node.publish_novice_vw(novice_vw)

        # Spin briefly to keep callbacks flowing
        t_end = _now() + 0.02
        while rclpy.ok() and _now() < t_end:
            rclpy.spin_once(self.node, timeout_sec=0.0)

        next_obs = self.node.build_obs_uint8() if self.node.stacks_ready() else obs

        # Behavior action for buffer: executed cmd (normalized)
        raw_action = self._vw_to_action(exec_vw)

        reward = 0.0
        terminated = False
        truncated = False

        info = {
            "takeover": float(takeover),
            "takeover_start": float(takeover_start),
            "takeover_cost": 0.0,
            "raw_action": raw_action.astype(np.float32),
        }
        return next_obs, reward, terminated, truncated, info


def load_config(config_path: Optional[str]) -> dict:
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PVP4Real HITL training (publishes /pvp/novice_cmd_vel)")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml file. Default: scripts/config.yaml")
    p.add_argument("--model_dir", type=str, default=None, help="Directory to save checkpoints and logs.")
    p.add_argument("--is_resume_training", type=str, default=None, choices=["True", "False", None], 
                   help="Enable resume from checkpoint")
    p.add_argument("--resume_from", type=str, default=None, 
                   help="Directory path containing checkpoint files to resume from")
    p.add_argument("--hz", type=float, default=None)
    p.add_argument("--max_lin", type=float, default=None)
    p.add_argument("--max_ang", type=float, default=None)
    p.add_argument("--depth_max_m", type=float, default=None)
    p.add_argument("--stack_n", type=int, default=None)
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--learning_starts", type=int, default=None)
    p.add_argument("--buffer_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--q_value_bound", type=float, default=None)
    p.add_argument("--bc_loss_weight", type=float, default=None)
    p.add_argument("--with_human_proxy_value_loss", type=str, default=None, choices=["True", "False", None])
    p.add_argument("--with_agent_proxy_value_loss", type=str, default=None, choices=["True", "False", None])
    p.add_argument("--only_bc_loss", type=str, default=None, choices=["True", "False", None])
    p.add_argument("--add_bc_loss", type=str, default=None, choices=["True", "False", None])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)

    model_dir = args.model_dir or cfg["training"]["model_dir"]
    
    # Resume training parameters
    is_resume_training = args.is_resume_training if args.is_resume_training is not None else str(cfg["training"].get("is_resume_training", False))
    is_resume_training = is_resume_training.lower() == "true" if isinstance(is_resume_training, str) else bool(is_resume_training)
    resume_from = args.resume_from or cfg["training"].get("resume_from", None)
    
    hz = args.hz if args.hz is not None else cfg["common"]["hz"]
    max_lin = args.max_lin if args.max_lin is not None else cfg["common"]["max_lin"]
    max_ang = args.max_ang if args.max_ang is not None else cfg["common"]["max_ang"]
    depth_max_m = args.depth_max_m if args.depth_max_m is not None else cfg["common"]["depth_max_m"]
    stack_n = args.stack_n if args.stack_n is not None else cfg["common"]["stack_n"]
    resize = args.resize if args.resize is not None else [cfg["common"]["resize"]["height"], cfg["common"]["resize"]["width"]]
    total_steps = args.total_steps if args.total_steps is not None else cfg["training"]["total_steps"]
    learning_starts = args.learning_starts if args.learning_starts is not None else cfg["training"]["learning_starts"]
    buffer_size = args.buffer_size if args.buffer_size is not None else cfg["training"]["buffer_size"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"]
    seed = args.seed if args.seed is not None else cfg["common"]["seed"]
    device = args.device or cfg["common"]["device"]
    save_every = args.save_every if args.save_every is not None else cfg["training"]["save_every"]
    log_interval = args.log_interval if args.log_interval is not None else cfg["training"]["log_interval"]
    q_value_bound = args.q_value_bound if args.q_value_bound is not None else cfg["training"]["pvp"]["q_value_bound"]
    bc_loss_weight = args.bc_loss_weight if args.bc_loss_weight is not None else cfg["training"]["pvp"]["bc_loss_weight"]
    with_human_proxy_value_loss = args.with_human_proxy_value_loss if args.with_human_proxy_value_loss is not None else str(cfg["training"]["pvp"]["with_human_proxy_value_loss"])
    with_agent_proxy_value_loss = args.with_agent_proxy_value_loss if args.with_agent_proxy_value_loss is not None else str(cfg["training"]["pvp"]["with_agent_proxy_value_loss"])
    only_bc_loss = args.only_bc_loss if args.only_bc_loss is not None else str(cfg["training"]["pvp"]["only_bc_loss"])
    add_bc_loss = args.add_bc_loss if args.add_bc_loss is not None else str(cfg["training"]["pvp"]["add_bc_loss"])

    gamma = cfg["training"]["pvp"]["gamma"]
    tau = cfg["training"]["pvp"]["tau"]
    learning_rate = cfg["training"]["pvp"]["learning_rate"]

    os.makedirs(model_dir, exist_ok=True)

    rclpy.init()
    node: Optional[HITLCache] = None
    gui: Optional[HITLControlGUI] = None
    try:
        obs_cfg = ObsCfg(
            resize_hw=(int(resize[0]), int(resize[1])),
            stack_n=int(stack_n),
            depth_max_m=float(depth_max_m),
        )
        node = HITLCache(obs_cfg)

        dt = 1.0 / float(hz)
        env = StretchHITLEnv(node=node, dt=dt, max_lin=float(max_lin), max_ang=float(max_ang))

        # Initialize trained step counter
        trained = 0
        
        # Check if resuming from checkpoint
        if is_resume_training and resume_from:
            resume_path = Path(resume_from)
            
            # If resume_from is a .zip file, use it directly
            if resume_path.is_file() and resume_path.suffix == '.zip':
                checkpoint_file = str(resume_path)
            # If it's a directory, look for the .zip file inside
            elif resume_path.is_dir():
                zip_files = list(resume_path.glob("*.zip"))
                if len(zip_files) == 0:
                    raise FileNotFoundError(f"No .zip checkpoint file found in directory: {resume_from}")
                elif len(zip_files) > 1:
                    node.get_logger().warn(f"Multiple .zip files found in {resume_from}, using the first one: {zip_files[0]}")
                checkpoint_file = str(zip_files[0])
            else:
                raise FileNotFoundError(f"Checkpoint path not found or invalid: {resume_from}")
            
            node.get_logger().info(f"Resuming training from checkpoint: {checkpoint_file}")
            
            # Load the model from checkpoint
            model = PVPTD3.load(
                checkpoint_file,
                env=env,
                verbose=1,
                device=device,
                learning_rate=learning_rate,
                # Note: Replay buffer will be reinitialized
            )
            
            # Try to extract step number from filename
            import re
            match = re.search(r'step(\d+)', checkpoint_file)
            if match:
                trained = int(match.group(1))
                node.get_logger().info(f"Resuming from step {trained}")
            else:
                node.get_logger().warn(f"Could not extract step number from checkpoint filename, starting from 0")
            
            # For resumed training, set learning_starts to 0 (model is already trained)
            learning_starts = 0
            node.get_logger().info("Set learning_starts=0 for resumed training (model already pre-trained)")
            
        else:
            # Create new model from scratch
            node.get_logger().info("Creating new model from scratch")
            model = PVPTD3(
                True,  # use_balance_sample
                q_value_bound,
                "CnnPolicy",
                env,
                seed=seed,
                verbose=1,
                device=device,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                train_freq=(1, "step"),
                gradient_steps=1,
                gamma=gamma,
                tau=tau,
                learning_rate=learning_rate,
                bc_loss_weight=bc_loss_weight,
                with_human_proxy_value_loss=with_human_proxy_value_loss,
                with_agent_proxy_value_loss=with_agent_proxy_value_loss,
                only_bc_loss=only_bc_loss,
                add_bc_loss=add_bc_loss,
                adaptive_batch_size="False",  # Fixed: add missing parameter
            )

        # Ensure ready
        _ = env.reset(seed=seed)

        resume_status = f" (resumed from step {trained})" if is_resume_training and trained > 0 else ""
        node.get_logger().info(
            f"Starting HITL training for {total_steps} steps @ {hz}Hz (dt={dt:.3f}s){resume_status}.\n"
            f"Publishing novice commands: /pvp/novice_cmd_vel\n"
            f"Saving to: {model_dir}"
        )

        # Start GUI
        gui = HITLControlGUI(node=node, total_steps=total_steps, 
                            model_dir=model_dir, save_every=save_every)
        gui.start_thread()
        
        # Update GUI with current step if resuming
        if trained > 0:
            gui.update_steps(trained)
        
        node.get_logger().info("GUI started. Use the control panel to switch modes or quit.")

        remaining = int(total_steps) - trained
        while rclpy.ok() and remaining > 0 and not gui.quit_requested:
            chunk = min(int(save_every), remaining)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, log_interval=int(log_interval))
            trained += chunk
            remaining -= chunk
            
            # Update GUI
            gui.update_steps(trained)
            
            ckpt_path = os.path.join(model_dir, f"pvp4real_stretch3_step{trained}.zip")
            model.save(ckpt_path)
            node.get_logger().info(f"Checkpoint saved: {ckpt_path}")

        final_path = os.path.join(model_dir, "pvp4real_stretch3_final.zip")
        model.save(final_path)
        node.get_logger().info(f"Final model saved: {final_path}")
    except KeyboardInterrupt:
        node.get_logger().info("\n[KeyboardInterrupt] Stopping training and saving current model...")
        if 'model' in locals() and 'trained' in locals() and trained > 0:
            interrupt_path = os.path.join(model_dir, f"pvp4real_stretch3_interrupted_step{trained}.zip")
            model.save(interrupt_path)
            node.get_logger().info(f"Model saved at interruption: {interrupt_path}")
        else:
            node.get_logger().warn("No model to save (training not started yet)")
    finally:
        if gui is not None:
            try:
                gui.root.quit()
                gui.root.destroy()
            except:
                pass
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
