import json
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import draccus
import numpy as np
import torch
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from rclpy.node import Node
from safetensors.torch import load_file

from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench


# aic-dagger-data schema:
#   observation.images.{left,center,right}_camera shape=[512, 576, 3]
#   observation.state shape=[18]:
#     tcp_pos_x, tcp_pos_y, tcp_pos_z,
#     tcp_rot6d_0..5,
#     cable_sfp, cable_sc,
#     rail_0..4,
#     port_0, port_1
#   action shape=[9]:
#     x, y, z, rot6d_0..5      (absolute TCP pose target)
NUM_RAILS = 5
NUM_PORTS = 2


def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm < 1e-9:
        return np.eye(3, dtype=np.float32)

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def _quat_to_rot6d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert xyzw quaternion to Zhou et al. 6D rotation representation."""
    rotmat = _quat_to_rotmat(qx, qy, qz, qw)
    return np.concatenate([rotmat[:, 0], rotmat[:, 1]]).astype(np.float32)


def _rot6d_to_quat(rot6d: np.ndarray) -> tuple[float, float, float, float]:
    """Convert Zhou et al. 6D rotation representation to an xyzw quaternion."""
    a1 = rot6d[:3].astype(np.float64)
    a2 = rot6d[3:6].astype(np.float64)

    n1 = np.linalg.norm(a1)
    b1 = a1 / n1 if n1 > 1e-9 else np.array([1.0, 0.0, 0.0])
    a2_orthogonal = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2_orthogonal)
    b2 = a2_orthogonal / n2 if n2 > 1e-9 else np.array([0.0, 1.0, 0.0])
    b3 = np.cross(b1, b2)
    rotmat = np.stack([b1, b2, b3], axis=1)

    trace = rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2]
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * scale
        qx = (rotmat[2, 1] - rotmat[1, 2]) / scale
        qy = (rotmat[0, 2] - rotmat[2, 0]) / scale
        qz = (rotmat[1, 0] - rotmat[0, 1]) / scale
    elif rotmat[0, 0] > rotmat[1, 1] and rotmat[0, 0] > rotmat[2, 2]:
        scale = (1.0 + rotmat[0, 0] - rotmat[1, 1] - rotmat[2, 2]) ** 0.5 * 2.0
        qw = (rotmat[2, 1] - rotmat[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotmat[0, 1] + rotmat[1, 0]) / scale
        qz = (rotmat[0, 2] + rotmat[2, 0]) / scale
    elif rotmat[1, 1] > rotmat[2, 2]:
        scale = (1.0 + rotmat[1, 1] - rotmat[0, 0] - rotmat[2, 2]) ** 0.5 * 2.0
        qw = (rotmat[0, 2] - rotmat[2, 0]) / scale
        qx = (rotmat[0, 1] + rotmat[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotmat[1, 2] + rotmat[2, 1]) / scale
    else:
        scale = (1.0 + rotmat[2, 2] - rotmat[0, 0] - rotmat[1, 1]) ** 0.5 * 2.0
        qw = (rotmat[1, 0] - rotmat[0, 1]) / scale
        qx = (rotmat[0, 2] + rotmat[2, 0]) / scale
        qy = (rotmat[1, 2] + rotmat[2, 1]) / scale
        qz = 0.25 * scale

    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    return float(qx / norm), float(qy / norm), float(qz / norm), float(qw / norm)


def _task_to_one_hots(task: Task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cable[2], rail[5], and port[2] one-hots for bha-51 state.

    cable: index 0 = sfp, 1 = sc.
    rail: nic_card_mount_<i>, i in 0..4. For SC trials this remains all zeros.
    port: sfp_port_<i>, i in 0..1. For SC trials this remains all zeros.
    """
    cable = np.zeros(2, dtype=np.float32)
    rail = np.zeros(NUM_RAILS, dtype=np.float32)
    port = np.zeros(NUM_PORTS, dtype=np.float32)

    plug_type = (task.plug_type or "").lower()
    if plug_type == "sfp":
        cable[0] = 1.0
    elif plug_type == "sc":
        cable[1] = 1.0

    target_module_name = task.target_module_name or ""
    if target_module_name.startswith("nic_card_mount_"):
        try:
            rail_index = int(target_module_name.split("_")[-1])
            if 0 <= rail_index < NUM_RAILS:
                rail[rail_index] = 1.0
        except ValueError:
            pass

    port_name = task.port_name or ""
    if port_name.startswith("sfp_port_"):
        try:
            port_index = int(port_name.split("_")[-1])
            if 0 <= port_index < NUM_PORTS:
                port[port_index] = 1.0
        except ValueError:
            pass

    return cable, rail, port


class CustomACTPolicy(Policy):
    DEFAULT_POLICY_PATH = (
        "/media/mbed/T7/datasets/aic/aic-dagger-data-outputs/train/act_aic_dagger_data/"
        "checkpoints/last/pretrained_model"
    )
    POLICY_PATH_PARAMETER = "custom_act_policy_path"

    _IMAGE_SOURCES = {
        "left_camera": "left_image",
        "left_wrist": "left_image",
        "center_camera": "center_image",
        "overhead": "center_image",
        "right_camera": "right_image",
        "right_wrist": "right_image",
        "side_camera": "center_image",
    }

    def __init__(self, parent_node: Node, policy_path: str | Path | None = None):
        self._parent_node = parent_node
        Policy.__init__(self, parent_node)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_policy_path = self._resolve_policy_path(parent_node, policy_path)
        self._load_local_policy(resolved_policy_path)

        self.get_logger().info(
            f"CustomACTPolicy loaded on {self.device} from {resolved_policy_path}"
        )

    def _resolve_policy_path(
        self, parent_node: Node, policy_path: str | Path | None
    ) -> Path:
        if policy_path is None:
            if not parent_node.has_parameter(self.POLICY_PATH_PARAMETER):
                parent_node.declare_parameter(
                    self.POLICY_PATH_PARAMETER, self.DEFAULT_POLICY_PATH
                )
            policy_path = (
                parent_node.get_parameter(self.POLICY_PATH_PARAMETER)
                .get_parameter_value()
                .string_value
            )

        resolved_policy_path = Path(
            policy_path or self.DEFAULT_POLICY_PATH
        ).expanduser()
        if not resolved_policy_path.exists():
            raise FileNotFoundError(
                f"ACT policy path does not exist: {resolved_policy_path}"
            )
        if not resolved_policy_path.is_dir():
            raise NotADirectoryError(
                f"ACT policy path must be a directory: {resolved_policy_path}"
            )
        return resolved_policy_path

    def _load_local_policy(self, policy_path: Path) -> None:
        config_path = policy_path / "config.json"
        model_weights_path = policy_path / "model.safetensors"
        stats_path = (
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        for required_path in (config_path, model_weights_path, stats_path):
            if not required_path.exists():
                raise FileNotFoundError(f"Missing ACT checkpoint file: {required_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config_dict.pop("type", None)

        config = draccus.decode(ACTConfig, config_dict)
        if hasattr(config, "device"):
            config.device = str(self.device)

        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(model_weights_path))
        self.policy.eval()
        self.policy.to(self.device)

        stats = load_file(stats_path)
        self._configure_inputs(config_dict, stats)

        self.action_mean = self._get_stat(stats, "action.mean", (1, -1))
        self.action_std = self._get_stat(stats, "action.std", (1, -1))
        self.action_dim = int(self.action_mean.shape[1])
        if self.action_dim != 9:
            raise ValueError(
                "CustomACTPolicy expects a bha-51 action shape of 9 "
                f"[x, y, z, rot6d_0..5], got {self.action_dim}."
            )

    def _configure_inputs(self, config_dict: dict[str, Any], stats: dict[str, Any]):
        input_features = config_dict.get("input_features", {})
        self.image_features: dict[str, dict[str, Any]] = {}
        self.img_stats: dict[str, dict[str, torch.Tensor]] = {}
        self.state_dim = 0

        for feature_name, feature_cfg in input_features.items():
            feature_type = feature_cfg.get("type")
            shape = feature_cfg.get("shape", [])

            if feature_type == "VISUAL":
                source_name = feature_name.split(".")[-1]
                image_attr = self._IMAGE_SOURCES.get(source_name)
                if image_attr is None:
                    raise ValueError(
                        f"Unsupported ACT image feature '{feature_name}'. "
                        f"Known image sources: {sorted(self._IMAGE_SOURCES)}"
                    )
                if len(shape) != 3:
                    raise ValueError(
                        f"Unsupported ACT image shape for '{feature_name}': {shape}"
                    )
                if shape[0] == 3:
                    height = int(shape[1])
                    width = int(shape[2])
                elif shape[2] == 3:
                    height = int(shape[0])
                    width = int(shape[1])
                else:
                    raise ValueError(
                        f"Unsupported ACT image shape for '{feature_name}': {shape}"
                    )

                self.image_features[feature_name] = {
                    "image_attr": image_attr,
                    "height": height,
                    "width": width,
                }
                self.img_stats[feature_name] = {
                    "mean": self._get_stat(stats, f"{feature_name}.mean", (1, 3, 1, 1)),
                    "std": self._get_stat(stats, f"{feature_name}.std", (1, 3, 1, 1)),
                }

            elif feature_name == "observation.state":
                if len(shape) != 1:
                    raise ValueError(f"Unsupported ACT state shape: {shape}")
                self.state_dim = int(shape[0])

            elif feature_name == "observation.contact":
                raise ValueError(
                    "CustomACTPolicy now targets bha-51's schema and does not "
                    "support observation.contact in the checkpoint input features."
                )

        if not self.image_features:
            raise ValueError("ACT config does not define any visual input features.")
        if self.state_dim != 18:
            raise ValueError(
                "CustomACTPolicy expects a bha-51 observation.state shape of 18, "
                f"got {self.state_dim}."
            )

        self.state_mean = self._get_stat(stats, "observation.state.mean", (1, -1))
        self.state_std = self._get_stat(stats, "observation.state.std", (1, -1))

        self.get_logger().info(
            "CustomACTPolicy input features: "
            f"images={list(self.image_features.keys())}, "
            f"state_dim={self.state_dim}"
        )

    def _get_stat(
        self, stats: dict[str, Any], key: str, shape: tuple[int, ...]
    ) -> torch.Tensor:
        if key not in stats:
            raise KeyError(f"Missing normalization statistic '{key}'")
        return stats[key].to(self.device).view(*shape)

    @staticmethod
    def _img_to_tensor(
        raw_img,
        device: torch.device,
        height: int,
        width: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if img_np.shape[0] != height or img_np.shape[1] != width:
            img_np = cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)

        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        return (tensor - mean) / std

    def prepare_observations(
        self, obs_msg: Observation, task: Task = None
    ) -> Dict[str, torch.Tensor]:
        obs = {}

        # Convert each configured image feature from the observation message into a normalized tensor and add it to the obs dict.
        for feature_name, feature_info in self.image_features.items():
            obs[feature_name] = self._img_to_tensor(
                getattr(obs_msg, feature_info["image_attr"]),
                self.device,
                feature_info["height"],
                feature_info["width"],
                self.img_stats[feature_name]["mean"],
                self.img_stats[feature_name]["std"],
            )

        state_np = self._dataset_state_from_observation(obs_msg, task)
        if state_np.shape[0] != self.state_dim:
            raise ValueError(
                f"ACT checkpoint expects state_dim={self.state_dim}, "
                f"but CustomACTPolicy built {state_np.shape[0]} state values."
            )

        # Normalize the state vector and add it to the obs dict.
        raw_state_tensor = (
            torch.from_numpy(state_np)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std

        return obs

    @staticmethod
    def _dataset_state_from_observation(
        obs_msg: Observation, task: Task = None
    ) -> np.ndarray:
        tcp_pose = obs_msg.controller_state.tcp_pose
        rot6d = _quat_to_rot6d(
            tcp_pose.orientation.x,
            tcp_pose.orientation.y,
            tcp_pose.orientation.z,
            tcp_pose.orientation.w,
        )
        cable, rail, port = (
            _task_to_one_hots(task)
            if task
            else (
                np.zeros(2, dtype=np.float32),
                np.zeros(NUM_RAILS, dtype=np.float32),
                np.zeros(NUM_PORTS, dtype=np.float32),
            )
        )

        return np.concatenate(
            [
                np.array(
                    [
                        tcp_pose.position.x,
                        tcp_pose.position.y,
                        tcp_pose.position.z,
                    ],
                    dtype=np.float32,
                ),
                rot6d,
                cable,
                rail,
                port,
            ]
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        policy_name = type(self).__name__
        self.get_logger().info(f"{policy_name}.insert_cable() enter. Task: {task}")

        start_time = time.time()

        while time.time() - start_time < 30.0:
            loop_start = time.time()
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(observation_msg, task)

            with torch.inference_mode():
                normalized_action = self.policy.select_action(obs_tensors)

            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean
            action = raw_action_tensor[0].cpu().numpy()

            self.get_logger().info(f"Action: {action}")

            motion_update = self.set_cartesian_pose_target(
                self._action_to_pose(action)
            )
            move_robot(motion_update=motion_update)
            send_feedback("in progress...")

            # Maintain control rate (approx 4Hz loop = 0.25s sleep)
            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.25 - elapsed))

        self.get_logger().info(f"{policy_name}.insert_cable() exiting...")
        return True

    @staticmethod
    def _action_to_pose(action: np.ndarray) -> Pose:
        if action.shape[0] < 9:
            raise ValueError(f"CustomACTPolicy expected 9-dim action, got {action.shape}")

        qx, qy, qz, qw = _rot6d_to_quat(action[3:9])
        return Pose(
            position=Point(
                x=float(action[0]),
                y=float(action[1]),
                z=float(action[2]),
            ),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )

    def set_cartesian_pose_target(self, pose: Pose, frame_id: str = "base_link"):
        motion_update_msg = MotionUpdate()
        motion_update_msg.pose = pose
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()

        motion_update_msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
        )

        motion_update_msg.wrench_feedback_gains_at_tip = [
            0.5,
            0.5,
            0.5,
            0.0,
            0.0,
            0.0,
        ]
        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_POSITION
        )

        return motion_update_msg
