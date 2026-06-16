import json
import os
import time
from pathlib import Path
from typing import Any, Dict

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
from lerobot.processor.pipeline import (
    DataProcessorPipeline,
    PolicyProcessorPipeline,
    batch_to_transition,
    transition_to_batch,
)
from rclpy.node import Node
from safetensors.torch import load_file

from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
import torchvision.transforms.functional as TF

# -------------------------------------------------------------------------
# aic-dagger-data schema:
# -------------------------------------------------------------------------
#   observation.images.{left,center,right}_camera shape=[512, 576, 3]
#   observation.state shape=[18]:
#     tcp_pos_x, tcp_pos_y, tcp_pos_z,
#     tcp_rot6d_0..5,
#     cable_sfp, cable_sc,
#     rail_0..4,
#     port_0, port_1
#   action shape=[9]:
#     x, y, z, rot6d_0..5      (absolute TCP pose target)

# Schema baked the task target into the state vector. We parse the
# Task message (plug_type, target_module_name, port_name) to set the bits.

# Rails 0..4 — extracted from "nic_card_mount_<i>" target_module_name.
NUM_RAILS = 5
# Ports 0..1 — extracted from "sfp_port_<i>" port_name.
NUM_PORTS = 2

# dataset image resolution (training native).
IMG_H = 512
IMG_W = 576

# Control loop cadence — must match training (20 Hz)
CONTROL_DT_S = 0.05

# Impedance constants — verified to match CheatCode's MotionUpdate stream
# during training (inspect_motion_updates.py). Hardcoded at deployment so the
# learned pose targets execute under the same controller behavior they trained
# against.
STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]


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
    POLICY_PATH_ENV_VAR = "CUSTOM_ACT_POLICY_PATH"
    CONFIG_FILENAME = "config.json"
    MODEL_WEIGHTS_FILENAME = "model.safetensors"
    PREPROCESSOR_CONFIG_FILENAME = "policy_preprocessor.json"
    POSTPROCESSOR_CONFIG_FILENAME = "policy_postprocessor.json"
    IMAGE_FEATURES = (
        "observation.images.left_camera",
        "observation.images.center_camera",
        "observation.images.right_camera",
    )

    def __init__(self, parent_node: Node, policy_path: str | Path | None = None):
        self._parent_node = parent_node
        Policy.__init__(self, parent_node)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------------------------------------------------
        # 1. Configuration & Weights Loading
        # -------------------------------------------------------------------------
        resolved_policy_path = self._resolve_policy_path(policy_path)
        self._load_local_policy(resolved_policy_path)

        self.get_logger().info(
            f"CustomACTPolicy loaded on {self.device} from {resolved_policy_path}"
        )

    def _resolve_policy_path(self, policy_path: str | Path | None) -> Path:
        if policy_path is None:
            env_policy_path = os.environ.get(self.POLICY_PATH_ENV_VAR)
            if not env_policy_path:
                raise RuntimeError(
                    f"{self.POLICY_PATH_ENV_VAR} must be set to a CustomACTPolicy "
                    "checkpoint directory. The VS Code 'Policy: start' task defines "
                    "this environment variable."
                )
            policy_path = env_policy_path

        if not policy_path:
            raise RuntimeError(
                f"{self.POLICY_PATH_ENV_VAR} resolved to an empty checkpoint path."
            )

        resolved_policy_path = Path(policy_path).expanduser()
        if not resolved_policy_path.exists():
            raise RuntimeError(
                f"ACT policy path does not exist: {resolved_policy_path}"
            )
        if not resolved_policy_path.is_dir():
            raise RuntimeError(
                f"ACT policy path must be a directory: {resolved_policy_path}"
            )
        return resolved_policy_path

    def _load_local_policy(self, policy_path: Path) -> None:
        config_path = policy_path / self.CONFIG_FILENAME
        model_weights_path = policy_path / self.MODEL_WEIGHTS_FILENAME
        preprocessor_config_path = policy_path / self.PREPROCESSOR_CONFIG_FILENAME
        postprocessor_config_path = policy_path / self.POSTPROCESSOR_CONFIG_FILENAME

        for required_path in (
            config_path,
            model_weights_path,
            preprocessor_config_path,
            postprocessor_config_path,
        ):
            if not required_path.exists():
                raise RuntimeError(f"Missing ACT checkpoint file: {required_path}")

        try:
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

            self._configure_inputs(config_dict)
            device_override = {"device_processor": {"device": str(self.device)}}
            self.preprocessor = DataProcessorPipeline.from_pretrained(
                policy_path,
                config_filename=self.PREPROCESSOR_CONFIG_FILENAME,
                overrides=device_override,
            )
            self.postprocessor = PolicyProcessorPipeline.from_pretrained(
                policy_path,
                config_filename=self.POSTPROCESSOR_CONFIG_FILENAME,
                overrides=device_override,
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ACT checkpoint from {policy_path}: {exc}"
            ) from exc

    def _configure_inputs(self, config_dict: dict[str, Any]) -> None:
        input_features = config_dict.get("input_features", {})
        output_features = config_dict.get("output_features", {})

        state_shape = input_features.get("observation.state", {}).get("shape")
        if state_shape != [18]:
            raise ValueError(
                "CustomACTPolicy expects a bha-51 observation.state shape of 18, "
                f"got {state_shape}."
            )

        expected_image_features = set(self.IMAGE_FEATURES)
        actual_image_features = {
            feature_name
            for feature_name in input_features
            if feature_name.startswith("observation.images.")
        }
        if actual_image_features != expected_image_features:
            raise ValueError(
                "CustomACTPolicy expects exactly the bha-51 image features "
                f"{sorted(expected_image_features)}, got {sorted(actual_image_features)}."
            )

        action_shape = output_features.get("action", {}).get("shape")
        if action_shape != [9]:
            raise ValueError(
                "CustomACTPolicy expects a bha-51 action shape of 9 "
                f"[x, y, z, rot6d_0..5], got {action_shape}."
            )

        self.state_dim = 18
        self.action_dim = 9

        self.get_logger().info(
            "CustomACTPolicy input features: "
            f"images={list(self.IMAGE_FEATURES)}, state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}"
        )

    @staticmethod
    def _img_to_tensor(
        raw_img,
        device: torch.device,
        height: int,
        width: int,
    ) -> torch.Tensor:
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )

        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).to(device)
        return TF.resize(tensor, [height, width], antialias=True)

    def prepare_observations(
        self, obs_msg: Observation, task: Task = None
    ) -> Dict[str, torch.Tensor]:
        tcp_pose = obs_msg.controller_state.tcp_pose
        cable, rail, port = _task_to_one_hots(task)
        state_np = np.concatenate(
            [
                np.array(
                    [
                        tcp_pose.position.x,
                        tcp_pose.position.y,
                        tcp_pose.position.z,
                    ],
                    dtype=np.float32,
                ),
                _quat_to_rot6d(
                    tcp_pose.orientation.x,
                    tcp_pose.orientation.y,
                    tcp_pose.orientation.z,
                    tcp_pose.orientation.w,
                ),
                cable,
                rail,
                port,
            ]
        )

        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image, self.device, IMG_H, IMG_W
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image, self.device, IMG_H, IMG_W
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image, self.device, IMG_H, IMG_W
            ),
            "observation.state": torch.from_numpy(state_np).float().to(self.device),
        }
        return self.preprocessor(obs)

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

        deadline = time.monotonic() + float(task.time_limit)
        next_step = time.monotonic()

        while time.monotonic() < deadline:
            # 1. Get & Process Observation
            observation_msg = get_observation()
            if observation_msg is None:
                self.sleep_for(CONTROL_DT_S)
                continue

            batch = self.prepare_observations(observation_msg, task)

            # 2. Model Inference
            with torch.inference_mode():
                # returns shape [1, action_dim] = [1, 9] (first action of chunk)
                normalized_action = self.policy.select_action(batch)

            # 3. Un-normalize Action
            action_tensor = self.postprocessor({"action": normalized_action})["action"]
            action = action_tensor.detach().cpu().float().numpy().flatten()
            self.get_logger().info(f"Action: {action}")

            # 4. Send Robot Command
            self.set_pose_target(
                move_robot=move_robot,
                pose=self._action_to_pose(action),
                stiffness=STIFFNESS,
                damping=DAMPING,
            )

            send_feedback("in progress...")

            # 5. Sleep to maintain control loop cadence.
            next_step += CONTROL_DT_S
            sleep_for = next_step - time.monotonic()
            if sleep_for > 0:
                self.sleep_for(sleep_for)
            else:
                next_step = time.monotonic()

        self.get_logger().info(f"{policy_name}.insert_cable() exiting...")
        return True

    @staticmethod
    def _action_to_pose(action: np.ndarray) -> Pose:
        if action.shape[0] < 9:
            raise ValueError(
                f"CustomACTPolicy expected 9-dim action, got {action.shape}"
            )

        qx, qy, qz, qw = _rot6d_to_quat(action[3:9])
        return Pose(
            position=Point(
                x=float(action[0]),
                y=float(action[1]),
                z=float(action[2]),
            ),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )
