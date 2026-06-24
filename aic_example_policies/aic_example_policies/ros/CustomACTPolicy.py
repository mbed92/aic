import json
import os
import time
import torchvision.transforms.functional as TF
import draccus
import numpy as np
import torch

from pathlib import Path
from typing import Any

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
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench

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


def _quat_to_rot6d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert xyzw quaternion to Zhou et al. 6D rotation representation."""
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm < 1e-9:
        rotmat = np.eye(3, dtype=np.float64)
    else:
        rotmat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
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
    qx, qy, qz, qw = Rotation.from_matrix(rotmat).as_quat()
    return float(qx), float(qy), float(qz), float(qw)


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


def _make_action_marker(
    marker_id: int,
    marker_type: int,
    stamp,
    lifetime_sec: int = 1,
) -> Marker:
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = stamp
    marker.ns = "custom_act_action"
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.lifetime.sec = lifetime_sec
    return marker


def _publish_action_marker(
    marker_pub,
    stamp,
    current_pose: Pose,
    target_pose: Pose,
) -> None:
    line_marker = _make_action_marker(0, Marker.LINE_STRIP, stamp)
    line_marker.scale.x = 0.004
    line_marker.color.r = 0.1
    line_marker.color.g = 0.8
    line_marker.color.b = 1.0
    line_marker.color.a = 0.9
    line_marker.points = [
        current_pose.position,
        target_pose.position,
    ]

    target_marker = _make_action_marker(1, Marker.SPHERE, stamp)
    target_marker.pose.position = target_pose.position
    target_marker.scale = Vector3(x=0.025, y=0.025, z=0.025)
    target_marker.color.r = 0.0
    target_marker.color.g = 1.0
    target_marker.color.b = 0.25
    target_marker.color.a = 0.95

    orientation_marker = _make_action_marker(2, Marker.ARROW, stamp)
    orientation_marker.pose = target_pose
    orientation_marker.scale = Vector3(x=0.06, y=0.01, z=0.018)
    orientation_marker.color.r = 1.0
    orientation_marker.color.g = 0.55
    orientation_marker.color.b = 0.0
    orientation_marker.color.a = 0.95

    marker_pub.publish(line_marker)
    marker_pub.publish(target_marker)
    marker_pub.publish(orientation_marker)


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

        self.action_chunk_marker_pub = self._parent_node.create_publisher(
            Marker, "/custom_act/action_chunk", 1
        )

        self.get_logger().info(
            f"CustomACTPolicy loaded on {self.device} from {resolved_policy_path}"
        )
        self.get_logger().info(
            "CustomACTPolicy sends pose targets with wrench feedback disabled."
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

    def prepare_observations(
        self, obs_msg: Observation, task: Task = None
    ) -> dict[str, torch.Tensor]:
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
            "observation.images.left_camera": _img_to_tensor(
                obs_msg.left_image, self.device, IMG_H, IMG_W
            ),
            "observation.images.center_camera": _img_to_tensor(
                obs_msg.center_image, self.device, IMG_H, IMG_W
            ),
            "observation.images.right_camera": _img_to_tensor(
                obs_msg.right_image, self.device, IMG_H, IMG_W
            ),
            "observation.state": torch.from_numpy(state_np).float().to(self.device),
        }
        return obs

    def _set_pose_target_without_wrench_feedback(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        frame_id: str = "base_link",
    ) -> None:
        motion_update = MotionUpdate()
        motion_update.header.frame_id = frame_id
        motion_update.header.stamp = self.get_clock().now().to_msg()
        motion_update.pose = pose
        motion_update.target_stiffness = np.diag(STIFFNESS).flatten()
        motion_update.target_damping = np.diag(DAMPING).flatten()
        motion_update.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.wrench_feedback_gains_at_tip = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        motion_update.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_POSITION
        )

        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")

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
            batch_tensor = self.preprocessor(batch)

            # 2. Model Inference
            with torch.inference_mode():
                # returns shape [1, action_dim] = [1, 9] (first action of chunk)
                normalized_action = self.policy.select_action(batch_tensor)

            # 3. Un-normalize Action
            action_tensor = self.postprocessor({"action": normalized_action})["action"]
            action = action_tensor.detach().cpu().float().numpy().flatten()
            self.get_logger().info(
                f"{policy_name} action: {np.round(action, 4).tolist()}"
            )

            # 4. Send Robot Command
            target_pose = _action_to_pose(action)
            _publish_action_marker(
                self.action_chunk_marker_pub,
                self.get_clock().now().to_msg(),
                observation_msg.controller_state.tcp_pose,
                target_pose,
            )

            self._set_pose_target_without_wrench_feedback(
                move_robot=move_robot,
                pose=target_pose,
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
