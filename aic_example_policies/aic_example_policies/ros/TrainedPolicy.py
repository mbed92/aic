import json
import os
import time

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
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Vector3, Wrench
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
from visualization_msgs.msg import Marker

from aic_example_policies.ros.implementations.utils import (
    action_to_pose,
    img_to_tensor,
    publish_action_marker,
    quat_to_rot6d,
    task_to_one_hots,
)

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

IMG_H = 512
IMG_W = 576
CONTROL_DT_S = 0.05
STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]


class TrainedPolicy(Policy):
    POLICY_PATH_ENV_VAR = "TRAINED_POLICY_PATH"
    CONFIG_FILENAME = "config.json"
    MODEL_WEIGHTS_FILENAME = "model.safetensors"
    PREPROCESSOR_CONFIG_FILENAME = "policy_preprocessor.json"
    POSTPROCESSOR_CONFIG_FILENAME = "policy_postprocessor.json"
    IMAGE_FEATURES = (
        "observation.images.left_camera",
        "observation.images.center_camera",
        "observation.images.right_camera",
    )
    MARKER_TOPIC = "/trained_policy/action_chunk"
    MARKER_NAMESPACE = "trained_policy_action"

    def __init__(self, parent_node: Node, policy_path: str | Path | None = None):
        self._parent_node = parent_node
        Policy.__init__(self, parent_node)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_policy_path = self.resolve_policy_path(policy_path)
        self.load_policy(resolved_policy_path)

        self.action_chunk_marker_pub = self._parent_node.create_publisher(
            Marker, self.MARKER_TOPIC, 1
        )

        policy_name = type(self).__name__
        self.get_logger().info(
            f"{policy_name} loaded on {self.device} from {resolved_policy_path}"
        )
        self.get_logger().info(
            f"{policy_name} sends pose targets with wrench feedback disabled."
        )

    def resolve_policy_path(self, policy_path: str | Path | None) -> Path:
        if policy_path is None:
            env_policy_path = os.environ.get(self.POLICY_PATH_ENV_VAR)
            if not env_policy_path:
                raise RuntimeError(
                    f"{self.POLICY_PATH_ENV_VAR} must be set to a "
                    f"{type(self).__name__} checkpoint directory."
                )
            policy_path = env_policy_path

        if not policy_path:
            raise RuntimeError(
                f"{self.POLICY_PATH_ENV_VAR} resolved to an empty checkpoint path."
            )

        resolved_policy_path = Path(policy_path).expanduser()
        if not resolved_policy_path.exists():
            raise RuntimeError(f"Policy path does not exist: {resolved_policy_path}")
        if not resolved_policy_path.is_dir():
            raise RuntimeError(
                f"Policy path must be a directory: {resolved_policy_path}"
            )
        return resolved_policy_path

    def load_policy(self, policy_path: Path) -> None:
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
                raise RuntimeError(f"Missing checkpoint file: {required_path}")

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

            self.configure_data(config_dict)
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
                f"Failed to load checkpoint from {policy_path}: {exc}"
            ) from exc

    def configure_data(self, config_dict: dict[str, Any]) -> None:
        input_features = config_dict.get("input_features", {})
        output_features = config_dict.get("output_features", {})
        policy_name = type(self).__name__

        state_shape = input_features.get("observation.state", {}).get("shape")
        if state_shape != [18]:
            raise ValueError(
                f"{policy_name} expects an aic-dagger-data observation.state "
                f"shape of 18, got {state_shape}."
            )

        expected_image_features = set(self.IMAGE_FEATURES)
        actual_image_features = {
            feature_name
            for feature_name in input_features
            if feature_name.startswith("observation.images.")
        }
        if actual_image_features != expected_image_features:
            raise ValueError(
                f"{policy_name} expects exactly the aic-dagger-data image "
                f"features {sorted(expected_image_features)}, "
                f"got {sorted(actual_image_features)}."
            )

        action_shape = output_features.get("action", {}).get("shape")
        if action_shape != [9]:
            raise ValueError(
                f"{policy_name} expects an aic-dagger-data action shape of 9 "
                f"[x, y, z, rot6d_0..5], got {action_shape}."
            )

        self.state_dim = 18
        self.action_dim = 9

        self.get_logger().info(
            f"{policy_name} input features: images={list(self.IMAGE_FEATURES)}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}"
        )

    def prepare_observations(
        self, obs_msg: Observation, task: Task = None
    ) -> dict[str, torch.Tensor]:
        tcp_pose = obs_msg.controller_state.tcp_pose
        cable, rail, port = task_to_one_hots(task)
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
                quat_to_rot6d(
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

        return {
            "observation.images.left_camera": img_to_tensor(
                obs_msg.left_image, self.device, IMG_H, IMG_W
            ),
            "observation.images.center_camera": img_to_tensor(
                obs_msg.center_image, self.device, IMG_H, IMG_W
            ),
            "observation.images.right_camera": img_to_tensor(
                obs_msg.right_image, self.device, IMG_H, IMG_W
            ),
            "observation.state": torch.from_numpy(state_np).float().to(self.device),
        }

    def preprocess(self, batch: dict[str, torch.Tensor]):
        return self.preprocessor(batch)

    def infer(self, batch_tensor):
        with torch.inference_mode():
            return self.policy.select_action(batch_tensor)

    def postprocess(self, normalized_action) -> np.ndarray:
        action_tensor = self.postprocessor({"action": normalized_action})["action"]
        return action_tensor.detach().cpu().float().numpy().flatten()

    def publish_markers(self, current_pose: Pose, target_pose: Pose) -> None:
        publish_action_marker(
            self.action_chunk_marker_pub,
            self.get_clock().now().to_msg(),
            current_pose,
            target_pose,
            self.MARKER_NAMESPACE,
        )

    def set_pose_target_without_wrench_feedback(
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
            observation_msg = get_observation()
            if observation_msg is None:
                self.sleep_for(CONTROL_DT_S)
                continue

            batch = self.prepare_observations(observation_msg, task)
            batch_tensor = self.preprocess(batch)
            normalized_action = self.infer(batch_tensor)
            action = self.postprocess(normalized_action)
            self.get_logger().info(
                f"{policy_name} action: {np.round(action, 4).tolist()}"
            )

            target_pose = action_to_pose(action, policy_name=policy_name)
            self.publish_markers(
                observation_msg.controller_state.tcp_pose,
                target_pose,
            )
            self.set_pose_target_without_wrench_feedback(
                move_robot=move_robot,
                pose=target_pose,
            )

            send_feedback("in progress...")

            next_step += CONTROL_DT_S
            sleep_for = next_step - time.monotonic()
            if sleep_for > 0:
                self.sleep_for(sleep_for)
            else:
                next_step = time.monotonic()

        self.get_logger().info(f"{policy_name}.insert_cable() exiting...")
        return True
