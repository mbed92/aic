import os

from pathlib import Path

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.node import Node

from aic_example_policies.ros.TrainedPolicy import TrainedPolicy


class SmolVLATrainedPolicy(TrainedPolicy):
    """Template for a future SmolVLA policy implementation."""

    POLICY_PATH_ENV_VAR = "SMOLVLA_TRAINED_POLICY_PATH"

    def __init__(self, parent_node: Node, policy_path: str | Path | None = None):
        self._parent_node = parent_node
        Policy.__init__(self, parent_node)
        self.policy_path = self._resolve_template_policy_path(policy_path)
        self.get_logger().warn(
            "SmolVLATrainedPolicy is a template only. "
            "Model loading and inference are not implemented."
        )

    def _resolve_template_policy_path(self, policy_path: str | Path | None):
        if policy_path is not None:
            return Path(policy_path).expanduser()

        env_policy_path = os.environ.get(self.POLICY_PATH_ENV_VAR)
        if env_policy_path:
            return Path(env_policy_path).expanduser()

        return None

    def load_policy(self, policy_path: Path) -> None:
        raise NotImplementedError(
            "SmolVLA model loading is not implemented yet. "
            "Add SmolVLA checkpoint loading here."
        )

    def configure_data(self, config_dict) -> None:
        raise NotImplementedError(
            "SmolVLA data configuration is not implemented yet. "
            "Define the SmolVLA input/output schema here."
        )

    def prepare_observations(self, obs_msg, task: Task = None):
        raise NotImplementedError(
            "SmolVLA observation preparation is not implemented yet."
        )

    def infer(self, batch_tensor):
        raise NotImplementedError("SmolVLA inference is not implemented yet.")

    def postprocess(self, normalized_action):
        raise NotImplementedError("SmolVLA postprocessing is not implemented yet.")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        raise NotImplementedError(
            "SmolVLATrainedPolicy is a template only and does not command the robot. "
            "Implement loading, observation preparation, inference, postprocessing, "
            "and insertion behavior before using it for evaluation."
        )

