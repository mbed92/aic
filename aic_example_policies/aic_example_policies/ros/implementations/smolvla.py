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
    POLICY_PATH_ENV_VAR = "SMOLVLA_TRAINED_POLICY_PATH"