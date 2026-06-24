import numpy as np
import torch
import torchvision.transforms.functional as TF

from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker

NUM_RAILS = 5
NUM_PORTS = 2


def quat_to_rot6d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert xyzw quaternion to Zhou et al. 6D rotation representation."""
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm < 1e-9:
        rotmat = np.eye(3, dtype=np.float64)
    else:
        rotmat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return np.concatenate([rotmat[:, 0], rotmat[:, 1]]).astype(np.float32)


def rot6d_to_quat(rot6d: np.ndarray) -> tuple[float, float, float, float]:
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


def task_to_one_hots(task: Task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cable[2], rail[5], and port[2] one-hots for bha-51 state."""
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


def img_to_tensor(
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


def action_to_pose(action: np.ndarray, policy_name: str = "TrainedPolicy") -> Pose:
    if action.shape[0] < 9:
        raise ValueError(f"{policy_name} expected 9-dim action, got {action.shape}")

    qx, qy, qz, qw = rot6d_to_quat(action[3:9])
    return Pose(
        position=Point(
            x=float(action[0]),
            y=float(action[1]),
            z=float(action[2]),
        ),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )


def make_action_marker(
    marker_id: int,
    marker_type: int,
    stamp,
    marker_namespace: str,
    lifetime_sec: int = 1,
) -> Marker:
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = stamp
    marker.ns = marker_namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.lifetime.sec = lifetime_sec
    return marker


def publish_action_marker(
    marker_pub,
    stamp,
    current_pose: Pose,
    target_pose: Pose,
    marker_namespace: str,
) -> None:
    line_marker = make_action_marker(0, Marker.LINE_STRIP, stamp, marker_namespace)
    line_marker.scale.x = 0.004
    line_marker.color.r = 0.1
    line_marker.color.g = 0.8
    line_marker.color.b = 1.0
    line_marker.color.a = 0.9
    line_marker.points = [
        current_pose.position,
        target_pose.position,
    ]

    target_marker = make_action_marker(1, Marker.SPHERE, stamp, marker_namespace)
    target_marker.pose.position = target_pose.position
    target_marker.scale = Vector3(x=0.025, y=0.025, z=0.025)
    target_marker.color.r = 0.0
    target_marker.color.g = 1.0
    target_marker.color.b = 0.25
    target_marker.color.a = 0.95

    orientation_marker = make_action_marker(2, Marker.ARROW, stamp, marker_namespace)
    orientation_marker.pose = target_pose
    orientation_marker.scale = Vector3(x=0.06, y=0.01, z=0.018)
    orientation_marker.color.r = 1.0
    orientation_marker.color.g = 0.55
    orientation_marker.color.b = 0.0
    orientation_marker.color.a = 0.95

    marker_pub.publish(line_marker)
    marker_pub.publish(target_marker)
    marker_pub.publish(orientation_marker)

