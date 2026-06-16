# Example Policies

This package contains baseline policy implementations that demonstrate different approaches to the cable insertion task. These examples serve as reference implementations and starting points for developing your own policies.

> [!NOTE]
> **Prerequisites:** Before running these policies, ensure you have the evaluation environment running. See [Getting Started](../docs/getting_started.md) for setup instructions.
>
> **Command Format:**
> - If using the **container workflow** (recommended): Launch with `distrobox enter -r aic_eval -- /entrypoint.sh [parameters]`
> - If **built from source**: Launch with `ros2 launch aic_bringup aic_gz_bringup.launch.py [parameters]`
> - Run policies with `pixi run ros2 run` (Pixi workspace) or `ros2 run` (native ROS 2)

---

## Available Policies

### 1. WaveArm - Minimal Example

![Wave Arm Policy](../../media/wave_arm_policy.gif)

A minimal example showing how to implement the `insert_cable()` callback and issue motion commands to the arm. This policy simply moves the robot arm back and forth in a waving motion without attempting to solve the task.

**Purpose:** Demonstrates the basic Policy API structure.

**Launch the evaluation environment:**
```bash
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

**Source:** [`WaveArm.py`](./aic_example_policies/ros/WaveArm.py)

---

### 2. CheatCode - Ground Truth Policy

![Cheat Code Policy](../../media/cheat_code_policy.gif)

A "cheating" solution that uses the TF transformation tree provided by the simulation when `ground_truth:=true` is set at launch time. This policy uses the poses of the plug and port to calculate target poses to send to `aic_controller`.

**Purpose:** Useful for training and debugging. Ground truth data will not be available during official evaluation.

**Launch simulation *with ground truth*:**
```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
```

**Source:** [`CheatCode.py`](./aic_example_policies/ros/CheatCode.py)

---

### 3. RunACT - ACT Policy

![Run ACT Policy](../../media/run_act_policy.gif)

A proof-of-concept implementation of a [LeRobot ACT](https://huggingface.co/docs/lerobot/en/act) (Action Chunking with Transformers) policy available on [HuggingFace](https://huggingface.co/grkw/aic_act_policy). This policy was trained on an NVIDIA RTX A5000 machine using `lerobot-train` with default parameters, on a small dataset collected using `lerobot-record` as explained in [`lerobot_robot_aic`](../aic_utils/lerobot_robot_aic/README.md#recording-training-data).

You may need to modify `pixi.toml` in order to run `lerobot` with your hardware setup. See [Troubleshooting](../docs/troubleshooting.md#nvidia-rtx-50xx-cards-not-supported-on-pytorch-version-locked-in-pixi). 

**Purpose:** Demonstrates integration of a trained neural network policy for the cable insertion task.

**Launch the evaluation environment:**
```bash
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.RunACT
```

**Source:** [`RunACT.py`](./aic_example_policies/ros/RunACT.py)

---

### 4. CustomACTPolicy - Local ACT Checkpoint

A standalone LeRobot ACT policy that loads a checkpoint from a filesystem path instead of downloading the reference HuggingFace model. It expects ACT checkpoints trained on the `bha-51/aic-dagger-data` schema:

- `observation.state`: 18 values with TCP position, TCP rotation in 6D representation, and task one-hots
- `observation.images.{left,center,right}_camera`: wrist camera images
- `action`: 9 values representing an absolute TCP pose target, `[x, y, z, rot6d_0..5]`

By default it uses:

```bash
/media/mbed/T7/datasets/aic/aic-dagger-data-outputs/train/act_aic_dagger_data/checkpoints/last/pretrained_model
```

**Run the policy:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CustomACTPolicy
```

**Run with a different checkpoint path:**
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CustomACTPolicy -p custom_act_policy_path:=/path/to/pretrained_model
```

**Source:** [`CustomACTPolicy.py`](./aic_example_policies/ros/CustomACTPolicy.py)

---

## Scoring Examples

For expected scoring results and reproducible test commands for each policy, see the [Scoring Test & Evaluation Guide](../../docs/scoring_tests.md).
