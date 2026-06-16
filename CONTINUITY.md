# CONTINUITY

## Snapshot
Goal: Keep AIC policy experiments in `aic_example_policies` and align `CustomACTPolicy` with the current local dataset format.
Current state: `CustomACTPolicy.py` targets the `bha-51/aic-dagger-data` ACT schema: 18D task-conditioned state, three 512x576 camera streams, and 9D absolute TCP pose actions.
Next step: Run policy-level smoke checks against a real trained checkpoint path when one is available.
Success criteria: `CustomACTPolicy` imports, builds the 18D state from ROS observations/tasks, converts 9D rot6d actions to pose targets, and sends Cartesian `MODE_POSITION` updates.

## Invariants
- Work from the nearest relevant module or subproject.
- Prefix shell commands with `rtk` by default.
- Avoid simulator or hardware-facing validation unless explicitly requested.

## Decisions
- `CustomACTPolicy` remains an ACT checkpoint adapter, not a diffusion-policy adapter.
- `CustomACTPolicy` does not preserve compatibility with the old 46D/contact/twist checkpoint format.
- `bha-51/aic-dagger-data` task target encoding is baked into `observation.state` via cable, rail, and port one-hots.

## State

### Done (recent)
- Added a concise root `README.md` Lessons section summarizing AIC participant learnings: causal confusion checks, data diversity, ACT chunk/observation tuning, cheatcode limits, state-order validation, Diffusion Policy comparison, and inference latency risk.
- Added `CustomACTPolicy.py` under `aic_example_policies/aic_example_policies/ros/`.
- Updated `CustomACTPolicy` for `aic-dagger-data`: `observation.state` shape `[18]`, image features `observation.images.{left,center,right}_camera`, and action shape `[9]`.
- Added quaternion/rot6d conversion helpers and action-to-`Pose` conversion for absolute TCP pose targets.
- Updated `aic_example_policies/README.md` with the new `CustomACTPolicy` schema and checkpoint path behavior.
- Resolved `TODO(codex)` items in `CustomACTPolicy`: checkpoint path now supports `CUSTOM_ACT_POLICY_PATH`, VS Code `Policy: start` supplies that env var, and camera feature mapping uses an explicit resolver.
- Simplified `CustomACTPolicy` camera mapping to only the three canonical dataset sources: `left_camera`, `center_camera`, and `right_camera`.
- Removed the code fallback checkpoint path from `CustomACTPolicy`; `CUSTOM_ACT_POLICY_PATH` is required unless a path is passed directly to the constructor.
- Renamed VS Code task `Policy: start` to `Policy: start CustomACT` and hardcoded it to `aic_example_policies.ros.CustomACTPolicy`.

### Now
- `CustomACTPolicy.py` is present in the working tree and should be treated as part of the active local policy package.

### Next
- Verify with a real ACT checkpoint at `custom_act_policy_path`.

## Open Questions
- Whether the default checkpoint path exists on every target development machine.

## Working Set
- `README.md`
- `aic_example_policies/aic_example_policies/ros/CustomACTPolicy.py`
- `aic_example_policies/README.md`
- `CONTINUITY.md`

## Incidents
- Previous work was interrupted while `CustomACTPolicy.py` was still untracked; repository state is authoritative and the file is retained as current work.

## Receipts
- `rtk pixi run python -m py_compile aic_example_policies/aic_example_policies/ros/CustomACTPolicy.py` passed.
- `rtk pixi run env PYTHONPATH=/home/mbed/Projects/aic/aic_example_policies python -c "import importlib; importlib.import_module('aic_example_policies.ros.CustomACTPolicy'); print('import ok')"` passed.
- A no-simulator helper smoke test passed for SFP task one-hots, 18D state construction, quaternion/rot6d round-trip, 9D action-to-`Pose`, and Cartesian `MODE_POSITION` `MotionUpdate`.
- `rtk rg -n "TODO\\(codex\\)|TODO" aic_example_policies/aic_example_policies/ros/CustomACTPolicy.py` returned no matches after resolving the TODOs.
- `rtk jq empty .vscode/tasks.json` passed after adding `CUSTOM_ACT_POLICY_PATH` to `Policy: start`.
- `rtk pixi run python -m py_compile aic_example_policies/aic_example_policies/ros/CustomACTPolicy.py`, module import, and image resolver smoke checks passed after removing camera aliases.
- A no-env smoke check confirmed `_resolve_policy_path()` raises before loading when `CUSTOM_ACT_POLICY_PATH` is not set.
- `rtk jq empty .vscode/tasks.json` passed after hardcoding `Policy: start CustomACT` and removing the `policyClass` input.
