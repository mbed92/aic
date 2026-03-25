# Optimized Prompt (Clavix Enhanced)

*Optimized by Clavix on 2026-03-25. This version is ready for implementation.*

Fix a silent data-loss bug in `CompositeInsertCableStrategy` (`strategies/composite.py`): the refined pose computed in `ROS2ServiceApproach.execute()` is stored in a local variable and never returned, so insertion always receives the original coarse perception pose.

**Change `ApproachStrategy.execute()` signature** (`strategies/approach/base.py`) from `-> bool` to `-> Pose | None`, where `None` = approach failure and `Pose` = authoritative final position for the insertion stage (refined if available, otherwise original).

**Update `ROS2ServiceApproach.execute()`** (`strategies/approach/ros2_service.py`) to return `port_pose` at the end (which may have been replaced by `refined_pose` mid-method). The existing early-return `False` becomes `return None`.

**Update `CompositeInsertCableStrategy.execute()`** (`strategies/composite.py`):
```python
final_pose = self._approach.execute(port_pose, get_observation, move_robot)
if final_pose is None:
    send_feedback("Approach failed")
    return False
return self._insertion.execute(final_pose, get_observation, move_robot)
```

Invariant: when `_refine()` returns `None` (not yet implemented), `ROS2ServiceApproach` must still return the original `port_pose` — not `None`.

---

## Optimization Improvements Applied

1. **[Structured]** — Organized into: problem statement → signature change → implementation per file → invariant to preserve.
2. **[Clarified]** — Made explicit that `None` return = failure (not "no refinement"), to prevent misuse of the new signature.
3. **[Actionability]** — Added concrete code snippet for the composite caller so the change is unambiguous.
4. **[Completeness]** — Added the invariant for the `_refine() = None` case, which was implicit in discussion but critical for correctness.
5. **[Efficiency]** — Removed conversational framing; each paragraph targets one specific file change.

---
*Optimized by Clavix on 2026-03-25. This version is ready for implementation.*
