"""Regression checks for the task conditioning used by local checkpoints."""

import unittest

import numpy as np
from aic_task_interfaces.msg import Task

from aic_example_policies.ros.implementations.utils import task_to_one_hots


class TaskEncodingTest(unittest.TestCase):
    def test_sc_targets_match_training_schema(self):
        for index in range(2):
            with self.subTest(target=index):
                task = Task(
                    plug_type="sc",
                    target_module_name=f"sc_port_{index}",
                    port_name="sc_port_base",
                )
                cable, rail, port = task_to_one_hots(task)
                np.testing.assert_array_equal(cable, [0, 1])
                np.testing.assert_array_equal(rail, np.eye(5)[index])
                np.testing.assert_array_equal(port, [1, 0])

    def test_sfp_targets_keep_existing_encoding(self):
        for rail_index in range(5):
            for port_index in range(2):
                with self.subTest(rail=rail_index, port=port_index):
                    task = Task(
                        plug_type="sfp",
                        target_module_name=f"nic_card_mount_{rail_index}",
                        port_name=f"sfp_port_{port_index}",
                    )
                    cable, rail, port = task_to_one_hots(task)
                    np.testing.assert_array_equal(cable, [1, 0])
                    np.testing.assert_array_equal(rail, np.eye(5)[rail_index])
                    np.testing.assert_array_equal(port, np.eye(2)[port_index])


if __name__ == "__main__":
    unittest.main()
