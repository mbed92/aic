from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

BASE = Path("/media/mbed/T7/datasets/aic/aic-dagger-data")

dataset_names = [
    "dagger_sc_138_cheatcode",
    "dagger_sc_372_cheatcode",
    "dagger_sc_493_cheatcode",
    "dagger_sc_85_cheatcode",
    "dagger_sfp_483_cheatcode",
    "dagger_sfp_753_cheatcode",
    "dagger_sfp_300_finealign_v3",
    "dagger_sfp_iter3_finealign",
]

datasets = []

for name in dataset_names:
    datasets.append(
        LeRobotDataset(
            repo_id="local",
            root=BASE / name,
        )
    )

merged = merge_datasets(
    datasets=datasets,
    output_repo_id="aic_cheatcode_finealign",
    output_dir="/media/mbed/T7/datasets/aic/aic_cheatcode_finealign",
)

print(len(merged))