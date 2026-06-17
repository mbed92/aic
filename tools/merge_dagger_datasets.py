from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

BASE = Path("/media/mbed/T7/datasets/aic/aic-dagger-data")
OUT_NAME = "aic_cheatcode_finealign"
OUT = Path(f"/media/mbed/T7/datasets/aic/{OUT_NAME}")

dataset_names = [
    "dagger_sc_138_cheatcode",
    "dagger_sc_372_cheatcode",
    "dagger_sc_493_cheatcode",
    "dagger_sc_85_cheatcode",
    "dagger_sfp_483_cheatcode",
    "dagger_sfp_753_cheatcode",
    "dagger_sfp_300_finealign_v3",
    "dagger_sfp_iter3_finealign"
]

datasets = []

for name in dataset_names:
    datasets.append(
        LeRobotDataset(
            repo_id=f"local/{name}",
            root=BASE / name,
        )
    )
    print(f"Loaded dataset {name} with {len(datasets[-1])} frames and {datasets[-1].num_episodes} episodes")

# Merge the datasets and save to a new local repository.
merged = merge_datasets(
    datasets=datasets,
    output_repo_id=OUT_NAME,
    output_dir=OUT,
)

expected_frames = sum(len(ds) for ds in datasets)
expected_episodes = sum(ds.num_episodes for ds in datasets)

# Sanity check the merged dataset.
assert len(merged) == expected_frames
assert merged.num_episodes == expected_episodes

# Try reloading the merged dataset to ensure it was saved correctly.
reloaded = LeRobotDataset(
    repo_id=f"local/{OUT_NAME}",
    root=OUT,
)

# Sanity check the reloaded dataset.
assert len(reloaded) == expected_frames
assert reloaded.num_episodes == expected_episodes

print("Merged dataset OK")
print("frames:", len(reloaded))
print("episodes:", reloaded.num_episodes)
print("root:", OUT)