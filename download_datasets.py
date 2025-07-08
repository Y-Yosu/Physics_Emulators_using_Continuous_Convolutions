from huggingface_hub import snapshot_download
import os

base_path = "datasets/SFBC"
datasets = {
    "SFBC_dataset_I": "Wi-Re/SFBC_dataset_I",
    "SFBC_dataset_II": "Wi-Re/SFBC_dataset_II",
    "SFBC_dataset_III": "Wi-Re/SFBC_dataset_III",
    "SFBC_dataset_IV": "Wi-Re/SFBC_dataset_IV",
}

os.makedirs(base_path, exist_ok=True)

for folder_name, repo_id in datasets.items():
    target_dir = os.path.join(base_path, folder_name)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",  # <-- this is the key fix
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f"✅ Downloaded {repo_id} to {target_dir}")