from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Neph0s/CoSER",
    repo_type="dataset",
    local_dir="/Users/charliecyli/Downloads/CoSER",
)