from huggingface_hub import snapshot_download

# LLaMA-7B original version from Meta
snapshot_download(
    repo_id="nyanko7/LLaMA-7B",
    cache_dir="~/.cache/huggingface/hub",
)
