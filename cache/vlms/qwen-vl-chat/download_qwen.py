# -*- coding: utf-8 -*-

"""TODO."""

import logging

logger = logging.getLogger(__name__)

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen-VL-Chat",
    cache_dir="~/.cache/huggingface/hub",
)
