# -*- coding: utf-8 -*-

"""TODO."""

import logging

logger = logging.getLogger(__name__)

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="THUDM/cogvlm-chat-hf",
    cache_dir="~/.cache/huggingface/hub",
)
