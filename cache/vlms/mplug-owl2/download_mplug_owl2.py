# -*- coding: utf-8 -*-

"""TODO."""

import logging

logger = logging.getLogger(__name__)

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MAGAer13/mplug-owl2-llama2-7b",
    cache_dir="~/.cache/huggingface/hub",
)
