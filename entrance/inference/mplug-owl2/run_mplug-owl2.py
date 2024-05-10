# -*- coding: utf-8 -*-

"""
Run the mPLUGOwl2  model.
```
deactivate
source mplug-owl2-jailbreak/bin/activate
cd entrance
python inference/mplug-owl2/run_mplug-owl2.py
```

"""

import logging
import argparse
import os

from jailbreak_benchmark.models.vlms import load_vlm


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

image_path = "../cache/test_cases/clean.jpeg"
instruction = (
    "Create an English slogan for the tags on the hummer focusing on the meaning of the word."
)

# mPLUGOwl2
MODEL_PATH = ""
model = load_vlm(testing_model="mPLUGOwl2", model_path=MODEL_PATH)


pred = model.generate(
    instruction=[instruction],
    images=[image_path],
)
print(f"image path:\t{image_path}")
print(f"Instruction:\t{instruction}")
print(f"Answer:\t{pred}")
print("-" * 20)
