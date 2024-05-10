# -*- coding: utf-8 -*-

"""
Run the MiniGPT4-13B model.
```
deactivate
source minigpt4-jailbreak/bin/activate
cd entrance
python inference/minigpt4/run_minigpt4.py
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

# 7B to run today
MODEL_PATH = ""
MINIGPT4_7B_VICUNA = ""
model = load_vlm(testing_model="MiniGPT4-7B", model_path=MODEL_PATH, lm_path=MINIGPT4_7B_VICUNA)


pred = model.generate(
    instruction=[instruction],
    images=[image_path],
)
print(f"image path:\t{image_path}")
print(f"Instruction:\t{instruction}")
print(f"Answer:\t{pred}")
print("-" * 20)
