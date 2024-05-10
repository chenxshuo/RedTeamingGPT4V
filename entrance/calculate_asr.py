# -*- coding: utf-8 -*-

"""Calculate ASR on generated responses."""
import glob
import logging
import os.path
import time
import json
import pandas as pd

from jailbreak_benchmark.evaluation.evaluate import (
    calculate_asr_by_checking_refusal_words,
    calculate_asr_by_llama_guard,
)
from jailbreak_benchmark.models.llms.llm_inference_zoo import load_lm

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)


TASK_TYPES = ["vl", "lm"]
ATTACK_METHODS = [
    # "clean-img",
    # "FigStep",
    # "ImageHijacks-lp16-our",
    # "ImageHijacks-lp32-our",
    # "ImageHijacks-unconstrained-our",
    # "VisualAdv-lp16-our",
    "VisualAdv-lp32-our",
    "VisualAdv-unconstrained-our",
    # "VisualAdv-lp16",
    # "VisualAdv-unconstrained",
]

TARGET_MODELS = [
    # "gpt-4-vision-preview"
    "MiniGPT4-7B",
    "LLaVAv1.5-7B",
    "InstructBLIP-7B",
    "CogVLM",
    "Fuyu",
    "Qwen-VL-Chat",
    # "InstructBLIP",
    # "LlamaAdapterV2",
    # "LLaVA_llama2-13B",
    # "MiniGPT4-13B",
    # "mPlugOwl2",
]

SURROGATE_MODELS = [
    # "None",
    # "instructblip-vicuna-7b",
    # "instructblip-vicuna-13b",
    # "llava-llama2-13b",
    # "minigpt4-13b",
    "minigpt4-7b",
]

BASE_DIR = ""
MODEL_PATH = "SebastianSchramm/LlamaGuard-7b-GPTQ-4bit-128g-actorder_True"
model = load_lm(testing_model="Llama-Guard-4bit", model_path=MODEL_PATH)

for surrogate_model in SURROGATE_MODELS:
    result_table = pd.DataFrame(
        columns=["Target Model"] + ATTACK_METHODS,
    )
    for target_model in TARGET_MODELS:
        new_row = {"Target Model": target_model}
        for method in ATTACK_METHODS:
            target_model_base_dir = os.path.join(BASE_DIR, method, target_model, surrogate_model)
            if os.path.exists(target_model_base_dir):
                all_results = glob.glob(os.path.join(target_model_base_dir, "results-*.json"))
                all_results.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                result = all_results[0]
                logger.info(f"loading results from {result}")
                with open(os.path.join(target_model_base_dir, result), "r") as f:
                    data = json.load(f)
                responses = [
                    (
                        p["harmful_behavior"],
                        p["response"][0] if type(p["response"]) == list else p["response"],
                    )
                    for p in data["pairs"]
                ]
                if len(responses) < 1467:
                    logger.info(f"only {len(responses)} responses")
                    # continue
                logger.info(
                    f"computing ASR for target model {target_model} and attack method {method}"
                    f" with surrogate model {surrogate_model}"
                )
                time_start = time.time()

                asr, label_list = calculate_asr_by_llama_guard(
                    responses, model, return_judge=True, saving_tmp=False
                )
                for i, d in enumerate(data["pairs"]):
                    d["judge_by_llama_guard"] = label_list[i]
                with open(
                    os.path.join(target_model_base_dir, "judge-by-llama-guard.json"), "w"
                ) as f:
                    json.dump(data, f, indent=4)
                logger.info(
                    f"Dumped results-with-judge-by-llama-guard.json in {target_model_base_dir}"
                )
                time_end = time.time()

                asr_by_refusal_words = calculate_asr_by_checking_refusal_words(
                    [r[1] for r in responses]
                )
                logger.info(f"ASR by refusal words: {asr_by_refusal_words:.4f}")
                new_row[method] = f"{asr*100:.2f}%; {asr_by_refusal_words * 100:.2f}%"
                # new_row[method] = f"{asr_by_refusal_words * 100:.2f}%"
        # insert row into the DataFrame, do not use append
        result_table = result_table._append(new_row, ignore_index=True)
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    result_table.to_csv(f"llama-guard-4bit-asr-vl-visualadv-{surrogate_model}-{date_time}.csv", index=False)
    logger.info(
        f"ASR table saved llama-guard-4bit-asr-vl-imagehijacks-{surrogate_model}-{date_time}.csv; time spent: {time_end - time_start}"
    )
