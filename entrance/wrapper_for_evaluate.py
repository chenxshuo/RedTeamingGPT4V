# -*- coding: utf-8 -*-

"""A wrapper on `minigpt_test_manual_prompts_visual_llm_for_wrapper.py` to do the eval in parallel."""

import logging
from datetime import datetime
import os
import argparse
import subprocess
import json
import pandas as pd
import csv
from jailbreak_benchmark.evaluation.eval_datasets import JailbreakVLDataset, JailbreakLMDataset
from jailbreak_benchmark.evaluation.evaluate import (
    calculate_asr_by_checking_refusal_words,
    calculate_asr_by_toxicity,
    calculate_asr_by_llama_guard,
)
from jailbreak_benchmark.models.vlms.vlm_inference_zoo import load_vlm
from jailbreak_benchmark.models.llms.llm_inference_zoo import load_lm
from jailbreak_benchmark.constants import TASK_LM, TASK_VL

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(f"--portions", type=int, default=1, help="Number of portions")
    parser.add_argument(f"--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--model_name", type=str, help="Name of the model to evaluate, i.e. target model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model directory for the target model",
    )
    parser.add_argument(
        "--model_lm_path",
        type=str,
        default="NoLMPath",
        help="Path to the model language model if needed",
    )
    parser.add_argument("--image_dir_path", type=str, help="Path to the image directory if needed")
    parser.add_argument(
        "--harmful_behaviors_json_path", type=str, help="Path to the harmful prompts json file"
    )
    parser.add_argument(
        "--task_type", type=str, required=True, help="vl or lm", choices=[TASK_LM, TASK_VL]
    )
    parser.add_argument(
        "--attack_method", type=str, required=True, help="Name of the attack method"
    )
    parser.add_argument(
        "--surrogate_model", type=str, required=True, help="Name of the surrogate model"
    )
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")
    args = parser.parse_args()
    return args


args = parse_args()

base_dir = (
    f"./results/{args.task_type}/{args.attack_method}/{args.model_name}/{args.surrogate_model}"
)
os.makedirs(base_dir, exist_ok=True)
result_name = (
    f"results-{args.task_type}-{args.attack_method}-source-{args.surrogate_model}-target-{args.model_name}-"
    + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    + ".json"
)


if args.task_type == TASK_VL:
    eval_dataset = JailbreakVLDataset(
        harmful_behaviors_json=args.harmful_behaviors_json_path,
    )
elif args.task_type == TASK_LM:
    eval_dataset = JailbreakLMDataset(
        harmful_behaviors_json=args.harmful_behaviors_json_path,
    )
else:
    raise ValueError(f"Unknown task type: {args.task_type}; should be {TASK_LM} or {TASK_VL}")

pids = []
for i in range(args.portions):
    logger.info(f"portion {i+1} / {args.portions}")
    start_index = i * len(eval_dataset) // args.portions
    end_index = (i + 1) * len(eval_dataset) // args.portions
    cmd = (
        f"CUDA_VISIBLE_DEVICES={i % args.num_gpus} python run_eval_for_wrapper.py "
        f"--model_name {args.model_name} "
        f"--model_path {args.model_path} "
        f"--model_lm_path {args.model_lm_path} "
        f"--harmful_behaviors_json_path {args.harmful_behaviors_json_path} "
        f"--task_type {args.task_type} "
        f"--attack_method {args.attack_method} "
        f"--surrogate_model {args.surrogate_model} "
        f"--seed {args.seed} "
        f"--start_index {start_index} "
        f"--end_index {end_index} "
        f"--portion_result_name {base_dir}/tmp_{result_name}_{i}.json"
    )
    logger.info(f"running command: {cmd}")
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    process = subprocess.Popen(cmd, shell=True, stdout=open("/dev/null", "w"))
    pid = process.pid
    pids.append(pid)

for pid in pids:
    os.waitpid(pid, 0)


# merge the results
final_results = {
    "task_type": args.task_type,
    "method": args.attack_method,
    "surrogate_model": args.surrogate_model,
    "target_model": args.model_name,
    "pairs": [],
}
success_count = 0
total_count = 0
for i in range(args.portions):
    res = json.load(open(f"{base_dir}/tmp_{result_name}_{i}.json"))
    final_results["pairs"].extend(res["pairs"])
    # os.remove(f"./results/tmp/tmp_{result_name}_{i}.json")
    # for pair in res["pairs"]:
    #     total_count += 1
    #     if pair["perspective_toxicity"] >= 0.5 or pair["flagged_openai"]:
    #         success_count += 1

# responses = [res["response"] for res in final_results["pairs"]]
# MODEL_PATH = "SebastianSchramm/LlamaGuard-7b-GPTQ-4bit-128g-actorder_True"
# model = load_lm(testing_model="Llama-Guard-4bit", model_path=MODEL_PATH)
# asr, label_list = calculate_asr_by_llama_guard(responses, model, return_judge=True, saving_tmp=True)
# for i, d in enumerate(final_results["pairs"]):
#     d["judge_by_llama_guard"] = label_list[i]
# # rate = calculate_asr_by_checking_refusal_words(responses)
# # rate = calculate_asr_by_toxicity(responses)
# # rate = success_count / total_count
# final_results["asr"] = asr
save_to = os.path.join(base_dir, result_name)
json.dump(final_results, open(save_to, "w"), indent=4)
logger.info(f"all results saved to {save_to}")
# logger.info(f"ASR: {rate}")
