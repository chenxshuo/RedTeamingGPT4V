# -*- coding: utf-8 -*-

"""Sub function for wrapper."""

import logging
import argparse
from jailbreak_benchmark.evaluation.evaluate import evaluate_entrance
from jailbreak_benchmark.constants import TASK_LM, TASK_VL


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)


def arg_parse():
    parser = argparse.ArgumentParser(description="Evaluate a model on a task")
    parser.add_argument("--model_name", type=str, help="Name of the model to evaluate")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model, could be the config file or the model directory",
    )
    parser.add_argument(
        "--model_lm_path", type=str, default="NoLMPath", help="Path to the model language model"
    )
    parser.add_argument(
        "--harmful_behaviors_json_path", type=str, help="Path to the harmful prompts json file"
    )
    parser.add_argument(
        "--task_type", type=str, help="Name of the task", required=True, choices=[TASK_LM, TASK_VL]
    )
    parser.add_argument(
        "--attack_method", type=str, required=True, help="Name of the attack method"
    )
    parser.add_argument(
        "--surrogate_model", type=str, required=True, help="Name of the surrogate model"
    )
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")
    parser.add_argument(
        "--start_index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--end_index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--portion_result_name",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


args = arg_parse()

responses = evaluate_entrance(
    target_model_name=args.model_name,
    model_path=args.model_path,
    model_lm_path=args.model_lm_path,
    harmful_behaviors_json_path=args.harmful_behaviors_json_path,
    task_type=args.task_type,
    attack_method=args.attack_method,
    surrogate_model_name=args.surrogate_model,
    seed=42,
    batch_mode=False,
    start_index=args.start_index,
    end_index=args.end_index,
    result_name=args.portion_result_name,
)
# logger.info(f"Rate: {rate}")
logger.info(f"Saved to {args.portion_result_name}")
