from openai import OpenAI
import logging
import time
import os
import json
import base64
import jailbreak_benchmark
from PIL import Image
import glob
from tqdm import tqdm
from datetime import datetime
from jailbreak_benchmark.models.proprietary_apis.chatgpt import OpenaiModel
from jailbreak_benchmark.constants import (
    LP_16,
    LP_32,
    LP_UNCONSTRAINED,
    CLEAN,
    VisualAdv_LP_16,
    VisualAdv_LP_32,
    VisualAdv_LP_UNCONSTRAINED,
    VisualAdv_LP_UNCONSTRAINED,
    FigStep,
    ImageHijacks,
    ImageHijacks_LP_16,
    ImageHijacks_LP_32,
    ImageHijacks_LP_UNCONSTRAINED,
    MINIGPT4_7B,
    MINIGPT4_13B,
    LLAVA_LLAMA2_13B,
    INSTRUCTBLIP_VICUNA_13B,
    INSTRUCTBLIP_VICUNA_7B,
    LLAVA_LLAMA2_7B
)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(jailbreak_benchmark.__file__), "../../")
# KEY_FILE = os.path.join(BASE_DIR, ".key")
# KEY = json.load(open(KEY_FILE, "r"))["openAI"]
# CLIENT = OpenAI(api_key=KEY)
GPT4_LANGUAGE_MODEL = "gpt-4-1106-preview"
GPT4_VISION_MODEL = "gpt-4-vision-preview"

JAILBREAK_METHODS = f"{BASE_DIR}/datasets/jailbreak_attacks.json"
jailbreak_methods_file = json.load(open(JAILBREAK_METHODS, "r"))
handcrafted_methods = list(jailbreak_methods_file.keys())


def try_vision():
    one_img = os.path.join(
        BASE_DIR,
        "./datasets/visual_adversarial_example_our/surrogate_instructblip-vicuna-7b/bad_prompt_constrained_16.bmp",
    )
    msg = "What is in this image?"
    gpt4v = OpenaiModel(model_name=GPT4_VISION_MODEL)
    response = gpt4v.inference_with_image(msg=msg, img_path=one_img)
    print(response)


def main(method, model_name=GPT4_VISION_MODEL, source_model="None", json_file=None):
    existing_record = []

    if model_name == GPT4_VISION_MODEL:
        gpt4v = OpenaiModel(model_name=model_name)
        if json_file is None:
            if method == "FigStep":
                harmful_json = json.load(
                    open(
                        os.path.join(
                            BASE_DIR,
                            "./datasets/json_for_vlm_our/harmful_behaviors_filtered-FigStep.json",
                        ),
                        "r",
                    )
                )
            if method == "VisualAdv-lp16-our":
                harmful_json = json.load(
                    open(
                        os.path.join(
                            BASE_DIR,
                            f"./datasets/json_for_vlm_our/harmful_behaviors_filtered-VisualAdv-lp16-our-{source_model}.json",
                        ),
                        "r",
                    )
                )
            else:
                raise NotImplementedError
        else:
            harmful_json = json.load(open(json_file, "r"))
        source_model = source_model
        result_dir = os.path.join(
            BASE_DIR, f"./entrance/results/vl/{method}/{GPT4_VISION_MODEL}/{source_model}"
        )
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        ans = {
            "task_type": "vl",
            "method": method,
            "surrogate_model": source_model,
            "target_model": GPT4_VISION_MODEL,
            "pairs": [],
        }
        ans_name = os.path.join(
            result_dir,
            f"results-vl-{method}-source-{source_model}-target-{GPT4_VISION_MODEL}-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.json",
        )
        existing_results = glob.glob(os.path.join(result_dir, "*.json"))
        if len(existing_results) > 0:
            existing_results.sort(key=lambda x: os.path.getctime(x))
            last_result = existing_results[-1]
            existing_record = [
                b["harmful_behavior"] + b["img"] for b in json.load(open(last_result, "r"))["pairs"]
            ]
            ans["pairs"] = json.load(open(last_result, "r"))["pairs"]

        for one_record in tqdm(
            harmful_json["harmful_behaviors"], desc=f"running inference on {model_name}"
        ):
            behavior = one_record["behavior"]
            img_path = one_record["image"]
            if behavior + img_path in existing_record:
                continue
            response = gpt4v.inference_with_image(msg=behavior, img_path=img_path)
            ans["pairs"].append(
                {"harmful_behavior": behavior, "response": response, "img": img_path}
            )
            json.dump(ans, open(ans_name, "w"), indent=4)
    elif model_name == GPT4_LANGUAGE_MODEL:
        gpt4l = OpenaiModel(model_name=model_name)
        if method == "AutoDan":
            harmful_json = json.load(
                open(
                    os.path.join(
                        BASE_DIR,
                        f"./datasets/json_for_lm_autodan/harmful_behaviors_filtered-{source_model}-AutoDan.json",
                    )
                )
            )
        elif method == "GCG":
            harmful_json = json.load(
                open(
                    os.path.join(
                        BASE_DIR,
                        f"./datasets/json_for_lm_suffix/harmful_behaviors_filtered-{source_model}-GCG.json",
                    )
                )
            )
        elif method == "clean":
            harmful_json = json.load(
                open(
                    os.path.join(
                        BASE_DIR,
                        f"./datasets/json_for_lm/harmful_behaviors_filtered-clean.json",
                    )
                )
            )
        elif method in handcrafted_methods:
            harmful_json = json.load(open(json_file, "r"))
        else:
            raise NotImplementedError
        source_model = source_model
        result_dir = os.path.join(
            BASE_DIR, f"./entrance/results/lm/{method}/{model_name}/{source_model}"
        )
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        ans = {
            "task_type": "vl",
            "method": method,
            "surrogate_model": source_model,
            "target_model": model_name,
            "pairs": [],
        }
        ans_name = os.path.join(
            result_dir,
            f"results-lm-{method}-source-{source_model}-target-{model_name}-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.json",
        )

        existing_results = glob.glob(os.path.join(result_dir, "*.json"))
        if len(existing_results) > 0:
            existing_results.sort(key=lambda x: os.path.getctime(x))
            last_result = existing_results[-1]
            existing_record = [
                b["instruction_with_behavior"] for b in json.load(open(last_result, "r"))["pairs"]
            ]
            ans["pairs"] = json.load(open(last_result, "r"))["pairs"]

        for one_record in tqdm(
            harmful_json["harmful_behaviors"], desc=f"running inference on {model_name}"
        ):
            behavior = one_record["instruction_with_behavior"]
            if behavior in existing_record:
                continue
            response = gpt4l.inference_language(msg=behavior)
            ans["pairs"].append(
                {"instruction_with_behavior": behavior, "response": response}
            )
            json.dump(ans, open(ans_name, "w"), indent=4)


if __name__ == "__main__":
    # main(method="FigStep", model_name=GPT4_VISION_MODEL)
    # main(method="VisualAdv-lp16-our", model_name=GPT4_VISION_MODEL, source_model="minigpt4-7b")
    # main(method="AutoDan", model_name=GPT4_LANGUAGE_MODEL, source_model="Guanaco7B")
    # main(method="AutoDan", model_name=GPT4_LANGUAGE_MODEL, source_model="Llama2_7B")
    # main(method="AutoDan", model_name=GPT4_LANGUAGE_MODEL, source_model="Vicuna7B")
    # main(method="GCG", model_name=GPT4_LANGUAGE_MODEL, source_model="Guanaco7B")
    # main(method="GCG", model_name=GPT4_LANGUAGE_MODEL, source_model="Llama2_7B")
    # main(method="GCG", model_name=GPT4_LANGUAGE_MODEL, source_model="Vicuna7B")
    # main(method="GCG", model_name=GPT4_LANGUAGE_MODEL, source_model="Vicuna7B_Guanaco7B")
    #
    # main(method="clean", model_name=GPT4_LANGUAGE_MODEL, source_model="clean")

    # for one_method in tqdm(handcrafted_methods, desc="running inference on handcrafted methods"):
    #     p = os.path.join(
    #         BASE_DIR,
    #         f"./datasets/subset_json_for_lm/harmful_behaviors_filtered_subset_0.3-{one_method}.json"
    #     )
    #     if os.path.exists(p):
    #         print(f"running inference on {one_method}")
    #         main(method=one_method, model_name=GPT4_LANGUAGE_MODEL, source_model="None", json_file=p)
    METHODS = [
        CLEAN,
        VisualAdv_LP_16 + "-our",
        VisualAdv_LP_32 + "-our",
        VisualAdv_LP_UNCONSTRAINED + "-our",
        ImageHijacks_LP_16 + "-our",
        ImageHijacks_LP_32 + "-our",
        ImageHijacks_LP_UNCONSTRAINED + "-our",
        FigStep
    ]
    SOURCE_MODELS = [
        MINIGPT4_7B,
        # MINIGPT4_13B,
        # LLAVA_LLAMA2_13B,
        # INSTRUCTBLIP_VICUNA_13B
        INSTRUCTBLIP_VICUNA_7B
        # LLAVA_LLAMA2_7B
    ]
    for one_json in os.listdir(os.path.join(BASE_DIR, "./datasets/subset_json_for_vl")):
        if one_json.endswith(".json") and "outdated" not in one_json:
            for m in METHODS:
                if m in one_json:
                    method = m
                    break
            for s in SOURCE_MODELS:
                if s in one_json:
                    source_model = s
                    break
            print(f"method: {method}, source_model: {source_model}, json_file: {one_json}")
            main(method=method, model_name=GPT4_VISION_MODEL, source_model=source_model,
                 json_file=os.path.join(BASE_DIR, "./datasets/subset_json_for_vl/", one_json))
