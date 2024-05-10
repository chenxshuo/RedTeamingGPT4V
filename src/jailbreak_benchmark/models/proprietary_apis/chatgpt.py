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

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(jailbreak_benchmark.__file__), "../../")
KEY_FILE = os.path.join(BASE_DIR, ".key")
KEY = json.load(open(KEY_FILE, "r"))["openAI"]
CLIENT = OpenAI(api_key=KEY)
GPT4_LANGUAGE_MODEL = "gpt-4-1106-preview"
GPT4_VISION_MODEL = "gpt-4-vision-preview"


def convert_image_format(input_image):
    TARGET_FORMAT = "png"
    # input_image = "../../datasets/visual_adversarial_example_our/surrogate_instructblip-vicuna-7b/bad_prompt_constrained_16.bmp"
    input_format = input_image.split(".")[-1]
    im = Image.open(input_image)
    rgb_im = im.convert("RGB")
    new_image_path = input_image.replace(input_format, TARGET_FORMAT)
    rgb_im.save(new_image_path)
    logger.info(f"Image saved successfully to {input_image.replace(input_format, TARGET_FORMAT)}")
    return new_image_path

class OpenaiModel:
    def __init__(self, model_name=GPT4_LANGUAGE_MODEL, add_system_prompt=True, sleep=5) -> None:
        self.model_name = model_name
        self.add_system_prompt = add_system_prompt
        self.sleep = sleep

    def fit_message(self, msg):
        if self.add_system_prompt:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg},
            ]
        else:
            conversation = [{"role": "user", "content": msg}]
        return conversation

    def inference_language(self, msg, max_tokens=128, **kwargs):
        while True:
            try:
                raw_response = CLIENT.chat.completions.create(
                    model=self.model_name,
                    messages=self.fit_message(msg),
                    max_tokens=max_tokens,
                    **kwargs,
                )
                time.sleep(self.sleep)
                return [str(m.message.content) for m in raw_response.choices]
            except Exception as e:
                logger.exception(f"Error in OpenAI API {e}. Retrying in 5 seconds.")
                time.sleep(5)

    # Function to encode the image
    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def fit_message_with_image(self, msg, img_path):
        # Getting the base64 string
        image_format = img_path.split(".")[-1]
        if image_format != "png":
            if os.path.exists(img_path.replace(image_format, "png")):
                img_path = img_path.replace(image_format, "png")
            else:
                logger.info(f"Converting image format from {image_format} to png.")
                img_path = convert_image_format(img_path)
                image_format = img_path.split(".")[-1]
        base64_image = self.encode_image(img_path)
        if self.add_system_prompt:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            },
                        },
                    ],
                },
            ]
        else:
            conversation = [{"role": "user", "content": msg}]
        return conversation

    def inference_with_image(self, msg, img_path, max_tokens=128, **kwargs):
        while True:
            try:
                raw_response = CLIENT.chat.completions.create(
                    model=self.model_name,
                    messages=self.fit_message_with_image(msg, img_path),
                    max_tokens=max_tokens,
                    **kwargs,
                )
                time.sleep(self.sleep)
                return [str(m.message.content) for m in raw_response.choices]
            except Exception as e:
                logger.exception(f"Error in OpenAI API {e}. Retrying in 5 seconds.")
                time.sleep(5)


def try_one():
    response = CLIENT.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    print(response.choices[0])