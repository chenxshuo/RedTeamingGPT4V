"""
Refer to https://huggingface.co/adept/fuyu-8b
"""

from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import FuyuForCausalLM, AutoTokenizer, FuyuProcessor, FuyuImageProcessor
import logging

# from lavis.models import load_model_and_preprocess

from .base import VLLMBaseModel

logger = logging.getLogger(__name__)


class VLLMFuyu(VLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(
            image_processor=self.image_processor, tokenizer=self.tokenizer
        )
        self.model = FuyuForCausalLM.from_pretrained(model_path, device_map=device)
        self.device = device

    def __str__(self):
        return "VLLMFuyu"

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        image_path = images[0]  # instruct_blip2 only supports single image
        image_pil = Image.open(image_path).convert("RGB")
        # test inference
        instruction = instruction[0] if type(instruction) == list else instruction
        text_prompt = instruction + "\n"
        # __import__("ipdb").set_trace()
        model_inputs = self.processor(text=text_prompt, images=[image_pil], device=self.device).to(
            self.device
        )
        text_input_len = len(text_prompt)

        generation_output = self.model.generate(**model_inputs, max_new_tokens=16)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ].strip()
        generation_text = (
            generation_text.split("<s>")[-1][text_input_len:].replace("\n\u0004", "").strip()
        )
        # __import__("ipdb").set_trace()
        return generation_text
