# -*- coding: utf-8 -*-

"""Refer to https://huggingface.co/THUDM/cogvlm-chat-hf ."""

from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# from lavis.models import load_model_and_preprocess

from .base import VLLMBaseModel

logger = logging.getLogger(__name__)


class VLLMCogVLMChat(VLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                "THUDM/cogvlm-chat-hf",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to("cuda")
            .eval()
        )

    def __str__(self):
        return "VLLMCogVLMChat"

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        image_path = images[0]
        image_pil = Image.open(image_path).convert("RGB")
        instruction = instruction[0] if type(instruction) == list else instruction
        text_prompt = instruction + "\n"

        inputs = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=text_prompt,
            history=[],
            images=[image_pil],
        )
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to("cuda"),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to("cuda"),
            "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            generation_text = self.tokenizer.decode(outputs[0])
            generation_text = generation_text.replace("</s>", "").strip()
        return generation_text
