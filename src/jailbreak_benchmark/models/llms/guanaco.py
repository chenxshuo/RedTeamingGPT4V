# -*- coding: utf-8 -*-

"""Vicuna."""

import logging
from .base import LLMBaseModel

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


CONV_TEMPLATE = """
### Human: {prompt}
### Assistant:
"""


class LLMGuanaco(LLMBaseModel):
    def __init__(self, model_path, model_type="", device="cuda:0"):
        super().__init__(model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        assert model_type != "", "Please specify the model type"
        self.model_type = model_type

    def generate(
        self,
        instruction,
        do_sample=True,
        max_new_tokens=300,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.05,
        length_penalty=1,
        temperature=1.0,
    ):
        """
        instruction: (str) a string of instruction
        Return: (str) a string of generated response
        """
        inputs = CONV_TEMPLATE.replace("{prompt}", instruction)
        inputs = self.tokenizer(inputs, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        output_text = output_text.split("Assistant:")[-1].strip()
        return output_text

    def __str__(self):
        return f"LLMGuanaco-{self.model_type}"
