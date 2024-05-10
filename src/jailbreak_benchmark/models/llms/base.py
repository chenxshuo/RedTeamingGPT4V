# -*- coding: utf-8 -*-

"""Base Model."""

import logging
import torch

logger = logging.getLogger(__name__)


class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def __call__(self, prompt):
        return self.template.replace("{prompt}", prompt)


class LLMBaseModel(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.model_path = model_path

    def generate(self, instruction):
        raise NotImplementedError

    def forward(self, instruction):
        return self.generate(instruction)

    def __str__(self):
        raise NotImplementedError
