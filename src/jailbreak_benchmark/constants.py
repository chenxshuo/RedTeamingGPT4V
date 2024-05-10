# -*- coding: utf-8 -*-

"""Constants used in this repo."""

import logging

logger = logging.getLogger(__name__)

# Jailbreak Attack Methods, LLM
Handcrafted = "Handcrafted"
GCG = "GCG"

JAILBREAK_ATTACK_METHODS_LLM = [
    Handcrafted,
    GCG,
]


# Jailbreak Attack Methods, VLM
CLEAN = "clean-img"
VisualAdv = "VisualAdv"
ImageHijacks = "ImageHijacks"
FigStep = "FigStep"
FigStepPro = "FigStepPro"
JailbreakPieces = "JailbreakPieces"

LP_16 = "lp16"
LP_32 = "lp32"
LP_64 = "lp64"
LP_UNCONSTRAINED = "unconstrained"

VisualAdv_LP_16 = f"{VisualAdv}-{LP_16}"
VisualAdv_LP_32 = f"{VisualAdv}-{LP_32}"
VisualAdv_LP_64 = f"{VisualAdv}-{LP_64}"
VisualAdv_LP_UNCONSTRAINED = f"{VisualAdv}-{LP_UNCONSTRAINED}"

ImageHijacks_LP_16 = f"{ImageHijacks}-{LP_16}"
ImageHijacks_LP_32 = f"{ImageHijacks}-{LP_32}"
ImageHijacks_LP_64 = f"{ImageHijacks}-{LP_64}"
ImageHijacks_LP_UNCONSTRAINED = f"{ImageHijacks}-{LP_UNCONSTRAINED}"

JAILBREAK_ATTACK_METHODS_VLM = [
    CLEAN,
    VisualAdv_LP_16,
    VisualAdv_LP_UNCONSTRAINED,
    ImageHijacks_LP_16,
    ImageHijacks_LP_UNCONSTRAINED,
    FigStep,
    FigStepPro,
    JailbreakPieces,
]

# VLMs used in this repo
# variable names are all uppercase, use '_' to connect words
# strings are all lowercase, use '_' to connect words
MINIGPT4_7B = "minigpt4-7b"
MINIGPT4_13B = "minigpt4-13b"
LLAVA_LLAMA2_13B = "llava-llama2-13b"
LLAVA_LLAMA2_7B = "llava-llama2-7b"
INSTRUCTBLIP_VICUNA_13B = "instructblip-vicuna-13b"
INSTRUCTBLIP_VICUNA_7B = "instructblip-vicuna-7b"
LLAMAADAPTERV2 = "llamaadapterv2"
MPLUGOWL2 = "mplug-owl-2"
COGVLM = "cogvlm"
FUYU = "fuyu"

# LLMs used in this repo

TASK_VL = "vl"
TASK_LM = "lm"
