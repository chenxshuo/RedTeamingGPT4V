import os, sys
import yaml
import jailbreak_benchmark.models.vlms as vlms

sys.path.append(os.getcwd())


def setup_lm_weights_path(model_name, lm_path):
    """
    Set up the path to the language model weights in the model config file.
    Needed in MiniGPT4 models.
    """
    if model_name == "MiniGPT4-13B":
        lm_config_path = os.path.join(
            os.path.abspath(
                vlms.__file__,
            ).split(
                "__init__.py"
            )[0],
            "minigpt4_utils",
            "configs",
            "models",
            "minigpt4_13b.yaml",
        )
        lm_config = yaml.load(open(lm_config_path, "r"), Loader=yaml.FullLoader)
        if lm_config["model"]["llama_model"] == lm_path:
            return
        lm_config["model"]["llama_model"] = lm_path
        yaml.dump(lm_config, open(lm_config_path, "w"))
    else:
        raise NotImplementedError


def load_vlm(testing_model, model_path=None, lm_path=None):
    # if testing_model == "Flamingo":
    #     from openflamingo_modeling import VLLMFlamingo
    #     ckpt_path = f'{cache_dir}/open_flamingo_9b_hf'
    #     model = VLLMFlamingo(ckpt_path, clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai", device="cuda:0")
    if testing_model == "MiniGPT4-7B":  ## mmqads
        from .minigpt4_modeling import VLLMMiniGPT4

        # model = VLLMMiniGPT4(f"{cache_dir}/minigpt4/minigpt4_eval_7b.yaml")
        assert (
            "7b" in model_path
        ), "Model path does not contain 7b, are your sure you are using the right model config?"
        model = VLLMMiniGPT4(model_path=model_path, model_type="minigpt4-7b")
    elif testing_model == "MiniGPT4v2":
        from .minigpt4_modeling import VLLMMiniGPT4

        # model = VLLMMiniGPT4(f"{cache_dir}/minigpt4/minigptv2_eval.yaml")
        assert (
            "v2" in model_path
        ), "Model path does not contain v2, are your sure you are using the right model config?"
        model = VLLMMiniGPT4(model_path=model_path, model_type="minigpt4-v2")
    elif testing_model == "MiniGPT4-llama2":  ## mmqads
        from .minigpt4_modeling import VLLMMiniGPT4

        assert (
            "llama2" in model_path
        ), "Model path does not contain llama2, are your sure you are using the right model config?"
        # model = VLLMMiniGPT4(f"{cache_dir}/minigpt4/minigpt4_eval_llama2.yaml")
        model = VLLMMiniGPT4(model_path=model_path, model_type="minigpt4-llama2")
    elif testing_model == "MiniGPT4-13B":  ## mmqads
        from .minigpt4_modeling import VLLMMiniGPT4

        if lm_path is not None:
            setup_lm_weights_path(testing_model, lm_path)
        assert (
            "13b" in model_path
        ), "Model path does not contain 13b, are your sure you are using the right model config?"
        # model = VLLMMiniGPT4(f"{cache_dir}/minigpt4/minigpt4_eval_13b.yaml")
        model = VLLMMiniGPT4(model_path=model_path, model_type="minigpt4-13b")
    elif testing_model == "LLaVA-7B":  ## mmqa
        from .llava_modeling import VLLMLLaVA

        assert (
            "llava-7b" in model_path
        ), "Model path does not contain llava-7b, are your sure you are using the right model config?"
        model = VLLMLLaVA(
            # {"model_path": f"{cache_dir}/llava/llava-7b", "device": 0, "do_sample": True}
            {"model_path": model_path, "device": 0, "do_sample": True},
            model_type="llava-7b",
        )
    elif testing_model == "LLaVA_llama2-13B":  ## mmqa
        from .llava_modeling import VLLMLLaVA

        assert (
            "llava-llama-2-13b" in model_path
        ), "Model path does not contain llava-llama-2-13b, are your sure you are using the right model config?"
        model = VLLMLLaVA(
            {
                # "model_path": f"{cache_dir}/llava/llava-llama-2-13b-chat",
                "model_path": model_path,
                "device": 0,
                "do_sample": True,
            },
            model_type="llava-llama2-13b",
        )
    elif testing_model == "LLaVA_llama2-7B-LoRA":  ## mmqa
        from .llava_modeling import VLLMLLaVA

        assert (
            "llava-llama-2-7b" in model_path
        ), "Model path does not contain llava-llama-2-7b, are your sure you are using the right model config?"
        model = VLLMLLaVA(
            {
                # "model_path": f"{cache_dir}/llava/llava-llama-2-13b-chat",
                "model_path": model_path,
                "device": 0,
                "do_sample": True,
            },
            model_type="llava-llama2-7b",
        )

    elif testing_model == "LLaVAv1.5-7B":
        from .llava_modeling import VLLMLLaVA

        assert (
            "llava-v1.5-7b" in model_path
        ), "Model path does not contain llava-v1.5-7b, are your sure you are using the right model config?"
        model = VLLMLLaVA(
            # {"model_path": f"{cache_dir}/llava/llava-v1.5-7b", "device": 0, "do_sample": False}
            {"model_path": model_path, "device": 0, "do_sample": False},
            model_type="llava-v1.5-7b",
        )
    elif testing_model == "LLaVAv1.5-13B":
        from .llava_modeling import VLLMLLaVA

        assert (
            "llava-v1.5-13b" in model_path
        ), "Model path does not contain llava-v1.5-13b, are your sure you are using the right model config?"
        model = VLLMLLaVA(
            # {"model_path": f"{cache_dir}/llava/llava-v1.5-13b", "device": 0, "do_sample": False}
            {"model_path": model_path, "device": 0, "do_sample": False},
            model_type="llava-v1.5-13b",
        )

    elif testing_model == "LlamaAdapterV2":  ## mmqa
        from .llama_adapter_v2_modeling import VLLMLlamaAdapterV2

        model = VLLMLlamaAdapterV2(
            # model_path=f"{cache_dir}/llama-adapterv2/BIAS-7B.pth",
            model_path=model_path,
            device="cuda:0",
            # base_lm_path=f"{cache_dir}/LLaMA-7B",
            base_lm_path=lm_path,
        )
    # elif testing_model == "mPLUGOwl": ## flamingo
    #     from mplug_owl_modeling import  VLLMmPLUGOwl
    #     model = VLLMmPLUGOwl(f"{cache_dir}/mplug-owl-7b-ft")
    elif testing_model == "mPLUGOwl2":
        from .mplug_owl2_modeling import VLLMmPLUGOwl2

        model = VLLMmPLUGOwl2(model_path)

    elif testing_model == "InstructBLIP-13B":  ## mmqa
        from .instruct_blip_modeling import VLLMInstructBLIP2

        model = VLLMInstructBLIP2(model_path, model_type="13B")
    elif testing_model == "InstructBLIP-7B":
        from .instruct_blip_modeling import VLLMInstructBLIP2

        model = VLLMInstructBLIP2(model_path, model_type="7B")

        # model = VLLMInstructBLIP2(f"{cache_dir}/instructblip-vicuna-7b")
    # elif testing_model == "InstructBLIP2-13B": ## mmqa
    #     from .instruct_blip_modeling import VLLMInstructBLIP2
    #     model = VLLMInstructBLIP2(f"{cache_dir}/instructblip-vicuna-13b")
    # elif testing_model == "InstructBLIP2-FlanT5-xl": ## mmqa
    #     from .instruct_blip_modeling import VLLMInstructBLIP2
    #     model = VLLMInstructBLIP2(f"{cache_dir}/instructblip-flan-t5-xl")
    # elif testing_model == "InstructBLIP2-FlanT5-xxl": ## mmqa
    #     from .instruct_blip_modeling import VLLMInstructBLIP2
    #     model = VLLMInstructBLIP2(f"{cache_dir}/instructblip-flan-t5-xxl")
    # #
    elif testing_model == "Qwen-VL-Chat":  ## mmqa
        from .qwen_vl_modeling import VLLMQwenVL

        model = VLLMQwenVL(model_path)
    elif testing_model == "CogVLM":
        from .cogvlm_modeling import VLLMCogVLMChat

        model = VLLMCogVLMChat(
            model_path=model_path,
        )

    elif testing_model == "Fuyu":  ## mmqa
        from .fuyu_modeling import VLLMFuyu

        model = VLLMFuyu(model_path=model_path)

    return model
