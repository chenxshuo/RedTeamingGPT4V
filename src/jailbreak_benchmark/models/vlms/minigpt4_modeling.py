"""
Refer to https://github.com/Vision-CAIR/MiniGPT-4 to prepare environments and checkpoints
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .base import VLLMBaseModel

from .base import VLLMForAttacker

from .minigpt4_utils.common.config import Config
from .minigpt4_utils.common.dist_utils import get_rank
from .minigpt4_utils.common.registry import registry
from .minigpt4_utils.conversation.conversation import (
    Chat,
    CONV_VISION_Vicuna0,
    CONV_VISION_LLama2,
    StoppingCriteriaSub,
)

from .minigpt4_utils.datasets.builders import *
from .minigpt4_utils.models import *
from .minigpt4_utils.processors import *
from .minigpt4_utils.runners import *
from .minigpt4_utils.tasks import *
from transformers import StoppingCriteriaList

conv_dict = {
    "pretrain_vicuna": CONV_VISION_Vicuna0,
    "pretrain_llama2": CONV_VISION_LLama2,
    "pretrain_vicuna_13b": CONV_VISION_Vicuna0,
}


class VLLMMiniGPT4(VLLMBaseModel):
    def __init__(self, model_path, model_type="", device="cuda:0"):
        super().__init__(model_path, device)
        ## './minigpt4_utils/minigpt4_eval.yaml'
        cfg = Config(cfg_path=self.model_path)
        self.model_type = model_type
        model_config = cfg.model_cfg
        self.CONV_VISION = conv_dict[model_config.model_type]
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(self.device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
            vis_processor_cfg
        )
        self.stop_words_ids = [[835], [2277, 29937]]
        self.stop_words_ids = [torch.tensor(ids).to(self.device) for ids in self.stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=self.stop_words_ids)]
        )
        self.chat = Chat(
            model, self.vis_processor, device=device, stopping_criteria=self.stopping_criteria
        )

    def __str__(self):
        return f"VLLMMiniGPT4-{self.model_type}-{self.model_path}"

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        if len(images) == 0:
            raise ValueError("No image is provided.")
        if len(images) > 1:
            return "[Skipped]: Currently only support single image."

        # Init chat state
        CONV_VISION = self.CONV_VISION
        chat_state = CONV_VISION.copy()
        img_list = []

        # download image image
        image_path = images[0]
        img = Image.open(image_path).convert("RGB")

        # upload Image
        self.chat.upload_img(img, chat_state, img_list)
        self.chat.encode_img(img_list)
        instruction = instruction[0] if type(instruction) == list else instruction
        # ask
        self.chat.ask(instruction, chat_state)

        # answer
        out = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=5,
            temperature=1.0,
            max_new_tokens=100,
            max_length=2000,
        )[0]

        return out


class MiniGPT4ForAttacker(VLLMForAttacker):
    def __init__(
        self,
        model_path,
        model_type="",
        device="cuda:0",
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
    ):
        super().__init__(model_path, device)
        ## './minigpt4_utils/minigpt4_eval.yaml'
        cfg = Config(cfg_path=self.model_path)
        self.model_type = model_type
        model_config = cfg.model_cfg
        self.CONV_VISION = conv_dict[model_config.model_type]
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
            vis_processor_cfg
        )
        self.stop_words_ids = [[835], [2277, 29937]]
        self.stop_words_ids = [torch.tensor(ids).to(self.device) for ids in self.stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=self.stop_words_ids)]
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    vis_processor_cfg.image_size,
                    # scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)
        self.tokenize = self.model.llama_tokenizer

    def get_loss(self, image, text_prompts, target_text):
        """
        image: tensor(1,c,h,w) PIL.Image.open(img).convert(RGB)--self.transform--self.normalize
        text_prompt: list(batch_size)
        target_text: list(batch_size)
        """
        assert len(text_prompts) == len(
            target_text
        ), f"Unmathced batch size of prompts and targets {len(text_prompts)} != {len(target_text)}"
        batch_size = len(target_text)
        images = [[image]] * batch_size
        text_embs = self.generate_text_embedding(text_prompts)
        img_embs = self.generate_img_embedding(images)
        context_embs = self.generate_context_embedding(text_embs, img_embs)
        assert len(context_embs) == len(
            target_text
        ), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(target_text)}"

        self.model.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.model.llama_tokenizer(
            target_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False,
        ).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        bos = (
            torch.ones(
                [1, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device,
            )
            * self.model.llama_tokenizer.bos_token_id
        )
        bos_embs = self.model.llama_model.model.embed_tokens(bos)

        pad = (
            torch.ones(
                [1, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device,
            )
            * self.model.llama_tokenizer.pad_token_id
        )
        pad_embs = self.model.llama_model.model.embed_tokens(pad)

        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )

        pos_padding = torch.argmin(T, dim=1)  # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):
            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i : i + 1, :target_length])
            input_embs.append(to_regress_embs[i : i + 1, :target_length])  # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):
            # masked out the context from loss computation
            context_mask = (
                torch.ones([1, context_tokens_length[i] + 1], dtype=torch.long)
                .to(self.device)
                .fill_(-100)  # plus one for bos
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = torch.ones([1, num_to_pad], dtype=torch.long).to(self.device).fill_(-100)

            targets_mask[i] = torch.cat([context_mask, targets_mask[i], padding_mask], dim=1)
            input_embs[i] = torch.cat(
                [bos_embs, context_embs[i], input_embs[i], pad_embs.repeat(1, num_to_pad, 1)], dim=1
            )
            attention_mask.append(
                torch.LongTensor([[1] * (1 + seq_tokens_length[i]) + [0] * num_to_pad])
            )

        targets = torch.cat(targets_mask, dim=0).to(self.device)
        inputs_embs = torch.cat(input_embs, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)

        outputs = self.model.llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return loss

    def get_logits(self, image, text_prompts, target_text):
        """
        image: tensor(1,c,h,w) PIL.Image.open(img).convert(RGB)--self.transform--self.normalize
        text_prompt: list(batch_size)
        target_text: list(batch_size)
        """
        assert len(text_prompts) == len(
            target_text
        ), f"Unmathced batch size of prompts and targets {len(text_prompts)} != {len(target_text)}"
        batch_size = len(target_text)
        images = [[image]] * batch_size
        text_embs = self.generate_text_embedding(text_prompts)
        img_embs = self.generate_img_embedding(images)
        context_embs = self.generate_context_embedding(text_embs, img_embs)
        assert len(context_embs) == len(
            target_text
        ), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(target_text)}"

        self.model.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.model.llama_tokenizer(
            target_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False,
        ).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        bos = (
            torch.ones(
                [1, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device,
            )
            * self.model.llama_tokenizer.bos_token_id
        )
        bos_embs = self.model.llama_model.model.embed_tokens(bos)

        pad = (
            torch.ones(
                [1, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device,
            )
            * self.model.llama_tokenizer.pad_token_id
        )
        pad_embs = self.model.llama_model.model.embed_tokens(pad)

        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )

        pos_padding = torch.argmin(T, dim=1)  # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):
            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i : i + 1, :target_length])
            input_embs.append(to_regress_embs[i : i + 1, :target_length])  # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):
            # masked out the context from loss computation
            context_mask = (
                torch.ones([1, context_tokens_length[i] + 1], dtype=torch.long)
                .to(self.device)
                .fill_(-100)  # plus one for bos
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = torch.ones([1, num_to_pad], dtype=torch.long).to(self.device).fill_(-100)

            targets_mask[i] = torch.cat([context_mask, targets_mask[i], padding_mask], dim=1)
            input_embs[i] = torch.cat(
                [bos_embs, context_embs[i], input_embs[i], pad_embs.repeat(1, num_to_pad, 1)], dim=1
            )
            attention_mask.append(
                torch.LongTensor([[1] * (1 + seq_tokens_length[i]) + [0] * num_to_pad])
            )

        targets = torch.cat(targets_mask, dim=0).to(self.device)
        inputs_embs = torch.cat(input_embs, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)

        outputs = self.model.llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        logits = outputs.logits

        return logits, targets

    def generate_text_embedding(self, text_prompts):
        if text_prompts is None:
            return []

        text_embs = []
        for item in text_prompts:  # for each prompt within a batch
            prompt_segs = item.split("<ImageHere>")  # each <ImageHere> corresponds to one image
            seg_tokens = [
                self.model.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0)
                .to(self.device)
                .input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            embs = [
                self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens
            ]  # text to embeddings
            text_embs.append(embs)

        return text_embs

    def generate_img_embedding(self, images):
        if images is None:
            return []

        img_embs = []
        for items in images:
            embs = []
            for img in items:
                feats, _ = self.model.encode_img(img)
                embs.append(feats)
            img_embs.append(embs)

        return img_embs

    def generate_context_embedding(
        self, batch_text_embs, batch_img_embs, max_new_tokens=100, max_length=2000
    ):
        # assert len(text_embs) == len(img_embs) + 1, "Unmatched numbers of image placeholders and images."

        assert len(batch_text_embs) == len(
            batch_img_embs
        ), "Unmathced batch size of text and image prompts"

        batch_size = len(batch_text_embs)
        batch_context_embs = []

        for i in range(batch_size):
            text_embs = batch_text_embs[i]
            img_embs = batch_img_embs[i]

            num_text_segs = len(text_embs)
            num_img_segs = len(img_embs)

            if num_text_segs == 0 and num_img_segs == 0:  # empty context
                mixed_embs = [torch.zeros([1, 0, 0])]
            elif num_text_segs == 0:  # pure img context
                mixed_embs = img_embs
            elif num_img_segs == 0:  # pure text context
                mixed_embs = text_embs
            else:  # mix
                s = t = 0
                mixed_embs = []
                while s < num_text_segs and t < num_img_segs:
                    mixed_embs.append(text_embs[s])
                    mixed_embs.append(img_embs[t])
                    s, t = s + 1, t + 1
                if s < num_text_segs:
                    mixed_embs += text_embs[s:]
                if t < num_img_segs:
                    mixed_embs += img_embs[t:]

            mixed_embs = torch.cat(mixed_embs, dim=1)

            current_max_len = mixed_embs.shape[1] + max_new_tokens
            if current_max_len - max_length > 0:
                print(
                    "Warning: The number of tokens in current conversation exceeds the max length. "
                    "The model will not see the contexts outside the range."
                )
            begin_idx = max(0, current_max_len - max_length)
            mixed_embs = mixed_embs[:, begin_idx:]

            batch_context_embs.append(mixed_embs)

        return batch_context_embs


if __name__ == "__main__":
    # test
    from test_cases import TEST_CASES

    model = VLLMMiniGPT4("./minigpt4_utils/minigpt4_eval.yaml")

    for test_case in TEST_CASES:
        pred = model.generate(
            instruction=test_case["instruction"],
            images=test_case["images"],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f"Answer:\t{pred}")
        print("-" * 20)

    # test loss/logits

    model = MiniGPT4ForAttacker("./minigpt4_utils/minigpt4_eval.yaml")
    image = Image.open(
        "/home/vcl3/projects/ywq/jailbreak_benchmark/cache/test_cases/clean.jpeg"
    ).convert("RGB")
    image = model.transform(image)
    image = model.normalize(image).to(model.device)
    minigpt4_chatbot_prompt_no_text_input = "Give the following image: <Img>ImageContent</Img>. \
    You will be able to see the image once I provide it to you. \
    Please answer my questions.###Human: <Img><ImageHere></Img> ###Assistant:"
    text_prompts = [minigpt4_chatbot_prompt_no_text_input] * 4
    target_text = [" "] * 4
    loss = model.get_loss(image=image, text_prompts=text_prompts, target_text=target_text)
    print(loss)
