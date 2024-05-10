import random

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from ..visual_adversarial_examples import attack_utils
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


class ImageHijacks:
    def __init__(self, model, targets, device="cuda:0"):
        """
        model: MiniGPT4ForAttacker,...
        targets: (list) all target texts in dataset
        """
        self.model = model
        self.device = device

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

    def attack_constrained(
        self,
        text_prompt,
        img,
        save_dir,
        batch_size=8,
        num_iter=2000,
        alpha=1 / 255,
        epsilon=16 / 255,
        steps=1,
    ):
        """
        text_prompt: list(batch_size)
        img: tensor(b,c,h,w) Image.open.convert--transform--unsqueeze(0)
        """

        print(">>> batch_size:", batch_size)
        # text_len = len(text_prompt)
        numbers = list(range(self.num_targets))

        # adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        # x = denormalize(img).clone().to(self.device)
        # adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        #
        # adv_noise.requires_grad_(True)
        # adv_noise.retain_grad()

        img = img.to(self.device)
        adv_noise = torch.zeros_like(img).to(self.device)
        adv_noise.data = attack_utils.rand_init_delta(adv_noise, img, np.inf, epsilon, 0.0, 1.0)
        adv_noise.data = attack_utils.clamp(img + adv_noise.data, min=0.0, max=1.0) - img
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        x = img.clone().to(self.device)

        # x = self.model.normalize(img).to(self.model.device).unsqueeze(0)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = []
            text_prompts = []

            random_numbers = random.sample(numbers, batch_size)
            for j in range(batch_size):
                batch_targets.append(self.targets[random_numbers[j]])
                text_prompts.append(text_prompt[random_numbers[j]])

            # batch_targets = random.sample(self.targets, batch_size)

            # text_prompts = [text_prompt] * batch_size
            for s in range(steps):
                x_adv = self.model.normalize(x + adv_noise).to(self.device)

                logits, targets = self.model.get_logits(
                    image=x_adv, text_prompts=text_prompts, target_text=batch_targets
                )
                loss = self.image_hijacks_loss(logits, targets)
                loss.backward()

                cur_grad = adv_noise.grad.data
                adv_noise.data -= alpha * cur_grad.sign()
                adv_noise.data = attack_utils.clamp(adv_noise.data, -epsilon, epsilon)
                adv_noise.data = attack_utils.clamp(x.data + adv_noise.data, 0.0, 1.0) - x.data

                adv_noise.grad.zero_()
                self.model.zero_grad()

                self.loss_buffer.append(loss.item())

                print("target_loss: %f" % (loss.item()))

            if t % 20 == 0:
                self.plot_loss(save_dir)

            if t % 100 == 0:
                print("######### Output - Iter = %d ##########" % t)
                adv_img_prompt = x + adv_noise
                adv_img_prompt = adv_img_prompt.detach().cpu().numpy()
                attack_utils.save_images(
                    adv_img_prompt, filename="bad_prompt_temp_%d.bmp" % t, output_dir=save_dir
                )

        return adv_img_prompt

    def plot_loss(self, save_dir):
        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label="Target Loss")

        # Add in a title and axes labels
        plt.title("Loss Plot")
        plt.xlabel("Iters")
        plt.ylabel("Loss")

        # Display the plot
        plt.legend(loc="best")
        plt.savefig("%s/loss_curve.png" % save_dir)
        plt.clf()

        torch.save(self.loss_buffer, "%s/loss" % save_dir)

    def image_hijacks_loss(self, logits, targets):
        batch_size = targets.shape[0]
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)

        loss = 0.0
        for b in range(batch_size):
            temp_loss = loss_fct(logits[b], targets[b].long().to(self.device))
            loss = loss + temp_loss
        return loss
