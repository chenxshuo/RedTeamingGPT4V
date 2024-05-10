import random

import torch
from tqdm import tqdm

from . import attack_utils
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


class VisualAdv:
    def __init__(self, model, targets, device="cuda:0"):
        """
        model: MiniGPT4,...
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
        prob=0.0,
        momentum=0.0,
        ti_size=1,
        IS_NI=False,
    ):
        """
        text_prompt: list(1)
        img: tensor(b,c,h,w) Image.open.convert--transform--unsqueeze(0)
        prob: DI-attack(0.4), enhance transferability; PGD-WITHOUT-DI:0.0
        momentum: MI/NI-attack(1.0) PGD-WITHOUT-MI/NI:0.0
        ti_size: TI-attack(15) PGD-WITHOUT-TI:1.0
        IS_NI: bool, true-->NI attack; false-->without NI
        """

        print(">>> batch_size:", batch_size)

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
        grad = torch.zeros_like(x)
        ti_conv = attack_utils.transition_invariant_conv(ti_size)

        # x = self.model.normalize(img).to(self.model.device).unsqueeze(0)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)

            text_prompts = [text_prompt] * batch_size

            x_t = x + adv_noise

            # NI attack
            if momentum > 0.0 and IS_NI is True:
                x_nes = x_t + alpha * momentum * grad
            else:
                x_nes = x_t

            # DI attack
            if prob > 0.0:
                x_di = attack_utils.input_diversity(
                    x_nes, image_width=img.shape[3], image_resize=255, prob=prob
                )
            else:
                x_di = x_nes

            x_adv = self.model.normalize(x_di).to(self.device)

            loss = self.model.get_loss(
                image=x_adv, text_prompts=text_prompts, target_text=batch_targets
            )
            loss.backward()

            cur_grad = adv_noise.grad.data

            # TI Attack
            if ti_size > 1:
                ti_conv.to(self.device)
                cur_grad = ti_conv(cur_grad)

            # MI Attack
            if momentum > 0.0:
                cur_grad = attack_utils.normalize_by_pnorm(cur_grad, p=1)
                grad = momentum * grad + cur_grad
            else:
                grad = cur_grad
            adv_noise.data -= alpha * grad.sign()
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
