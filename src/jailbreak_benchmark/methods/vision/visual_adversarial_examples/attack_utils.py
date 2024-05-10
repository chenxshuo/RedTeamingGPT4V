import torch
import torch.nn as nn


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1.0 / p)


def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    # TODO: Currently only considered one way of "uniform" sampling
    # for Linf, there are 3 ways:
    #   1) true uniform sampling by first calculate the rectangle then sample
    #   2) uniform in eps box then truncate using data domain (implemented)
    #   3) uniform sample in data domain then truncate with eps box
    # for L2, true uniform sampling is hard, since it requires uniform sampling
    #   inside a intersection of cube and ball, so there are 2 ways:
    #   1) uniform sample in the data domain, then truncate using the L2 ball
    #       (implemented)
    #   2) uniform sample in the L2 ball, then truncate using the data domain
    # for L1: uniform l1 ball init, then truncate using the data domain

    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        delta.data.uniform_(clip_min, clip_max)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    # elif ord == 1:
    #     ini = laplace.Laplace(
    #         loc=delta.new_tensor(0), scale=delta.new_tensor(1))
    #     delta.data = ini.sample(delta.data.shape)
    #     delta.data = normalize_by_pnorm(delta.data, p=1)
    #     ray = uniform.Uniform(0, eps).sample()
    #     delta.data *= ray
    #     delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data


import os

import numpy as np
from imageio import imsave


def save_images(images, filename, output_dir):
    cur_images = (np.round(images[0, :, :, :] * 255)).astype(np.uint8)
    with open(os.path.join(output_dir, filename), "wb") as f:
        imsave(f, cur_images.transpose(1, 2, 0), format="bmp")


from random import randint
from torchvision import transforms
from PIL import Image


def input_diversity(input_tensor, image_width, image_resize, prob):
    if prob > 0.0:
        rnd = randint(image_width, image_resize)
        rescaled = transforms.Resize([rnd, rnd], interpolation=Image.NEAREST)(input_tensor)
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = randint(0, w_rem)
        pad_right = w_rem - pad_left
        # 要看一下padded的维度来验证  left, top, right and bottom
        padded = transforms.Pad([pad_left, pad_top, pad_right, pad_bottom])(rescaled)

        padded = transforms.Resize([image_width, image_width], interpolation=Image.NEAREST)(padded)

        # padded.set_shape((input_tensor.shape[0], image_resize, image_resize, 3))
        rnd_prob = randint(0, 100) / 100.0
        if rnd_prob < prob:
            return padded
        else:
            return input_tensor
    else:
        return input_tensor


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils

    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1.0 / norm, x)


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def transition_invariant_conv(size=15):
    kernel = gkern(size, 3).astype(np.float32)
    padding = size // 2
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)

    conv = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=size,
        stride=1,
        groups=3,
        padding=padding,
        bias=False,
    )
    conv.weight.data = conv.weight.new_tensor(data=stack_kernel)

    return conv
