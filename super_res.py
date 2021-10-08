import gc
import io
import math
import sys

import lpips
from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import numpy as np

import clip
from guided_diffusion.script_util import sr_create_model_and_diffusion, sr_model_and_diffusion_defaults

from einops import rearrange
from math import log2, sqrt

model_path = './models/model012000.pt'
prompts = ['your prompt here']
image_prompts = []
batch_size = 1
clip_guidance = False       # Clip guidance typically does not have much effect for the super resolution model
clip_guidance_scale = 1000  # Controls how much the image should look like the prompt.
tv_scale = 0                # Controls the smoothness of the final output
range_scale = 0             # Controls how far out of range RGB values are allowed to be
cutn = 16
n_batches = 1
init_image = 'init.jpg'     # This can be an URL or local path
seed = 0
stop_at = 1000              # The last few iterations have a tendency to add spurious detail. Stopping early (at 800 to 900 steps) can help improve image quality, but will give the final result an air brushed look

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        print(cut_size)
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

model_config = sr_model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',  # Modify this value to decrease the number of
                                   # timesteps.
    'large_size': 256,
    'small_size': 64,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 192,
    'num_heads': 4,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})

# Load models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model, diffusion = sr_create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.requires_grad_(False).eval().to(device)

for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()

if model_config['use_fp16']:
    model.convert_to_fp16()

clip_model, clip_preprocess = clip.load('ViT-B/16', jit=False)
clip_model.eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

lpips_model = lpips.LPIPS(net='vgg').to(device)

def do_run():
    if seed is not None:
        torch.manual_seed(seed)

    make_cutouts = MakeCutouts(clip_size, cutn)
    side_x = side_y = model_config['large_size']

    target_embeds, weights = [], []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(prompt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = clip_model.encode_image(normalize(batch)).float()
        target_embeds.append(embed)
        weights.extend([weight / cutn] * cutn)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    init = Image.open(fetch(init_image)).convert('RGB')
    width, height = init.size   # Get dimensions

    # crop square
    if width != 64 or height != 64:
        if width > height:
            new_width = height
            new_height = height
        else:
            new_width = width
            new_height = width
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        init = init.crop((left, top, right, bottom))

    init_tensor = transforms.ToTensor()(init).unsqueeze(0)
    init_tensor = F.interpolate(init_tensor, size=64, mode='area') # this specific downscaling method was used during training and is necessary for best results
    init_tensor = init_tensor.to(device).mul(2).sub(1).clamp(-1,1)

    init_tensor = init_tensor.repeat(batch_size, 1, 1, 1)

    cur_t = None

    def cond_fn(x, t, low_res=None):
        with torch.enable_grad():

            x = x.detach().requires_grad_()
            n = x.shape[0]

            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t

            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'low_res': low_res})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            clip_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(clip_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
            dists = dists.view([cutn, n, -1])
            losses = dists.mul(weights).sum(2).mean(0)
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out['pred_xstart'])
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale + range_losses.sum() * range_scale
            return -torch.autograd.grad(loss, x)[0]

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={'low_res': init_tensor},
            cond_fn=cond_fn if clip_guidance else None,
            progress=True,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 50 == 0 or cur_t == -1 or j > stop_at:
                for k, image in enumerate(sample['pred_xstart']):
                    filename = f'progress_{i * batch_size + k:05}.png'
                    pimg = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                    pimg.save(filename)

            if j > stop_at:
                break

gc.collect()
do_run()
