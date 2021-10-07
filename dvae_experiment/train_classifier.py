import math
from math import sqrt
import argparse
from pathlib import Path

# torch

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torch.nn import functional as F
from PIL import Image

# dalle classes and utils

from dalle_pytorch import distributed_utils
import clip

from torch import nn, einsum
from math import log2, sqrt

from einops import rearrange

import matplotlib.pyplot as plt

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class DiscreteClassifier(nn.Module):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.,
        normalization = ((0.5,) * 3, (0.5,) * 3)
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers

        enc_chans = [channels, *enc_chans]

        enc_chans_io = list(zip(enc_chans[:-1], enc_chans[1:]))

        enc_layers = []
        classifier_layers = []

        for (enc_in, enc_out) in enc_chans_io:
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            enc_layers.append(ResBlock(enc_chans[-1]))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))

        linear_dim = 4096

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        classifier_layers = [
            nn.Linear(codebook_dim*8*8, linear_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(linear_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(linear_dim, 512)
        ]

        self.encoder = nn.Sequential(*enc_layers)
        self.classifier = nn.Sequential(*classifier_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        self.cosine_sim = nn.CosineSimilarity()

        # take care of normalization within class
        self.normalization = normalization

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        embedding = None,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)

        out = self.avgpool(sampled)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = 1 - self.cosine_sim(embedding, out).mean()

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--image_folder', type = str, required = True,
                    help='path to your folder of images for learning the discrete VAE and its codebook')

parser.add_argument('--image_size', type = int, required = False, default = 128,
                    help='image size')

parser = distributed_utils.wrap_arg_parser(parser)


train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--epochs', type = int, default = 20, help = 'number of epochs')

train_group.add_argument('--batch_size', type = int, default = 8, help = 'batch size')

train_group.add_argument('--learning_rate', type = float, default = 2e-3, help = 'learning rate')

train_group.add_argument('--lr_decay_rate', type = float, default = 0.999, help = 'learning rate decay')

train_group.add_argument('--starting_temp', type = float, default = 1., help = 'starting temperature')

train_group.add_argument('--temp_min', type = float, default = 0.5, help = 'minimum temperature to anneal to')

train_group.add_argument('--anneal_rate', type = float, default = 1e-7, help = 'temperature annealing rate')

train_group.add_argument('--num_images_save', type = int, default = 4, help = 'number of images to save')

model_group = parser.add_argument_group('Model settings')

model_group.add_argument('--num_tokens', type = int, default = 4096, help = 'number of image tokens')

model_group.add_argument('--num_layers', type = int, default = 3, help = 'number of layers (should be 3 or above)')

model_group.add_argument('--num_resnet_blocks', type = int, default = 2, help = 'number of residual net blocks')

model_group.add_argument('--smooth_l1_loss', dest = 'smooth_l1_loss', action = 'store_true')

model_group.add_argument('--emb_dim', type = int, default = 512, help = 'embedding dimension')

model_group.add_argument('--hidden_dim', type = int, default = 256, help = 'hidden dimension')

model_group.add_argument('--kl_loss_weight', type = float, default = 1e-7, help = 'KL loss weight')

args = parser.parse_args()

# constants

IMAGE_SIZE = args.image_size
IMAGE_PATH = args.image_folder

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
LR_DECAY_RATE = args.lr_decay_rate

NUM_TOKENS = args.num_tokens
NUM_LAYERS = args.num_layers
NUM_RESNET_BLOCKS = args.num_resnet_blocks
SMOOTH_L1_LOSS = args.smooth_l1_loss

EMB_DIM = args.emb_dim
HIDDEN_DIM = args.hidden_dim
KL_LOSS_WEIGHT = args.kl_loss_weight

STARTING_TEMP = args.starting_temp
TEMP_MIN = args.temp_min
ANNEAL_RATE = args.anneal_rate

NUM_IMAGES_SAVE = args.num_images_save

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

using_deepspeed = \
    distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

# data
def center_crop(im):
    width, height = im.size   # Get dimensions

    if width > height:
        new_width = height
        new_height = height
    elif height > width:
        new_height = width
        new_width = width
    else:
        new_width = width
        new_height = height

    left = (width - new_width)/2
    top = (height - new_height)/2

    right = (width + new_width)/2
    bottom = (height + new_height)/2

    if height > width: # a heuristic to prevent people's heads from being consistently chopped off
        top = top/2
        bottom = top + width

    # center crop
    im = im.crop((left, top, right, bottom))
    im = im.resize((224,224), Image.LANCZOS)

    return im

image_transforms = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Lambda(center_crop),
    T.ToTensor()
])

ds = ImageFolder(
    IMAGE_PATH,
    image_transforms
)

if distributed_utils.using_backend(distributed_utils.HorovodBackend):
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank())
else:
    data_sampler = None

dl = DataLoader(ds, BATCH_SIZE, shuffle = not data_sampler, sampler=data_sampler, num_workers=24)

vae_params = dict(
    image_size = IMAGE_SIZE,
    num_layers = NUM_LAYERS,
    num_tokens = NUM_TOKENS,
    codebook_dim = EMB_DIM,
    hidden_dim   = HIDDEN_DIM,
    num_resnet_blocks = NUM_RESNET_BLOCKS
)

vae = DiscreteClassifier(
    **vae_params,
    smooth_l1_loss = SMOOTH_L1_LOSS,
    kl_div_loss_weight = KL_LOSS_WEIGHT
)
if not using_deepspeed:
    vae = vae.cuda()


assert len(ds) > 0, 'folder does not contain any images'
if distr_backend.is_root_worker():
    print(f'{len(ds)} images found for training')

# optimizer

opt = Adam(vae.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

# distribute

distr_backend.check_batch_size(BATCH_SIZE)
deepspeed_config = {'train_batch_size': BATCH_SIZE}

(distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
    args=args,
    model=vae,
    optimizer=opt,
    model_parameters=vae.parameters(),
    training_data=ds if using_deepspeed else dl,
    lr_scheduler=sched if not using_deepspeed else None,
    config_params=deepspeed_config,
)

using_deepspeed_sched = False
# Prefer scheduler in `deepspeed_config`.
if distr_sched is None:
    distr_sched = sched
elif using_deepspeed:
    # We are using a DeepSpeed LR scheduler and want to let DeepSpeed
    # handle its scheduling.
    using_deepspeed_sched = True

def save_model(path):
    save_obj = {
        'hparams': vae_params,
    }
    #if using_deepspeed:
    #    cp_path = Path(path)
    #    path_sans_extension = cp_path.parent / cp_path.stem
    #    cp_dir = str(path_sans_extension) + '-ds-cp'

    #    distr_vae.save_checkpoint(cp_dir, client_state=save_obj)
        # We do not return so we do get a "normal" checkpoint to refer to.

    if not distr_backend.is_root_worker():
        return

    save_obj = {
        **save_obj,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)

clip_model, clip_preprocess = clip.load("ViT-B/16", device="cuda")
# starting temperature

global_step = 0
temp = STARTING_TEMP

clip_transform = T.Compose([
    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(distr_dl):
        images_clip = clip_transform(images)
        images = F.interpolate(images, size=IMAGE_SIZE, mode='area')
        images = images.cuda()
        images_clip = images_clip.cuda()

        with torch.no_grad():
            image_features = clip_model.encode_image(images_clip).float()
        
        loss, recons = distr_vae(
            images,
            embedding = image_features,
            return_loss = True,
            return_recons = True,
            temp = temp
        )

        if using_deepspeed:
            # Gradients are automatically zeroed after the step
            distr_vae.backward(loss)
            distr_vae.step()
        else:
            distr_opt.zero_grad()
            loss.backward()
            distr_opt.step()

        logs = {}

        if i % 100 == 0:
            if distr_backend.is_root_worker():

                with torch.no_grad():
                    codes = vae.get_codebook_indices(images[:12])
                    codes = codes.flatten().detach().cpu().numpy()
                    plt.clf()
                    plt.hist(codes, bins=100, color='skyblue', ec='skyblue')
                    plt.savefig('codebook-hist.png')

                save_model(f'./classifier.pt')

            # temperature anneal

            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # lr decay

            # Do not advance schedulers from `deepspeed_config`.
            if not using_deepspeed_sched:
                distr_sched.step()

        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)

        if distr_backend.is_root_worker():
            if i % 100 == 0:
                lr = distr_sched.get_last_lr()[0]
                print(epoch, i, f'lr - {lr:6f} loss - {avg_loss.item()}', 'temp: ', temp)

                logs = {
                    **logs,
                    'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item(),
                    'lr': lr
                }

           # wandb.log(logs)
        global_step += 1


if distr_backend.is_root_worker():
    # save final vae and cleanup

    save_model('./classifier-final.pt')
