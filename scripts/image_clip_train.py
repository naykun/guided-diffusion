"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from dalle_pytorch import DiscreteVAE, VQGanVAE

import torch

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from torchvision.transforms import functional as TF

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        #Resize(n_px, interpolation=BICUBIC),
        #CenterCrop(n_px),
        #_convert_image_to_rgb,
        #ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

preprocess = _transform(256)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("setting up clip...")

    import clip

    clip_model, _ = clip.load('ViT-B/16', device=dist_util.dev(), jit=False)
    clip_model.eval().requires_grad_(False)
    del clip_model.visual.transformer
    del clip_model.transformer

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    logger.log('args: ', args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


    logger.log("creating data loader...")
    data = load_data_custom(
        clip_model = clip_model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        lr_warmup_steps=args.lr_warmup_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        lr_warmup_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        emb_condition=True,
        emb_input_dims=768,
        emb_output_dims=1024,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_data_custom(clip_model, data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        emb_condition=True
    )
    for large_batch, model_kwargs in data:
        arr_tens = large_batch.to(dist_util.dev())
        arr_tens = (arr_tens+1)/2

        with torch.no_grad():
            img = preprocess(arr_tens)
            img = img.type(clip_model.dtype)

            image_embeds = clip_model.visual.conv1(img)

        model_kwargs["image_embeds"] = image_embeds.detach().cpu()

        del model_kwargs["image_128"]
        yield large_batch, model_kwargs

if __name__ == "__main__":
    main()
