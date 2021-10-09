# Conditional Clip-Guided Diffusion

This project is based on Clip-Guided Diffusion by [RiversHaveWings](https://twitter.com/RiversHaveWings)

This diffusion model is ultimately meant for image generation via [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch/) by replacing the VAE decoder. By itself, it can perform text-guided image-to-image translation. In this application the overall structure of the original image is preserved, while the details are re-generated with clip guidance. This repository also contains a separate model for clip-guided super-resolution from 64x64 to 256x256

more details at this [github discussion](https://github.com/lucidrains/DALLE-pytorch/discussions/375)

# Sample generations

images from Unsplash

reconstruction from 256x16x16 latents (best out of 8)

| Ground truth | GumbelVQ f8 | CCGD | CCGD + clip | CCGD + clip | CCGD + clip |
| --- | --- | --- | --- | --- | --- |
| <img src="./images/1-ground.png"></img> | <img src="./images/1-gumbel.png"></img> | <img src="./images/1-ccgd-noclip.png"></img> | <img src="./images/1-ccgd-clip-1.png"></img> | <img src="./images/1-ccgd-clip-2.png"></img> | <img src="./images/1-ccgd-clip-3.png"></img> |
| &nbsp; | &nbsp; | &nbsp; | <sub>Prompt: a girl</sub> | <sub>Prompt: a smiling girl</sub> | <sub>Prompt: a girl with blonde hair</sub> |
| <img src="./images/2-ground.png"></img> | <img src="./images/2-gumbel.png"></img> | <img src="./images/2-ccgd-noclip.png"></img> | <img src="./images/2-ccgd-clip-1.png"></img> | <img src="./images/2-ccgd-clip-2.png"></img> | <img src="./images/2-ccgd-clip-3.png"></img> |
| &nbsp; | &nbsp; | &nbsp; | <sub>Prompt: a DSLR camera</sub> | <sub>Prompt: a Canon DSLR camera</sub> | <sub>Prompt: a Nikon DSLR camera</sub> |
| <img src="./images/3-ground.png"></img> | <img src="./images/3-gumbel.png"></img> | <img src="./images/3-ccgd-noclip.png"></img> | <img src="./images/3-ccgd-clip-1.png"></img> | <img src="./images/3-ccgd-clip-2.png"></img> | <img src="./images/3-ccgd-clip-3.png"></img> |
| &nbsp; | &nbsp; | &nbsp; | <sub>Prompt: a cute dog</sub> | <sub>Prompt: a vicious wolf</sub> | <sub>Prompt: a cat </sub> |

superresolution from 64x64 to 256x256

| Ground truth | 64x64 | Upscaled |
| --- | --- | --- |
| <img src="./images/1-ground.png"></img> | <img src="./images/1-64x64.png"></img> | <img src="./images/1-upscaled.png"></img> |
| <img src="./images/2-ground.png"></img> | <img src="./images/2-64x64.png"></img> | <img src="./images/2-upscaled.png"></img> |
| <img src="./images/3-ground.png"></img> | <img src="./images/3-64x64.png"></img> | <img src="./images/3-upscaled.png"></img> |

# Download pre-trained models

Each of these diffusion models have been fine-tuned from the corresponding model released by OpenAI. I've also included the EMA and optimizer files for further training.

 * 64x64 GumbelVQ: [model064000.pt](https://dall-3.com/models/guided-diffusion/64/)
 * 128x128 GumbelVQ: [model072000.pt](https://dall-3.com/models/guided-diffusion/128/)
 * 256x256 GumbelVQ: [model054000.pt](https://dall-3.com/models/guided-diffusion/256/)
 * 64x64 -&gt; 256x256 upsampler: [model016000.pt](https://dall-3.com/models/guided-diffusion/64_256/)

Experimental models
 * 128x128 DVAE encoder : [model009000.pt](https://dall-3.com/models/guided-diffusion/128dvae/)
 * 64x64 DVAE Classifier encoder : [model022000.pt](https://dall-3.com/models/guided-diffusion/64dvae/)

# Installation

You will need to install [CLIP](https://github.com/openai/CLIP) and [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch/)

then clone this repository and
```
pip install -e .
```

# Sampling from pre-trained models

To sample from these models, you can use the `sample.py`, and `super_res.py` scripts in the root directory.


```
# first download the GumbelVQ model
mkdir -p models/vqgan_gumbel_f8/configs && wget 'https://heibox.uni-heidelberg.de/f/b24d14998a8d4f19a34f/?dl=1' -O 'models/vqgan_gumbel_f8/configs/model.yaml' 
mkdir -p models/vqgan_gumbel_f8/checkpoints && wget 'https://heibox.uni-heidelberg.de/f/34a747d5765840b5a99d/?dl=1' -O 'models/vqgan_gumbel_f8/checkpoints/last.ckpt' 

# (optional) for the 128x128 dvae model
mkdir -p models/dvae/ && wget 'https://dall-3.com/models/dvae/vae-final-128-8192.pt' -O 'models/dvae/vae-final-128-8192.pt' 

# (optional) for the 64x64 classifier model
mkdir -p models/dvae/ && wget 'https://dall-3.com/models/dvae/vae-classifier.pt' -O 'models/dvae/vae-classifier.pt' 

# download the appropriate diffusion model and put in ./models/

# edit configs in sample.py or super_res.py

python sample.py
python super_res.py
```

# Training

You can use the original OpenAI models to train from scratch, or continue training from the models in this repo by putting the modelXXX, emaXXX and optXXX files in the OPENAI_LOGDIR directory

note the new flags --emb_condition and --lr_warmup_steps


 * 64x64 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --emb_condition True --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --lr_warmup_steps 1000 --batch_size 70 --microbatch 35 --log_interval 1 --save_interval 2000 --resume_checkpoint models/64x64_diffusion.pt"
export OPENAI_LOGDIR=./64_logs/
mpiexec -n 4 python scripts/image_gumbel_train.py --data_dir ./path/to/data/ $MODEL_FLAGS $TRAIN_FLAGS
```

 * 128x128 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --emb_condition True --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --lr_warmup_steps 1000 --lr_warmup_steps 1000 --batch_size 66 --microbatch 11 --log_interval 1 --save_interval 1000 --resume_checkpoint models/128x128_diffusion.pt"
export OPENAI_LOGDIR=./128_logs/
mpiexec -n 4 python scripts/image_gumbel_train.py --data_dir ./path/to/data/ $MODEL_FLAGS $TRAIN_FLAGS
```

 * 256x256 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --emb_condition True --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --lr_warmup_steps 1000 --batch_size 64 --microbatch 4 --log_interval 1 --save_interval 500 --resume_checkpoint models/256x256_diffusion_uncond.pt"
export OPENAI_LOGDIR=./256_logs/
mpiexec -n 4 python scripts/image_gumbel_train.py --data_dir ./path/to/data/ $MODEL_FLAGS $TRAIN_FLAGS
```

 * 64x64 -&gt; 256x256 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --lr_warmup_steps 1000 --batch_size 64 --microbatch 4 --log_interval 1 --save_interval 1000 --resume_checkpoint models/64_256_upsampler.pt"
export OPENAI_LOGDIR=./64_256_logs/
mpiexec -n 4 python scripts/super_res_train.py --data_dir ./path/to/data/ $MODEL_FLAGS $TRAIN_FLAGS
```

for the dvae models it's the same as above, except use scripts/image_dvae_train.py

to resume training the models in this repo, use --resume_checkpoint OPENAI_LOGDIR/modelXXX.pt (the current step, optimizer and ema checkpoints are inferred from the filename)
