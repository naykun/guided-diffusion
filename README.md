# Conditional Clip-Guided Diffusion

This project is based on Clip-Guided Diffusion by [RiversHaveWings](https://twitter.com/RiversHaveWings)

It's ultimately meant for image generation via [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch/) by replacing the VAE decoder with a diffusion model. By itself, the diffusion model can perform text-guided image-to-image translation. In this application the overall structure of the original image is preserved, while the details are re-generated with clip guidance. This repository also contains a separate model for clip-guided super-resolution from 64x64 to 256x256

# Sample generations

images from Unsplash

reconstruction from 256x16x16 latents

| Ground truth | GumbelVQ f8 | CCGD without clip | CCGD + clip | CCGD + clip | CCGD + clip |
| --- | --- | --- | --- | --- | --- |
| <img src="./images/1-ground.png"></img> | <img src="./images/1-gumbel.png"></img> | <img src="./images/1-ccgd-noclip.png"></img> | <img src="./images/1-ccgd-clip-1.png"></img><br /><sub>Prompt: a girl</sub> | <img src="./images/1-ccgd-clip-2.png"></img><br /><sub>Prompt: a smiling girl</sub> | <img src="./images/1-ccgd-clip-3.png"></img><br /><sub>Prompt: a girl with glasses</sub> |
| <img src="./images/2-ground.png"></img> | <img src="./images/2-gumbel.png"></img> | <img src="./images/2-ccgd-noclip.png"></img> | <img src="./images/2-ccgd-clip-1.png"></img><br /><sub>Prompt: a DSLR camera</sub> | <img src="./images/2-ccgd-clip-2.png"></img><br /><sub>Prompt: a Canon DSLR camera</sub> | <img src="./images/2-ccgd-clip-3.png"></img><br /><sub>Prompt: a Nikon DSLR camera </sub> |

superresolution from 64x64 to 256x256

| Ground truth | 64x64 | Upscaled |
| <img src="./images/1-ground.png"></img> | <img src="./images/1-64x64.png"></img> | <img src="./images/1-upscaled.png"></img> |

# Download pre-trained models

Each of these models have been fine-tuned from the corresponding guided diffusion model released by OpenAI. I've also included the EMA and optimizer files for further training.

 * 64x64 GumbelVQ: [model064000.pt](https://dall-3.com/models/guided-diffusion/64/)
 * 128x128 GumbelVQ: [model072000.pt](https://dall-3.com/models/guided-diffusion/128/)
 * 256x256 GumbelVQ: [model021500.pt](https://dall-3.com/models/guided-diffusion/256/)
 * 64x64 -> 256x256 upsampler: [model010000.pt](https://dall-3.com/models/guided-diffusion/64_256/)
 * 128x128 DVAE: [model009000.pt](https://dall-3.com/models/guided-diffusion/128dvae/)

# Installation

You will need to install [clip](https://github.com/openai/CLIP), [VQGAN](https://github.com/CompVis/taming-transformers), and [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch/)

then clone this repository and
```
pip install -e .
```

# Sampling from pre-trained models

To sample from these models, you can use the `sample.py`, and `super_res.py` scripts in the root directory.


```
# first download the GumbelVQ model
wget 'https://heibox.uni-heidelberg.de/f/b24d14998a8d4f19a34f/?dl=1' -O 'models/vqgan_gumbel_f8/configs/model.yaml' 
wget 'https://heibox.uni-heidelberg.de/f/34a747d5765840b5a99d/?dl=1' -O 'models/vqgan_gumbel_f8/checkpoints/last.ckpt' 


python sample.py --init init.jpg --image_size 64 --checkpoint models/model064000.pt --batch_size 4"

# sample with a DVAE encoder instead of GumbelVQ
python sample.py --mode dvae --init init.jpg --image_size 128 --checkpoint models/model009000.pt --batch_size 4"
```
