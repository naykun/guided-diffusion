CUDA_VISIBLE_DEVICES=8 python sample_1024tk.py --model_size 256 --input images/1-ground.png --model_path /msrhyper-ddn/hai1/kun/diffusion_runs/vqgan1024_256in_256out_logs/model200000.pt
# CUDA_VISIBLE_DEVICES=8 python sample_1024tk.py --model_size 256_wide --input images/1-ground.png --model_path /msrhyper-ddn/hai1/kun/diffusion_runs/vqgan1024_256in_256out_wide_logs/model053000.pt
# CUDA_VISIBLE_DEVICES=8 python super_res.py --output_size 1024 --input generated/images/1-ground.png/diffused/exposefix_rescale.png