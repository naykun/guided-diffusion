INPUT=version_6/validate_76999.res
CUDA_VISIBLE_DEVICES=8 python sample_1024tk.py --model_size 256 --input $INPUT --model_path /msrhyper-ddn/hai1/kun/diffusion_runs/vqgan1024_256in_256out_logs/model106500.pt
# CUDA_VISIBLE_DEVICES=8 python super_res.py --output_size 1024 --input 1024tk_progress_00000.png