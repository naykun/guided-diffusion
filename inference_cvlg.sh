subset=$1
INPUT=/mnt/default/cvlg_produce/ckpt-epoch=769-cross_entropy=3.10.ckpt/generate_result_part_${subset}.pt
CUDA_VISIBLE_DEVICES=$subset python sample_cvlg.py --model_size 256 --input $INPUT --model_path /mnt/default/diffusion_runs/vqgan1024_256in_256out_logs/model200000.pt
# CUDA_VISIBLE_DEVICES=8 python super_res.py --output_size 1024 --input 1024tk_progress_00000.png