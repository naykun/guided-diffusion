MODEL_FLAGS="--attention_resolutions 32,16,8 --emb_condition True --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --lr_warmup_steps 1000 --batch_size 32 --microbatch 4 --log_interval 1 --save_interval 500 --resume_checkpoint /msrhyper-ddn/hai1/kun/diffusion_runs/256_logs/model027500.pt"
export OPENAI_LOGDIR=/msrhyper-ddn/hai1/kun/diffusion_runs/256_logs/
bash /tmp/code/tools/changephase.bash
mpiexec -n 8 python scripts/image_gumbel_train.py --data_dir /msrhyper-ddn/hai1/G/dataset/coco/train2017/ $MODEL_FLAGS $TRAIN_FLAGS
bash /tmp/code/tools/changephase.bash