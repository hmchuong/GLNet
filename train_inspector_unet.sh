# Train global
# export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
# python -m torch.distributed.launch --nproc_per_node=6 --use_env inspector.py \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "inspector_deepglobe_global_unet" \
# --batch_size 6 \
# --sub_batch_size 6 \
# --patch_sizes 1224 996 640 \
# --size 640 \
# --origin_size 2448 \
# --training_level -1 \
# --level_decay 0 \
# --dist-url "env://" \
# --workers 10 --world-size 6

# Train local 0
# export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
# python -m torch.distributed.launch --nproc_per_node=6 --use_env inspector.py \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "inspector_deepglobe_local0_unet" \
# --restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_global_unet/inspector_deepglobe_global_unet.pth" \
# --batch_size 6 \
# --sub_batch_size 6 \
# --patch_sizes 1224 996 640 \
# --size 640 \
# --origin_size 2448 \
# --training_level 0 \
# --level_decay 0 \
# --dist-url "env://" \
# --workers 10 --world-size 6

# Train local 1
export CUDA_VISIBLE_DEVICES=1,5,7
python -m torch.distributed.launch --nproc_per_node=3 --use_env inspector.py \
--dataset_name "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--n_class 7 \
--num_scaling_level 3 \
--log_path "/vinai/chuonghm/inspector/logs" \
--task_name "inspector_deepglobe_local1_unet" \
--restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_local0_unet/inspector_deepglobe_local0_unet" \
--batch_size 6 \
--sub_batch_size 6 \
--patch_sizes 1224 996 640 \
--size 640 \
--origin_size 2448 \
--training_level 1 \
--level_decay 0 \
--reduce_step_size 10 \
--early_stopping 20 \
--num_epochs 100 \
--dist-url "tcp://127.0.0.1:1234" \
--workers 10 --world-size 3