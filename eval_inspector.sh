# export CUDA_VISIBLE_DEVICES=0
# python inspector.py \
# --evaluation \
# --generate_image \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "eval_inspector_deepglobe_global_fcn" \
# --restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_global_fcn/inspector_deepglobe_global_fcn.pth" \
# --batch_size 6 \
# --sub_batch_size 6 \
# --patch_sizes 1224 996 640 \
# --size 640 \
# --origin_size 2448 \
# --training_level -1 \
# --level_decay 0 \
# --workers 10

export CUDA_VISIBLE_DEVICES=0
python inspector.py \
--evaluation \
--generate_image \
--dataset_name "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--n_class 7 \
--num_scaling_level 3 \
--log_path "/vinai/chuonghm/inspector/logs" \
--task_name "eval_inspector_deepglobe_global_unet" \
--restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_global_unet/inspector_deepglobe_global_unet.pth" \
--batch_size 6 \
--sub_batch_size 6 \
--patch_sizes 1224 996 640 \
--size 640 \
--origin_size 2448 \
--training_level -1 \
--level_decay 0 \
--workers 10