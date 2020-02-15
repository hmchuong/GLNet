export CUDA_VISIBLE_DEVICES=7
python inspector.py \
--evaluation \
--generate_image \
--dataset_name "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--n_class 7 \
--num_scaling_level 3 \
--refinement 1 \
--glob2local \
--supervision \
--log_path "/vinai/chuonghm/inspector/logs" \
--task_name "eval_global_transpose" \
--restore_path "/vinai/chuonghm/inspector/logs/inspector_global_508_transpose/inspector_global_508_transpose.pth" \
--patch_sizes 1350 702 508 \
--size 508 \
--origin_size 2448 \
--training_level -1 \
--lr 5e-5 \
--reduce_step_size 50 \
--early_stopping 120 \
--num_epochs 120 \
--reduce_factor 0.4 \
--level_decay 0 \
--lamb_fmreg 0.15 \
--add_weight \
--workers 10