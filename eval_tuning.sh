# export CUDA_VISIBLE_DEVICES=0
# python inspector.py \
# --evaluation \
# --generate_image \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --refinement 1 \
# --glob2local \
# --supervision \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "eval_local0_1350" \
# --restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_global_resnetfpn_508_local0_3/inspector_deepglobe_global_resnetfpn_508_local0_3.pth" \
# --batch_size 6 \
# --sub_batch_size 6 \
# --patch_sizes 1350 721 508 \
# --size 508 \
# --origin_size 2448 \
# --training_level 0 \
# --lr 5e-5 \
# --reduce_step_size 50 \
# --early_stopping 120 \
# --num_epochs 120 \
# --reduce_factor 0.5 \
# --level_decay 0 \
# --lamb_fmreg 0.15 \
# --add_weight \
# --workers 10

export CUDA_VISIBLE_DEVICES=0
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
--task_name "eval_local0_1384" \
--restore_path "/vinai/chuonghm/inspector/logs/tuning_patchsize_1384/tuning_patchsize_1384.pth" \
--batch_size 6 \
--sub_batch_size 6 \
--patch_sizes 1384 721 508 \
--size 508 \
--origin_size 2448 \
--training_level 0 \
--lr 5e-5 \
--reduce_step_size 50 \
--early_stopping 120 \
--num_epochs 120 \
--reduce_factor 0.5 \
--level_decay 0 \
--lamb_fmreg 0.15 \
--add_weight \
--workers 10