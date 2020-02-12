# export CUDA_VISIBLE_DEVICES=2,3,4,6
# python -m torch.distributed.launch --nproc_per_node=4 --use_env inspector.py \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --refinement 1 \
# --glob2local \
# --supervision \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "tuning_patchsize_1384" \
# --restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_global_resnetfpn_508/inspector_deepglobe_global_resnetfpn_508.pth" \
# --batch_size 6 \
# --sub_batch_size 6 \
# --patch_sizes 1384 721 508 \
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
# --dist-url "tcp://127.0.0.1:1240" \
# --workers 10 --world-size 4

export CUDA_VISIBLE_DEVICES=0,4,5,6

python -m torch.distributed.launch --nproc_per_node=4 --use_env inspector.py \
--dataset_name "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--n_class 7 \
--num_scaling_level 3 \
--refinement 1 \
--glob2local \
--supervision \
--log_path "/vinai/chuonghm/inspector/logs" \
--task_name "tuning_patchsize_1400_1" \
--restore_path "/vinai/chuonghm/inspector/logs/tuning_patchsize_1400/tuning_patchsize_1400.pth" \
--continue_train \
--batch_size 16 \
--sub_batch_size 16 \
--patch_sizes 1400 721 508 \
--size 508 \
--origin_size 2448 \
--training_level 0 \
--lr 2e-5 \
--reduce_step_size 50 \
--early_stopping 120 \
--num_epochs 120 \
--reduce_factor 0.5 \
--level_decay 0 \
--lamb_fmreg 0.15 \
--add_weight \
--dist-url "tcp://127.0.0.1:1241" \
--workers 10 --world-size 4