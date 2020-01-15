# Train global
# export CUDA_VISIBLE_DEVICES=0,2,3,6
# python -m torch.distributed.launch --nproc_per_node=4 --use_env inspector.py \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "inspector_deepglobe_global_resnetfpn" \
# --batch_size 16 \
# --sub_batch_size 16 \
# --patch_sizes 1224 996 640 \
# --size 640 \
# --origin_size 2448 \
# --training_level -1 \
# --level_decay 0 \
# --dist-url "env://" \
# --workers 10 --world-size 4

# Train local 0
export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch --nproc_per_node=3 --use_env inspector.py \
--dataset_name "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--n_class 7 \
--num_scaling_level 3 \
--refinement 4 \
--log_path "/vinai/chuonghm/inspector/logs" \
--task_name "inspector_deepglobe_local0_resnetfpn_refinement4_weight_reg" \
--restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_global_resnetfpn/inspector_deepglobe_global_resnetfpn.pth" \
--batch_size 3 \
--sub_batch_size 3 \
--patch_sizes 1224 996 640 \
--size 640 \
--origin_size 2448 \
--training_level 0 \
--level_decay 0 \
# --reduce_step_size 10 \
# --early_stopping 50 \
# --num_epochs 100 \
--lamb_fmreg 0.15 \
--add_weight \
--dist-url "tcp://127.0.0.1:1240" \
--workers 10 --world-size 3

# Train local 1
# export CUDA_VISIBLE_DEVICES=2,3,5,6
# python -m torch.distributed.launch --nproc_per_node=4 --use_env inspector.py \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "inspector_deepglobe_local1_resnetfpn" \
# --restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_local0_resnetfpn/inspector_deepglobe_local0_resnetfpn.pth" \
# --batch_size 4 \
# --sub_batch_size 4 \
# --patch_sizes 1224 996 640 \
# --size 640 \
# --origin_size 2448 \
# --training_level 1 \
# --level_decay 0 \
# --reduce_step_size 10 \
# --early_stopping 50 \
# --num_epochs 100 \
# --dist-url "tcp://127.0.0.1:1236" \
# --workers 10 --world-size 4