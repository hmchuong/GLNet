# Train local 0
# export CUDA_VISIBLE_DEVICES=3,4,5,6
# python -m torch.distributed.launch --nproc_per_node=4 --use_env inspector.py \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --refinement 1 \
# --glob2local \
# --supervision \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "inspector_deepglobe_global_resnetfpn_508" \
# --batch_size 16 \
# --sub_batch_size 16 \
# --patch_sizes 1350 702 508 \
# --size 508 \
# --origin_size 2448 \
# --training_level -1 \
# --lr 5e-5 \
# --reduce_step_size 120 \
# --early_stopping 120 \
# --num_epochs 120 \
# --reduce_factor 1.0 \
# --level_decay 0 \
# --lamb_fmreg 0.15 \
# --add_weight \
# --dist-url "tcp://127.0.0.1:1234" \
# --workers 10 --world-size 4

# Train local 0
# export CUDA_VISIBLE_DEVICES=0,1,2,5,6
# python -m torch.distributed.launch --nproc_per_node=5 --use_env inspector.py \
# --dataset_name "DeepGlobe" \
# --data_path "/vinai/chuonghm/deep_globe" \
# --n_class 7 \
# --num_scaling_level 3 \
# --refinement 1 \
# --glob2local \
# --supervision \
# --log_path "/vinai/chuonghm/inspector/logs" \
# --task_name "inspector_deepglobe_global_resnetfpn_508_local0_3" \
# --restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_global_resnetfpn_508_local0_2/inspector_deepglobe_global_resnetfpn_508_local0_2.pth" \
# --batch_size 10 \
# --sub_batch_size 10 \
# --patch_sizes 1350 702 508 \
# --size 508 \
# --origin_size 2448 \
# --training_level 0 \
# --lr 5e-5 \
# --reduce_step_size 50 \
# --early_stopping 120 \
# --num_epochs 120 \
# --reduce_factor 0.4 \
# --level_decay 0 \
# --lamb_fmreg 0.15 \
# --add_weight \
# --dist-url "tcp://127.0.0.1:1234" \
# --workers 10 --world-size 5

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,7
python -m torch.distributed.launch --nproc_per_node=6 --use_env inspector.py \
--dataset_name "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--n_class 7 \
--num_scaling_level 3 \
--refinement 1 \
--glob2local \
--supervision \
--log_path "/vinai/chuonghm/inspector/logs" \
--task_name "inspector_deepglobe_global_resnetfpn_508_local2" \
--restore_path "/vinai/chuonghm/inspector/logs/inspector_deepglobe_global_resnetfpn_508_local1/inspector_deepglobe_global_resnetfpn_508_local1.pth" \
--batch_size 6 \
--sub_batch_size 6 \
--patch_sizes 1350 702 508 \
--size 508 \
--origin_size 2448 \
--training_level 2 \
--lr 5e-5 \
--reduce_step_size 50 \
--early_stopping 120 \
--num_epochs 120 \
--reduce_factor 0.5 \
--level_decay 0 \
--lamb_fmreg 0.15 \
--add_weight \
--dist-url "tcp://127.0.0.1:1235" \
--workers 10 --world-size 6