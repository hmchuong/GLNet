export CUDA_VISIBLE_DEVICES=2,3,6,7
python -m torch.distributed.launch --nproc_per_node=4 --use_env inspector.py \
--dataset_name "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--n_class 7 \
--num_scaling_level 2 \
--log_path "/vinai/chuonghm/inspector/logs" \
--task_name "inspector_deepglobe_global" \
--batch_size 6 \
--sub_batch_size 6 \
--patch_sizes 1224 612 \
--size 612 \
--origin_size 2448 \
--training_level -1 \
--level_decay 0 \
--workers 10 --world-size 4