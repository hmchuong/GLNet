export CUDA_VISIBLE_DEVICES=0,2,3
python -m torch.distributed.launch --nproc_per_node=3 --use_env inspector.py \
--dataset_name "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--n_class 7 \
--num_scaling_level 3 \
--log_path "/vinai/chuonghm/inspector/logs" \
--task_name "inspector_deepglobe_global_fcn" \
--batch_size 3 \
--sub_batch_size 3 \
--patch_sizes 1224 996 640 \
--size 640 \
--origin_size 2448 \
--training_level -1 \
--level_decay 0 \
--dist-url "env://" \
--workers 10 --world-size 3