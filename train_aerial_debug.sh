export CUDA_VISIBLE_DEVICES=3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=5 --use_env train_aerial.py \
--n_class 2 \
--data_path "/vinai/chuonghm/aerial" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "debug_fpn_aerial_global" \
--mode 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 536 \
--size_p 536 \
--workers 5 --world-size 5

# python train_aerial.py \
# --n_class 2 \
# --data_path "/vinai/chuonghm/aerial" \
# --model_path "/vinai/chuonghm/glnet/saved_models" \
# --log_path "/vinai/chuonghm/glnet/logs" \
# --task_name "debug_fpn_aerial_global" \
# --mode 1 \
# --batch_size 6 \
# --sub_batch_size 6 \
# --size_g 500 \
# --size_p 500

# python train_aerial.py \
# --n_class 2 \
# --data_path "/vinai/chuonghm/aerial" \
# --model_path "/vinai/chuonghm/glnet/saved_models" \
# --log_path "/vinai/chuonghm/glnet/logs" \
# --task_name "debug_fpn_aerial_global2local" \
# --mode 2 \
# --batch_size 6 \
# --sub_batch_size 30 \
# --size_g 500 \
# --size_p 500 \
# --path_g "fpn_aerial_global.pth"

# python train_aerial.py \
# --n_class 2 \
# --data_path "/vinai/chuonghm/aerial" \
# --model_path "/vinai/chuonghm/glnet/saved_models" \
# --log_path "/vinai/chuonghm/glnet/logs" \
# --task_name "debug_fpn_aerial_local2global" \
# --mode 3 \
# --batch_size 6 \
# --sub_batch_size 96 \
# --size_g 500 \
# --size_p 500 \
# --path_g "fpn_aerial_global.pth" \
# --path_g2l "fpn_aerial_global2local.pth"