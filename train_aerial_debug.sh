export CUDA_VISIBLE_DEVICES=1
python train_aerial.py \
--n_class 2 \
--data_path "/vinai/chuonghm/aerial" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "debug_fpn_aerial_global" \
--mode 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 500 \
--size_p 500
# python train_aerial.py \
# --n_class 2 \
# --data_path "/vinai/chuonghm/aerial" \
# --model_path "/vinai/chuonghm/glnet/saved_models" \
# --log_path "/vinai/chuonghm/glnet/logs" \
# --task_name "debug_fpn_aerial_global2local" \
# --mode 2 \
# --batch_size 6 \
# --sub_batch_size 32 \
# --size_g 500 \
# --size_p 500 \
# --path_g "fpn_aerial_global.pth"