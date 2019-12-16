export CUDA_VISIBLE_DEVICES=1
python train_aerial.py \
--n_class 2 \
--data_path "/vinai/chuonghm/aerial" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "eval_aerial" \
--mode 3 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 500 \
--size_p 500 \
--path_g "fpn_aerial_global.pth" \
--path_g2l "fpn_aerial_global2local.pth" \
--path_l2g "fpn_aerial_local2global.pth" \
--evaluation \
--test