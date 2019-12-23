export CUDA_VISIBLE_DEVICES=1
python train_aerial.py \
--n_class 2 \
--data_path "/chuong/aerial" \
--model_path "/chuong/saved_models" \
--log_path "/chuong/logs" \
--task_name "eval_aerial" \
--mode 3 \
--batch_size 2 \
--sub_batch_size 2 \
--size_g 536 \
--size_p 536 \
--path_g "fpn_aerial_global_new.pth" \
--path_g2l "fpn_aerial_global2local_new.pth" \
--path_l2g "fpn_aerial_local2global_new.pth" \
--evaluation