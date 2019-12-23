export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--n_class 7 \
--data_path "/chuong/deep_globe/" \
--model_path "/chuong/saved_models/" \
--log_path "/chuong/logs/" \
--task_name "eval_deepglob" \
--mode 3 \
--batch_size 1 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_deepglobe_global.pth" \
--path_g2l "fpn_deepglobe_global2local.pth" \
--path_l2g "fpn_deepglobe_local2global.pth" \
--evaluation