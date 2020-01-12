export CUDA_VISIBLE_DEVICES=3
python glnet.py \
--n_class 7 \
--dataset "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "eval_deepglobe_global2local" \
--mode 3 \
--batch_size 1 \
--sub_batch_size 6 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_deepglobe_global.pth" \
--path_g2l "fpn_deepglobe_global2local.pth" \
--path_l2g "fpn_deepglobe_local2global.pth" \
--evaluation