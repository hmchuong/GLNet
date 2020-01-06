CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --use_env glnet.py \
--n_class 7 \
--dataset "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "fpn_deepglobe_global_new" \
--mode 1 \
--batch_size 32 \
--sub_batch_size 32 \
--size_g 508 \
--size_p 508 \
--workers 10 --world-size 4