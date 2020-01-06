CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --use_env glnet.py \
--n_class 7 \
--dataset "DeepGlobe" \
--data_path "/vinai/chuonghm/deep_globe" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "fpn_deepglobe_global2local_new_origin" \
--mode 2 \
--batch_size 20 \
--sub_batch_size 20 \
--size_g 508 \
--size_p 508 \
--path_g "fpn_deepglobe_global_new.pth" \
--workers 10 --world-size 4