export OMP_NUM_THREADS=8
python -m torch.distributed.launch --nproc_per_node=4 --use_env glnet.py \
--n_class 7 \
--dataset "DeepGlobe" \
--data_path "/chuong/deep_globe" \
--model_path "/chuong/saved_models" \
--log_path "/chuong/logs" \
--task_name "fpn_deepglobe_global2local_new" \
--mode 1 \
--batch_size 10 \
--sub_batch_size 10 \
--size_g 508 \
--size_p 508 \
--workers 7 --world-size 4