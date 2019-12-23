export OMP_NUM_THREADS=8
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_aerial.py \
--n_class 2 \
--data_path "/chuong/aerial" \
--model_path "/chuong/saved_models" \
--log_path "/chuong/logs" \
--task_name "fpn_aerial_local2global_new" \
--mode 3 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 536 \
--size_p 536 \
--path_g "fpn_aerial_global_new.pth" \
--path_g2l "fpn_aerial_global2local_new.pth" \
--workers 6 --world-size 4