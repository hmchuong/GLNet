export CUDA_VISIBLE_DEVICES=1
python train_idrid.py \
--n_class 2 \
--data_path "/vinai/chuonghm/drcl" \
--train_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/train_EX.csv" \
--val_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/test_EX.csv" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "fpn_idrid_global" \
--mode 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 640 \
--size_p 640