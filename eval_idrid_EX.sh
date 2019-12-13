export CUDA_VISIBLE_DEVICES=1
python train_idrid.py \
--n_class 2 \
--data_path "/vinai/chuonghm/drcl" \
--train_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/train_EX.csv" \
--val_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/test_EX.csv" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "eval_idrid_EX" \
--mode 3 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 640 \
--size_p 640 \
--path_g "fpn_idrid_global_EX.pth" \
--path_g2l "fpn_idrid_global2local_EX.pth" \
--path_l2g "fpn_idrid_local2global_EX.pth" \
--evaluation \
--test