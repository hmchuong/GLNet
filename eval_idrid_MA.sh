export CUDA_VISIBLE_DEVICES=1
python train_idrid.py \
--n_class 2 \
--data_path "/vinai/chuonghm/drcl" \
--train_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/train_MA.csv" \
--val_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/test_MA.csv" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "eval_idrid_MA" \
--mode 3 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 640 \
--size_p 640 \
--path_g "fpn_idrid_global_MA.pth" \
--path_g2l "fpn_idrid_global2local_MA.pth" \
--path_l2g "fpn_idrid_local2global_MA.pth" \
--evaluation \
--test