export CUDA_VISIBLE_DEVICES=1
python train_idrid.py \
--n_class 2 \
--data_path "/chuong/drcl" \
--model_path "/chuong/saved_models" \
--train_csv "/chuong/drcl/dataset/nonprocessed_extend/segmentation/train_EX.csv" \
--val_csv "/chuong/drcl/dataset/nonprocessed_extend/segmentation/test_EX.csv" \
--log_path "/chuong/logs" \
--task_name "eval_idrid_EX" \
--mode 3 \
--batch_size 2 \
--sub_batch_size 2 \
--size_g 640 \
--size_p 640 \
--path_g "fpn_idrid_global_EX.pth" \
--path_g2l "fpn_idrid_global2local_EX.pth" \
--path_l2g "fpn_idrid_local2global_EX.pth" \
--evaluation \
--test