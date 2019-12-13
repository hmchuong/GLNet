export CUDA_VISIBLE_DEVICES=0
python train_idrid.py \
--n_class 2 \
--data_path "/vinai/chuonghm/drcl" \
--train_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/train_SE.csv" \
--val_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/test_SE.csv" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "fpn_idrid_global_SE" \
--mode 1 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 640 \
--size_p 640

python train_idrid.py \
--n_class 2 \
--data_path "/vinai/chuonghm/drcl" \
--train_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/train_SE.csv" \
--val_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/test_SE.csv" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "fpn_idrid_global2local_SE" \
--mode 2 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 640 \
--size_p 640 \
--path_g "fpn_idrid_global_SE.pth"

python train_idrid.py \
--n_class 2 \
--data_path "/vinai/chuonghm/drcl" \
--train_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/train_SE.csv" \
--val_csv "/vinai/chuonghm/drcl/dataset/nonprocessed_extend/segmentation/test_SE.csv" \
--model_path "/vinai/chuonghm/glnet/saved_models" \
--log_path "/vinai/chuonghm/glnet/logs" \
--task_name "fpn_idrid_local2global_SE" \
--mode 3 \
--batch_size 6 \
--sub_batch_size 6 \
--size_g 640 \
--size_p 640 \
--path_g "fpn_idrid_global_SE.pth" \
--path_g2l "fpn_idrid_global2local_SE.pth"