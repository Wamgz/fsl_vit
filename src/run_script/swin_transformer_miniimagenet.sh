# coding=UTF-8
python ../train.py \
--dataset_root ~/WZ/data \
--epochs 3000 \
--model_name swin_transformer \
--dataset_name miniImagenet \
--classes_per_it_tr 5 \
--num_support_tr 5 \
--num_query_tr 15 \
--classes_per_it_val 5 \
--num_support_val 5 \
--num_query_val 15 \
--height 64 \
--width 64 \
--iterations 100 \
--learning_rate 0.001 \
--cuda 3
