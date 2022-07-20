# coding=UTF-8
python ../train.py \
--dataset_root ~/WZ/data \
--epochs 1000 \
--dataset_name miniImagenet \
--model_name tpn \
--classes_per_it_tr 5 \
--num_support_tr 5 \
--num_query_tr 15 \
--classes_per_it_val 5 \
--num_support_val 5 \
--num_query_val 15 \
--height 84 \
--width 84 \
--iterations 1000 \
--learning_rate 0.001 \
--cuda 1 \
--comment "tpn1"
