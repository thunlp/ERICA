python train.py --model_name_or_path ***path_to_your_roberta_model***/roberta-base \
--output_dir $output_dir --do_train  \
--evaluate_during_training  \
--model_type roberta --overwrite_output_dir \
--num_train_epochs 2 \
--train_file train.masked.json \
--predict_file dev.masked.json \
--ckpt_to_load $ckpt_to_load \
--max_doc_len 4000 \
--doc_stride 0
