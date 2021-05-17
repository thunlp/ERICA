python train.py --model_name_or_path ***path_to_your_bert_model***/uncased_L-12_H-768_A-12 \
--output_dir $output_dir --do_train --do_eval \
--evaluate_during_training  --do_lower_case \
--model_type bert --overwrite_output_dir \
--num_train_epochs 2 \
--train_file train.masked_0.01.json \
--predict_file dev.masked.json \
--ckpt_to_load $ckpt_to_load \
--max_doc_len 4000 \
--doc_stride 0

python train.py --model_name_or_path ***path_to_your_bert_model***/uncased_L-12_H-768_A-12 \
--output_dir $output_dir --do_train --do_eval \
--evaluate_during_training  --do_lower_case \
--model_type bert --overwrite_output_dir \
--num_train_epochs 2 \
--train_file train.masked_0.1.json \
--predict_file dev.masked.json \
--ckpt_to_load $ckpt_to_load \
--max_doc_len 4000 \
--doc_stride 0

python train.py --model_name_or_path ***path_to_your_bert_model***/uncased_L-12_H-768_A-12 \
--output_dir $output_dir --do_train --do_eval \
--evaluate_during_training  --do_lower_case \
--model_type bert --overwrite_output_dir \
--num_train_epochs 2 \
--train_file train.masked_0.5.json \
--predict_file dev.masked.json \
--ckpt_to_load $ckpt_to_load \
--max_doc_len 4000 \
--doc_stride 0
