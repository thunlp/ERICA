python3 code/run_typing.py    --do_train   --do_lower_case  \
--data_dir data/$data   --ernie_model ernie_base   --max_seq_length 256  \
--train_batch_size $train_batch_size   --learning_rate $learning_rate   --num_train_epochs $num_train   \
--output_dir $output_dir  --gradient_accumulation_steps $accum   \
--loss_scale 128 --warmup_proportion 0.2 --train_file $train_file \
--ckpt $ckpt \
--mean_pool $mean_pool \
--bert_model $bert_model

python3 get_eval.py --output_dir $output_dir
