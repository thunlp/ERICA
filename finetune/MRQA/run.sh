python3 run_squad.py --model_type bert --model_name_or_path ***path_to_your_bert_model*** \
            --do_train  --do_eval  --evaluate_during_training  \
            --train_file $task/$train_file --predict_file $task/dev.jsonl     \
            --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 512 --doc_stride 128     \
            --per_gpu_eval_batch_size 4 --per_gpu_train_batch_size 1 --save_steps 1500 --overwrite_output_dir \
            --output_dir models_$task/$output_dir  --gradient_accumulation_steps 8 --do_lower_case --ckpt_to_load $ckpt_to_load