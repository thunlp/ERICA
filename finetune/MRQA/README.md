## MRQA

This folder is the implementation for MRQA (squad / triviaqa / naturalqa).

Download the [MRQA dataset](https://github.com/mrqa/MRQA-Shared-Task-2019) and sort out the datasets:
```
sh download_data.sh
```

Randomly split the development set into two halves to generate new validation and test sets:
```
python3 split.py
```

Train (e.g. on SQuAD):
```
python3 run_squad.py --model_type bert --model_name_or_path ***path_to_your_bert_model*** \
            --do_train  --do_eval  --evaluate_during_training  \
            --train_file $task/$train_file --predict_file $task/dev.jsonl     \
            --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 512 --doc_stride 128     \
            --per_gpu_eval_batch_size 4 --per_gpu_train_batch_size 1 --save_steps 1500 --overwrite_output_dir \
            --output_dir models_$task/$output_dir  --gradient_accumulation_steps 8 --do_lower_case --ckpt_to_load $ckpt_to_load
```

ckpt_to_load is your trained model parameters.

Manualy select the best checkpoint on development set based on the training log and evaluate it on test set:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_squad.py --do_eval --model_type bert --model_name_or_path $model_path --train_file squad/train.jsonl --predict_file squad/test.jsonl --doc_stride 128 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 512 --save_steps 1500 --output_dir $output_dir
```