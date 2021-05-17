This is the fine-tuning implementation for DocRED. 

We modify the [official code](https://github.com/thunlp/DocRED) to implement BERT-based models.

Download the [DocRED dataset](https://github.com/thunlp/DocRED/tree/master/data) and put them into the folder 'docred_data'.

Preprocess the data:
```
python3 gen_data.py --model_type bert --model_name_or_path bert-base-uncased --data_dir data --output_dir prepro_data --max_seq_length 512 --do_lower_case
```

Train:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --model_type $model_type  --model_name_or_path $model_name_or_path  \
      --train_prefix train --test_prefix dev --evaluate_during_training_epoch $evaluate_during_training_epoch \
      --prepro_data_dir $prepro_data --max_seq_length 512 --batch_size $batch_size \
      --learning_rate 4e-5 --num_train_epochs $num_train_epochs --save_name $save_name \
      --ckpt $ckpt --ratio $ratio
```