python3 train.py --model_type $model_type  --model_name_or_path $model_name_or_path  \
      --train_prefix train --test_prefix dev --evaluate_during_training_epoch $evaluate_during_training_epoch \
      --prepro_data_dir $prepro_data --max_seq_length 512 --batch_size $batch_size \
      --learning_rate 4e-5 --num_train_epochs $num_train_epochs --save_name $save_name \
      --ckpt $ckpt --ratio $ratio
