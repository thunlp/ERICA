array=(42 43 44 45 46)
#ckpt="ckpt_cp/ckpt_of_step_1000"
#ckpt="ckpt_noise/ckpt_of_step_3500"
train_prop=0.01
max_epoch=20
dataset="tacred"
for seed in ${array[@]}
do
	python main.py 	--seed $seed \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch $max_epoch \
	--max_length 100 \
	--mode CM \
	--dataset $dataset \
	--entity_marker --ckpt_to_load $ckpt \
	--train_prop $train_prop
done

train_prop=0.1
max_epoch=20
dataset="tacred"
for seed in ${array[@]}
do
	python main.py 	--seed $seed \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch $max_epoch \
	--max_length 100 \
	--mode CM \
	--dataset $dataset \
	--entity_marker --ckpt_to_load $ckpt \
	--train_prop $train_prop
done

train_prop=1
max_epoch=8
dataset="tacred"
for seed in ${array[@]}
do
	python main.py	--seed $seed \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch $max_epoch \
	--max_length 100 \
	--mode CM \
	--dataset $dataset \
	--entity_marker --ckpt_to_load $ckpt \
	--train_prop $train_prop
done

train_prop=0.01
max_epoch=50
dataset="semeval"
for seed in ${array[@]}
do
	python main.py 	--seed $seed \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch $max_epoch \
	--max_length 100 \
	--mode CM \
	--dataset $dataset \
	--entity_marker --ckpt_to_load $ckpt \
	--train_prop $train_prop
done

train_prop=0.1
max_epoch=50
dataset="semeval"
for seed in ${array[@]}
do
	python main.py 	--seed $seed \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch $max_epoch \
	--max_length 100 \
	--mode CM \
	--dataset $dataset \
	--entity_marker --ckpt_to_load $ckpt \
	--train_prop $train_prop
done

train_prop=1
max_epoch=8
dataset="semeval"
for seed in ${array[@]}
do
	python main.py 	--seed $seed \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch $max_epoch \
	--max_length 100 \
	--mode CM \
	--dataset $dataset \
	--entity_marker --ckpt_to_load $ckpt \
	--train_prop $train_prop
done