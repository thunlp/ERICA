This directory contains code and data for downstream sentence-level RE (semeval and TACRED).

#### 1 Dataset

You need to download TACRED from [LDC](https://catalog.ldc.upenn.edu/LDC2018T24) manually. 
For the processed semeval dataset, please download from this repo: https://github.com/thunlp/RE-Context-or-Names

Please ensure every dataset has `train.txt`, `dev.txt`,`test.txt`and `rel2id.json`(**NA must be 0 if this benchmark has NA relation**). And `train.txt`(the same as `dev.txt`, `text.txt`) should have multiple lines, each line has the following json-format:

```python
{
    "tokens":["Microsoft", "was", "founded", "by", "Bill", "Gates", "."], 
    "h":{
        "name": "Microsotf", "pos":[0,1]  # Left closed and right open interval
    }
    "t":{
        "name": "Bill Gates", "pos":[4,6] # Left closed and right open interval
    }
    "relation": "founded_by"
}
```

#### 2 Train

Run the following scirpt:

```shell
bash run.sh
```

If you want to use different model, you can change `ckpt` in `run.sh`

```shell
array=(42 43 44 45 46)
#ckpt="ckpt_cp/ckpt_of_step_3500"
#ckpt="None"
#ckpt="MTB"
#ckpt="ckpt_cp_2/ckpt_of_step_3500"
#ckpt="mtb/ckpt_of_step_45000"
#ckpt="ckpt_cp_r/ckpt_of_step_4500"
#ckpt="ckpt_cp_no_marker/ckpt_of_step_1500"
ckpt="None"
cuda=2
train_prop=0.01
max_epoch=20
dataset="tacred"
for seed in ${array[@]}
do
	python main.py --cuda $cuda \
	--seed $seed \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch $max_epoch \
	--max_length 100 \
	--mode CM \
	--dataset $dataset \
	--entity_marker --ckpt_to_load $ckpt \
	--train_prop $train_prop
done
```

"None" means Bert. You can use any checkpoint in `../pretrain/ckpt` directory for finetuning.\

Note we modify the official code of this repo: https://github.com/thunlp/RE-Context-or-Names to implement sentence-level RE tasks and keep all the experimental settings almost the same for fair comparison.