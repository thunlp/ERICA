# ERICA

Pre-training for ERICA: Improving Entity and Relation Understanding for Pre-trained Language Models via Contrastive Learning. We'll further organize the codes and publish all codes / pre-training data / trained model parameters in future.

### Dependencies

Run the following script to install dependencies.

```shell
pip install -r requirement.txt
```

**You need to install transformers and apex manually.**

**transformers**
You should install transformers manually. We use huggingface transformers to implement Bert and RoBERTa, and the version is 2.5.0. You need to clone or download [transformers repo](https://github.com/huggingface/transformers). And for convenience, we have downloaded transformers into `code/pretrain/` so you can easily import it, and we have also modified some lines in the class `BertForMaskedLM` in `src/transformers/modeling_bert.py` while keeping the other codes unchanged.

**apex**
Install [apex](https://github.com/NVIDIA/apex) under the offical guidance.

### process pretraining data
In folder prepare_pretrain_data, we provide the codes for processing pre-training data.

### Pretrain

You can use this repo to pretrain a new model. To pretrain ERICA_bert:

```shell
cd code/pretrain

python -m torch.distributed.launch --nproc_per_node 8  main.py  \
    --model DOC  --lr 3e-5 --batch_size_per_gpu 16 --max_epoch 105  \
    --gradient_accumulation_steps 16    --save_step 500  --temperature 0.05  \
    --train_sample  --save_dir ckpt_doc_dw_f_alpha_1_uncased --n_gpu 8  --debug 1  --add_none 1 \
    --alpha 1 --flow 0 --dataset_name none.json  --wiki_loss 1 --doc_loss 1 \
    --change_dataset 1  --start_end_token 0 --bert_model bert \
    --pretraining_size -1 --ablation 0 --cased 0
```

some explanations for hyper-parameters:
temperature: \tau used in loss function of contrastive learning
debug: whether to debug (we provide an example_debug file for pre-training)
add_none: whether to add no_relation pair in RD loss.
alpha: the proportion of masking (1 means no masking, in experiments, we find masking is not helpful as is described in the main paper, so for all models, we do not mask in the pre-training phase. However, we leave this function here for further research explorations.)
flow: if masking, whether to use a linear decay
wiki_loss: whether to add ED loss.
doc_loss: whether to add RD loss.
start_end_token: use another entity encoding method
cased: whether to use cased version of BERT
