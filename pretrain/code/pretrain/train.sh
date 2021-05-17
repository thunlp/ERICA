python -m torch.distributed.launch --nproc_per_node 8  main.py  \
    --model DOC  --lr 3e-5 --batch_size_per_gpu 16 --max_epoch 105  \
    --gradient_accumulation_steps 16    --save_step 500  --temperature 0.05  \
    --train_sample  --save_dir ckpt_doc_dw_f_alpha_1_uncased --n_gpu 8  --debug 1  --add_none 1 \
    --alpha 1 --flow 0 --dataset_name none.json  --wiki_loss 1 --doc_loss 1 \
    --change_dataset 1  --start_end_token 0 --bert_model bert \
    --pretraining_size -1 --ablation 0 --cased 0
