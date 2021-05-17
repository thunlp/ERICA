## fine-tuning for FIGER (entity typing task)

We modify the code of "[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)" for implementation.

### Reqirements:

* Pytorch>=0.4.1
* Python3
* tqdm
* boto3
* requests
* apex (If you want to use fp16, you should make sure the commit is `79ad5a88e91434312b43b4a89d66226be5f2cc98`.)

### Fine-tune

We use the dataset processed by ERNIE. They use [TAGME](<https://tagme.d4science.org/tagme/>) to extract the entity mentions in the sentences and link them to their corresponding entitoes in KGs. They provide the annotated datasets [Google Drive](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/32668247e4fd4f9789f2/).

```shell
tar -xvzf data.tar.gz
```

In the root directory of the project, run the following codes to fine-tune different models on FIGER.

```

**FIGER:**

```bash
python3 code/run_typing.py    --do_train   --do_lower_case  \
--data_dir data/Wiki   --ernie_model ernie_base   --max_seq_length 256  \
--train_batch_size 256   --learning_rate 3e-5   --num_train_epochs 3   \
--output_dir $output_dir  --gradient_accumulation_steps 8   \
--loss_scale 128 --warmup_proportion 0.2 --train_file train.json \
--ckpt $ckpt \
--mean_pool 1 \
--bert_model bert

python3 get_eval.py --output_dir $output_dir
```

ckpt is your trained model.