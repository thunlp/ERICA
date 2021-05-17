#! /bin/bash
mkdir -p MRQA_train_data
mkdir -p MRQA_dev_data
mkdir -p squad
mkdir -p triviaqa
mkdir -p naturalqa

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz -O MRQA_train_data/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz -O MRQA_train_data/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz -O MRQA_train_data/NaturalQuestions.jsonl.gz

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz -O MRQA_dev_data/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz -O MRQA_dev_data/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz -O MRQA_dev_data/NaturalQuestions.jsonl.gz

gzip -d MRQA_train_data/SQuAD.jsonl.gz
gzip -d MRQA_dev_data/SQuAD.jsonl.gz

gzip -d MRQA_train_data/TriviaQA.jsonl.gz
gzip -d MRQA_dev_data/TriviaQA.jsonl.gz

gzip -d MRQA_train_data/NaturalQuestions.jsonl.gz
gzip -d MRQA_dev_data/NaturalQuestions.jsonl.gz