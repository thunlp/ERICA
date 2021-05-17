import json
import os
import random
import shutil

output_data_dirs = ["squad", "newsqa", "triviaqa", "searchqa", "hotpotqa", "naturalqa"]
data_files = ["SQuAD.jsonl", "NewsQA.jsonl", "TriviaQA.jsonl", "SearchQA.jsonl", "HotpotQA.jsonl", "NaturalQuestions.jsonl"]

for data_file, output_dir in zip(data_files, output_data_dirs):
    shutil.copy(os.path.join("MRQA_train_data", data_file), os.path.join(output_dir, "train.jsonl"))

    reader = open(os.path.join("MRQA_dev_data", data_file))

    reader.readline()
    data = []
    for x in reader:
        data.append(x)

    random.shuffle(data)

    L = len(data)
    print (output_dir, L)
    dev_data = data[:L//2]
    test_data = data[L//2:]

    w = open(os.path.join(output_dir, "dev.jsonl"), "w")
    w.write( json.dumps({"header": {"dataset": output_dir, "split": "dev"}} ) + '\n')
    for x in dev_data:
        w.write(x)

    w = open(os.path.join(output_dir, "test.jsonl"), "w")
    w.write( json.dumps({"header": {"dataset": output_dir, "split": "test"}} ) + '\n')
    for x in test_data:
        w.write(x)
