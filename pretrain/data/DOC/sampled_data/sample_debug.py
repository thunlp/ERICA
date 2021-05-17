import json
data = json.load(open('train_distant_0.json', 'r'))
data = data[: 1000]
json.dump(data, open('path_to_your_pre-train_data/train_distant_debug.json', 'w'))