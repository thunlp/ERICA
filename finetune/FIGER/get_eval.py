import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default=None, type=str, required=True)
args = parser.parse_args()
filenames = os.listdir(args.output_dir)
eval_filenames = [x for x in filenames if "eval_results_" in x]
test_filenames = [x for x in filenames if "test_results_" in x]
eval_filenames = {x.split('s_')[1]: x for x in eval_filenames}
test_filenames = {x.split('s_')[1]: x for x in test_filenames}
def get_eval(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    acc = lines[0].split('= ')[1]
    macro = lines[2].split('= ')[1]
    micro = lines[3].split('= ')[1]
    return [acc, macro, micro]
best_acc = 0
o1 = None
for file_name in eval_filenames:
    e = get_eval(args.output_dir + '/eval_results_' + file_name)
    if float(e[0]) > best_acc:
        best_acc = float(e[0])
        o1 = [file_name, e]
        o2 = [file_name, get_eval(args.output_dir + '/test_results_' + file_name)]
print({'epoch': o2[0], 'acc': o2[1][0], 'macro': o2[1][1], 'micro': o2[1][2]})