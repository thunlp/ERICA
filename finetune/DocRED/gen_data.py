import numpy as np
import os
import json
import argparse
from transformers import BertTokenizer, RobertaTokenizer
import random

MODEL_CLASSES = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
}

rel2id = {'P1376': 79, 'P607': 27, 'P136': 73, 'P137': 63, 'P131': 2, 'P527': 11, 'P1412': 38, 'P206': 33, 'P205': 77, 'P449': 52, 'P127': 34, 'P123': 49, 'P86': 66, 'P840': 85, 'P355': 72, 'P737': 93, 'P740': 84, 'P190': 94, 'P576': 71, 'P749': 68, 'P112': 65, 'P118': 40, 'P17': 1, 'P19': 14, 'P3373': 19, 'P6': 42, 'P276': 44, 'P1001': 24, 'P580': 62, 'P582': 83, 'P585': 64, 'P463': 18, 'P676': 87, 'P674': 46, 'P264': 10, 'P108': 43, 'P102': 17, 'P25': 81, 'P27': 3, 'P26': 26, 'P20': 37, 'P22': 30, 'Na': 0, 'P807': 95, 'P800': 51, 'P279': 78, 'P1336': 88, 'P577': 5, 'P570': 8, 'P571': 15, 'P178': 36, 'P179': 55, 'P272': 75, 'P170': 35, 'P171': 80, 'P172': 76, 'P175': 6, 'P176': 67, 'P39': 91, 'P30': 21, 'P31': 60, 'P36': 70, 'P37': 58, 'P35': 54, 'P400': 31, 'P403': 61, 'P361': 12, 'P364': 74, 'P569': 7, 'P710': 41, 'P1344': 32, 'P488': 82, 'P241': 59, 'P162': 57, 'P161': 9, 'P166': 47, 'P40': 20, 'P1441': 23, 'P156': 45, 'P155': 39, 'P150': 4, 'P551': 90, 'P706': 56, 'P159': 29, 'P495': 13, 'P58': 53, 'P194': 48, 'P54': 16, 'P57': 28, 'P50': 22, 'P1366': 86, 'P1365': 92, 'P937': 69, 'P140': 50, 'P69': 25, 'P1198': 96, 'P1056': 89}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bert")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--data_dir', type = str, default =  "docred_data")
    parser.add_argument('--output_dir', type = str, default = "prepro_data")
    parser.add_argument('--max_seq_length', type = int, default = 512)
    parser.add_argument('--ratio', type = float, default = 1.0)

    args = parser.parse_args()
    model_type = args.model_type
    data_dir = args.data_dir
    output_dir = args.output_dir
    max_seq_length = args.max_seq_length

    assert(model_type in ['bert', 'roberta'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_annotated_file_name = os.path.join(data_dir, 'train_annotated.json')
    dev_file_name = os.path.join(data_dir, 'dev.json')
    test_file_name = os.path.join(data_dir, 'test.json')

    fact_in_annotated_train = set([])

    tokenizer = MODEL_CLASSES[model_type].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    def save_data_format(ori_data, is_training):
        data = []
        for i in range(len(ori_data)):
            Ls = [0]
            L = 0
            for x in ori_data[i]['sents']:
                L += len(x)
                Ls.append(L)

            vertexSet =  ori_data[i]['vertexSet']
            # point position added with sent start position
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                    sent_id = vertexSet[j][k]['sent_id']
                    dl = Ls[sent_id]
                    pos1 = vertexSet[j][k]['pos'][0]
                    pos2 = vertexSet[j][k]['pos'][1]
                    vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)

            ori_data[i]['vertexSet'] = vertexSet

            item = {}
            item['vertexSet'] = vertexSet
            labels = ori_data[i].get('labels', [])

            train_triple = set([])
            new_labels = []
            for label in labels:
                rel = label['r']
                assert(rel in rel2id)
                label['r'] = rel2id[label['r']]

                train_triple.add((label['h'], label['t']))


                label['in_annotated_train'] = False

                if is_training:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            fact_in_annotated_train.add((n1['name'], n2['name'], rel))
                else:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                                if (n1['name'], n2['name'], rel) in fact_in_annotated_train:
                                    label['in_annotated_train'] = True


                new_labels.append(label)

            item['labels'] = new_labels
            item['title'] = ori_data[i]['title']

            na_triple = []
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet)):
                    if (j != k):
                        if (j, k) not in train_triple:
                            na_triple.append((j, k))

            item['na_triple'] = na_triple
            item['Ls'] = Ls
            item['sents'] = ori_data[i]['sents']
            data.append(item)
        return data

    def init(data_file_name, rel2id, args, max_seq_length = 512, is_training = True, suffix=''):
        ori_data = json.load(open(data_file_name))

        if args.ratio < 1 and suffix == 'train':
            random.shuffle(ori_data)
            print(len(ori_data))
            ori_data = ori_data[: int(args.ratio * len(ori_data))]
            print(len(ori_data))
        
        data = save_data_format(ori_data, is_training)
        print ('data_len:', len(data))

        print("Saving files")
        if args.ratio < 1 and suffix == 'train':
            json.dump(data , open(os.path.join(output_dir, suffix + '_' + str(args.ratio) +'.json'), "w"))
        else:
            json.dump(data , open(os.path.join(output_dir, suffix + '.json'), "w"))
        
        json.dump(rel2id , open(os.path.join(output_dir, 'rel2id.json'), "w"))

        sen_tot = len(ori_data)
        bert_token = np.zeros((sen_tot, max_seq_length), dtype = np.int64)
        bert_mask = np.zeros((sen_tot, max_seq_length), dtype = np.int64)
        bert_starts_ends = np.ones((sen_tot, max_seq_length, 2), dtype = np.int64) * (max_seq_length - 1)

        if model_type=='bert':
        
            for i in range(len(ori_data)):
                item = ori_data[i]
                tokens = []
                for sent in item['sents']:
                    tokens += sent

                subwords = list(map(tokenizer.tokenize, tokens))
                subword_lengths = list(map(len, subwords))
                flatten_subwords = [x for x_list in subwords for x in x_list ]

                tokens = [ tokenizer.cls_token ] + flatten_subwords[: max_seq_length - 2] + [ tokenizer.sep_token ]
                token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
                token_start_idxs[token_start_idxs >= max_seq_length-1] = max_seq_length - 1
                token_end_idxs = 1 + np.cumsum(subword_lengths)
                token_end_idxs[token_end_idxs >= max_seq_length-1] = max_seq_length - 1

                tokens = tokenizer.convert_tokens_to_ids(tokens)
                pad_len = max_seq_length - len(tokens)
                mask = [1] * len(tokens) + [0] * pad_len
                tokens = tokens + [0] * pad_len

                bert_token[i] = tokens
                bert_mask[i] = mask 

                bert_starts_ends[i, :len(subword_lengths), 0] = token_start_idxs
                bert_starts_ends[i, :len(subword_lengths), 1] = token_end_idxs
        else:
            for i in range(len(ori_data)):
                item = ori_data[i]
                words = []
                for sent in item['sents']:
                    words += sent

                idxs = []
                text = ""
                for word in words:
                    if len(text)>0:
                        text = text  + " "
                    idxs.append(len(text))
                    text += word
                
                subwords = tokenizer.tokenize(text)

                char2subwords = []
                L = 0
                sub_idx = 0
                L_subwords = len(subwords)
                while sub_idx < L_subwords:
                    subword_list = []
                    prev_sub_idx = sub_idx
                    while sub_idx < L_subwords:
                        subword_list.append(subwords[sub_idx])
                        sub_idx += 1

                        subword = tokenizer.convert_tokens_to_string(subword_list)
                        sub_l = len(subword)
                        if text[L:L+sub_l]==subword:
                            break
                    
                    assert(text[L:L+sub_l]==subword)
                    char2subwords.extend([prev_sub_idx] * sub_l)

                    L += len(subword)

                if len(text) > len(char2subwords):
                    text = text[:len(char2subwords)]

                assert(len(text)==len(char2subwords))
                tokens = [ tokenizer.cls_token ] + subwords[: max_seq_length - 2] + [ tokenizer.sep_token ]

                L_ori = len(tokens)
                tokens = tokenizer.convert_tokens_to_ids(tokens)

                pad_len = max_seq_length - len(tokens)
                mask = [1] * len(tokens) + [0] * pad_len
                tokens = tokens + [0] * pad_len

                bert_token[i] = tokens
                bert_mask[i] = mask 

                for j in range(len(words)):
                    idx = char2subwords[idxs[j]] + 1
                    idx = min(idx, max_seq_length-1)

                    x = idxs[j] + len(words[j])
                    if x == len(char2subwords):
                        idx2 = L_ori
                    else:
                        idx2 = char2subwords[x] + 1
                        idx2 = min(idx2, max_seq_length-1)

                    bert_starts_ends[i][j][0] = idx
                    bert_starts_ends[i][j][1] = idx2

        print("Finishing processing")
        if args.ratio < 1 and suffix == 'train':
            np.save(os.path.join(output_dir, suffix + '_' + str(args.ratio) + '_bert_token.npy'), bert_token)
            np.save(os.path.join(output_dir, suffix + '_' + str(args.ratio) + '_bert_mask.npy'), bert_mask)
            np.save(os.path.join(output_dir, suffix + '_' + str(args.ratio) + '_bert_starts_ends.npy'), bert_starts_ends)
        else:
            np.save(os.path.join(output_dir, suffix + '_bert_token.npy'), bert_token)
            np.save(os.path.join(output_dir, suffix + '_bert_mask.npy'), bert_mask)
            np.save(os.path.join(output_dir, suffix + '_bert_starts_ends.npy'), bert_starts_ends)
        print("Finish saving")


    init(train_annotated_file_name, rel2id, args, max_seq_length = max_seq_length, is_training = True, suffix='train')
    init(dev_file_name, rel2id, args, max_seq_length = max_seq_length, is_training = False, suffix='dev')
    init(test_file_name, rel2id, args, max_seq_length = max_seq_length, is_training = False, suffix='test')


if __name__ == '__main__':
    main()
