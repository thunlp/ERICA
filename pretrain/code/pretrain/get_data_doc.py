import numpy as np
import os
import json
import argparse
import random

rel2id = {'P1376': 79, 'P607': 27, 'P136': 73, 'P137': 63, 'P131': 2, 'P527': 11, 'P1412': 38, 'P206': 33, 'P205': 77, 'P449': 52, 'P127': 34, 'P123': 49, 'P86': 66, 'P840': 85, 'P355': 72, 'P737': 93, 'P740': 84, 'P190': 94, 'P576': 71, 'P749': 68, 'P112': 65, 'P118': 40, 'P17': 1, 'P19': 14, 'P3373': 19, 'P6': 42, 'P276': 44, 'P1001': 24, 'P580': 62, 'P582': 83, 'P585': 64, 'P463': 18, 'P676': 87, 'P674': 46, 'P264': 10, 'P108': 43, 'P102': 17, 'P25': 81, 'P27': 3, 'P26': 26, 'P20': 37, 'P22': 30, 'Na': 0, 'P807': 95, 'P800': 51, 'P279': 78, 'P1336': 88, 'P577': 5, 'P570': 8, 'P571': 15, 'P178': 36, 'P179': 55, 'P272': 75, 'P170': 35, 'P171': 80, 'P172': 76, 'P175': 6, 'P176': 67, 'P39': 91, 'P30': 21, 'P31': 60, 'P36': 70, 'P37': 58, 'P35': 54, 'P400': 31, 'P403': 61, 'P361': 12, 'P364': 74, 'P569': 7, 'P710': 41, 'P1344': 32, 'P488': 82, 'P241': 59, 'P162': 57, 'P161': 9, 'P166': 47, 'P40': 20, 'P1441': 23, 'P156': 45, 'P155': 39, 'P150': 4, 'P551': 90, 'P706': 56, 'P159': 29, 'P495': 13, 'P58': 53, 'P194': 48, 'P54': 16, 'P57': 28, 'P50': 22, 'P1366': 86, 'P1365': 92, 'P937': 69, 'P140': 50, 'P69': 25, 'P1198': 96, 'P1056': 89}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type = str, default = "path_to_this_folder/REAnalysis_doc/data/DOC")
    parser.add_argument('--alpha', type = float, default = 0.3)

    args = parser.parse_args()
    data_dir = '../../data/DOC'
    output_dir = args.output_dir
    max_seq_length = 512

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_annotated_file_name = os.path.join(data_dir, 'train_distant.json')
    dev_file_name = os.path.join(data_dir, 'dev.json')
    test_file_name = os.path.join(data_dir, 'test.json')

    removed_facts = set([])

    def save_data_format(ori_data, is_training):
        c1 = 0
        c2 = 100000
        data = []
        for i in range(len(ori_data)):
            if i > 1000:
                break
            vertexSet =  ori_data[i]['vertexSet']
            if is_training:
                sent_id_to_vertex = {}
                for jj in range(len(vertexSet)):
                    for k in range(len(vertexSet[jj])):
                        sent_id = int(vertexSet[jj][k]['sent_id'])
                        if sent_id not in sent_id_to_vertex:
                            sent_id_to_vertex[sent_id] = []
                        sent_id_to_vertex[sent_id].append([jj, k])
                for j in range(len(vertexSet)):
                    if random.random() > args.alpha:
                        for k in range(len(vertexSet[j])):
                            sent_id = int(vertexSet[j][k]['sent_id'])
                            sent = ori_data[i]['sents'][sent_id]
                            pos1 = vertexSet[j][k]['pos'][0]
                            pos2 = vertexSet[j][k]['pos'][1]
                            for kk in sent[pos1: pos2]:
                                assert kk in vertexSet[j][k]['name']
                            ori_data[i]['sents'][sent_id] = sent[: pos1] + ['[unused1]'] + sent[pos2: ]
                            for x,y in sent_id_to_vertex[sent_id]:
                                if vertexSet[x][y]['pos'][0] >= pos2:
                                    vertexSet[x][y]['pos'][0] -= pos2 - pos1 - 1
                                    vertexSet[x][y]['pos'][1] -= pos2 - pos1 - 1
                            vertexSet[j][k]['pos'][1] = pos1 + 1
            Ls = [0]
            L = 0
            for x in ori_data[i]['sents']:
                L += len(x)
                Ls.append(L)
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
            ignore_triple = set([])
            new_labels = []
            for label in labels:
                rel = label['r']
                assert(rel in rel2id)
                label['r'] = rel2id[label['r']]

                label['in_test+dev_set'] = False

                if not is_training:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            removed_facts.add((n1['name'].lower(), n2['name'].lower()))
                            removed_facts.add((n2['name'].lower(), n1['name'].lower()))
                else:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                                if (n1['name'].lower(), n2['name'].lower()) in removed_facts or (n2['name'].lower(), n1['name'].lower()) in removed_facts:
                                    label['in_test+dev_set'] = True

                if is_training:
                    if label['in_test+dev_set'] == False:
                        train_triple.add((label['h'], label['t']))
                        new_labels.append(label)
                    else:
                        ignore_triple.add((label['h'], label['t']))

            c2 += len(new_labels)
            if len(new_labels) > 0:
                item['labels'] = new_labels
                item['title'] = ori_data[i]['title']

                na_triple = []
                for j in range(len(vertexSet)):
                    for k in range(len(vertexSet)):
                        if (j != k):
                            if (j, k) not in train_triple and (j, k) not in ignore_triple:
                                na_triple.append((j, k))

                item['na_triple'] = na_triple
                item['Ls'] = Ls
                item['sents'] = ori_data[i]['sents']
                data.append(item)

        print(float(c2))
        print('\n')
        return data

    def init(data_file_name, rel2id, args, max_seq_length = 512, is_training = True, suffix=''):
        ori_data = json.load(open(data_file_name))

        data = save_data_format(ori_data, is_training)

        if suffix == 'train':
            print ('data_len:', len(data))

            print("Saving files")
            json.dump(data, open(os.path.join(output_dir, suffix + '_debug.json'), "w"))
            json.dump(rel2id, open(os.path.join(output_dir, 'rel2id_debug.json'), "w"))

            print("Finishing processing")

    init(dev_file_name, rel2id, args, max_seq_length = max_seq_length, is_training = False, suffix='dev')
    init(test_file_name, rel2id, args, max_seq_length = max_seq_length, is_training = False, suffix='test')
    init(train_annotated_file_name, rel2id, args, max_seq_length = max_seq_length, is_training = True, suffix='train')



if __name__ == '__main__':
    main()
