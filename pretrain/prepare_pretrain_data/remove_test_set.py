import sys
sys.path.append("..")
import numpy as np
import os
import json
import random
from utils import EntityMarker
from multiprocessing import Pool
import math
entityMarker = EntityMarker(bert_model='bert-base-cased')

def get_removed_entities():
    removed_facts = set([])
    for f_name in ['dev.json', 'test.json', 'dev_test.json']:
        with open(os.path.join('***task1_dataset***', f_name)) as f:
            docred_lines = json.load(f)
            for line in docred_lines:
                labels = line.get('labels', [])
                vertexSet = line['vertexSet']
                for label in labels:
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            removed_facts.add((n1['name'].lower(), n2['name'].lower()))

    for f_name in ['dev.txt', 'test.txt']:
        with open(os.path.join('***task2_dataset***', f_name)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                removed_facts.add((ins['h']['name'].lower(), ins['t']['name'].lower()))

    for f_name in ['test_wiki.json', 'test_pubmed.json']:
        with open(os.path.join('***task3_dataset***', f_name)) as f:
            all_lines = json.load(f)
            for key in all_lines:
                for sent in all_lines[key]:
                    removed_facts.add((sent['h'][0].lower(), sent['t'][0].lower()))

    for f_name in ['dev.txt', 'test.txt']:
        with open(os.path.join('***task4_dataset***', f_name)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                n1 = ins['h']
                n2 = ins['t']
                removed_facts.add((n1['name'].lower(), n2['name'].lower()))

    for f_name in ['dev.txt', 'test.txt']:
        with open(os.path.join('***task5_dataset***', f_name)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                n1 = ins['h']
                n2 = ins['t']
                removed_facts.add((n1['name'].lower(), n2['name'].lower()))
    # print(len(removed_facts))

    return removed_facts

def save_data_format(ori_data, file):
    removed_facts = get_removed_entities()
    data = []
    for i in range(len(ori_data)):
        item = {}
        vertexSet = ori_data[i]['vertexSet']
        item['vertexSet'] = vertexSet
        labels = ori_data[i].get('labels', [])

        train_triple = set([])
        ignore_triple = set([])
        new_labels = []

        for label in labels:
            label['in_test+dev_set'] = False

            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                        if (n1['name'].lower(), n2['name'].lower()) in removed_facts or (n2['name'].lower(), n1['name'].lower()) in removed_facts:
                            label['in_test+dev_set'] = True

            if label['in_test+dev_set'] == False:
                train_triple.add((label['h'], label['t']))
                new_labels.append(label)
            else:
                ignore_triple.add((label['h'], label['t']))

        if len(new_labels) > 0:
            item['labels'] = new_labels
            #item['sents'] = ori_data[i]['sents']
            item['tokenized_sents'] = []
            tokens = []
            for sent in ori_data[i]['sents']:
                subwords = list(map(entityMarker.tokenize, sent))
                item['tokenized_sents'].append(subwords)

            na_triple = []
            for j in range(len(item['vertexSet'])):
                for k in range(len(item['vertexSet'])):
                    if (j != k):
                        if (j, k) not in train_triple and (j, k) not in ignore_triple:
                            na_triple.append((j, k))

            item['na_triple'] = na_triple

            data.append(item)
    file_name = file.split('.')[0]
    json.dump(data, open(os.path.join('***path_to_save_your_data***/distant_doc_bert_base_cased/' + str(file_name) + '.json'), "w"))

    return data

def match_triple_proc(files):
    print(files)
    for file_num, file in enumerate(files):
        print(file.split('.')[0])
        ori_data = json.load(open('***loading_data_path***' + file, 'r'))
        save_data_format(ori_data, file)

threads = 5
all_files = sorted(os.listdir('***loading_data_path***'))
#all_files = ['corpus_wiki_04.json', 'corpus_wiki_05.json', 'corpus_wiki_06.json', 'corpus_wiki_07.json']
random.shuffle(all_files)
# match_triple_proc(all_files)

p = Pool(processes=threads)
for i in range(threads):
    seg = math.ceil(len(all_files) / threads)
    start = i * seg
    end = min(len(all_files), (i + 1) * seg)
    files = all_files[start: end]
    p.apply_async(match_triple_proc, args=(files,))
p.close()
p.join()
