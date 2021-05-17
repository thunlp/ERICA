# coding=utf-8
from bs4 import BeautifulSoup as bs
import nltk
import sys
from collections import defaultdict
from nltk.tokenize import WordPunctTokenizer
import json
import multiprocessing
import spacy
from util import *
import os
import itertools
import urllib.parse
from nltk.tokenize import WordPunctTokenizer
import random
from pprint import pprint
from tqdm import tqdm
from multiprocessing import Pool
import math

effect_Q = set()
name_to_Q = {}
pair_Q_to_rel = {}
base_path = './AA/'
nlp = spacy.load('xx_ent_wiki_sm')

# rel2id = {'P1376': 79, 'P607': 27, 'P136': 73, 'P137': 63, 'P131': 2, 'P527': 11, 'P1412': 38, 'P206': 33, 'P205': 77, 'P449': 52, 'P127': 34, 'P123': 49, 'P86': 66, 'P840': 85, 'P355': 72, 'P737': 93, 'P740': 84, 'P190': 94, 'P576': 71, 'P749': 68, 'P112': 65, 'P118': 40, 'P17': 1, 'P19': 14, 'P3373': 19, 'P6': 42, 'P276': 44, 'P1001': 24, 'P580': 62, 'P582': 83, 'P585': 64, 'P463': 18, 'P676': 87, 'P674': 46, 'P264': 10, 'P108': 43, 'P102': 17, 'P25': 81, 'P27': 3, 'P26': 26, 'P20': 37, 'P22': 30, 'Na': 0, 'P807': 95, 'P800': 51, 'P279': 78, 'P1336': 88, 'P577': 5, 'P570': 8, 'P571': 15, 'P178': 36, 'P179': 55, 'P272': 75, 'P170': 35, 'P171': 80, 'P172': 76, 'P175': 6, 'P176': 67, 'P39': 91, 'P30': 21, 'P31': 60, 'P36': 70, 'P37': 58, 'P35': 54, 'P400': 31, 'P403': 61, 'P361': 12, 'P364': 74, 'P569': 7, 'P710': 41, 'P1344': 32, 'P488': 82, 'P241': 59, 'P162': 57, 'P161': 9, 'P166': 47, 'P40': 20, 'P1441': 23, 'P156': 45, 'P155': 39, 'P150': 4, 'P551': 90, 'P706': 56, 'P159': 29, 'P495': 13, 'P58': 53, 'P194': 48, 'P54': 16, 'P57': 28, 'P50': 22, 'P1366': 86, 'P1365': 92, 'P937': 69, 'P140': 50, 'P69': 25, 'P1198': 96, 'P1056': 89}

debug = False

def get_entitys(doc, soup, sent_big):
    match_entitys = set()
    for c in soup.children:
        if c.name == 'a' and c.text in sent_big:
            try:
                link_name = urllib.parse.unquote(c['href']).lower()
            except:
                link_name = 'no_such_a_link_name'
            text_name = c.text.lower()
            if link_name in name_to_Q:
                match_entitys.add((link_name, text_name))
            elif text_name in name_to_Q:
                match_entitys.add((text_name, text_name))

    text_name_list_1 = [e[1] for e in match_entitys]

    name_to_type = {}
    for ent in doc.ents:
        # add spacy entitys
        name = ent.text.lower()
        if name in text_name_list_1:
            name_to_type[name] = ent.label_
            continue
        if name in name_to_Q:
            match_entitys.add((name, name))
            name_to_type[name] = ent.label_

    # remove names not in name_to_Q
    for name_pair in list(match_entitys):
        if (name_to_Q[name_pair[0]] not in effect_Q) or (len(name_pair[1]) < 3):
            match_entitys.remove(name_pair)

    match_entitys = get_longest(match_entitys)

    return match_entitys, name_to_type

def match_triple_proc(files):
    print(files)
    for file_num, file in enumerate(files):
        print(file)
        distant_docs = []
        with open(base_path + file) as f:
            for line_num, line in enumerate((f.readlines())):
                if debug:
                    if line_num > 100:
                        break

                article = json.loads(line.strip())

                org_content = article['text']
                org_content = [x for x in org_content.split('\n') if len(x.split(' ')) < 520]

                for content_idx, content in enumerate(org_content):
                    try:
                        sentences = nltk.sent_tokenize(content)
                    except:
                        print("error in tokenize = ", content)
                        continue

                    distant_entity = {}
                    entity_num = 0
                    distant_label = []
                    drop_doc = False
                    distant_doc = {'vertexSet': [], 'labels': [], 'sents': []}
                    for sent_id, sent_ori in enumerate(sentences):
                        soup = bs(sent_ori, "html.parser")
                        sent = soup.text
                        sent_entity = []
                        doc = nlp(sent)
                        words = [token.text.lower() for token in doc]
                        tokens = [token.text for token in doc]
                        entitys, name_to_type = get_entitys(doc, soup, sent)

                        if len(entitys) > 0:
                            ff = 0
                            for idx_1, e1 in enumerate(entitys):
                                for idx_2, e2 in enumerate(entitys):
                                    if idx_1 == idx_2:
                                        continue
                                    if e1[1] == e2[1]:
                                        ff = 1
                            if ff == 1:
                                drop_doc = True
                            # print(sent_ori)
                            # print(entitys)
                            # print(name_to_type)
                            # print('\n')

                        #确定vertex具体位置
                        entity_to_pos = defaultdict(list)
                        for link_name, text_name in entitys:
                            entity_name_tokens = WordPunctTokenizer().tokenize(text_name)
                            for start in range(len(words) - len(entity_name_tokens)):
                                if entity_name_tokens == words[start: start + len(entity_name_tokens)]:
                                    pos = list(range(start, start + len(entity_name_tokens)))
                                    entity_to_pos[(link_name, text_name)].append(pos)

                        for h in entity_to_pos:
                            h_Q = name_to_Q[h[0]]

                            if h[1] in name_to_type:
                                h_type = name_to_type[h[1]]
                            else:
                                h_type = "None"

                            pos = entity_to_pos[h]
                            new_pos = []
                            for p in pos:
                                new_pos += p
                            pos = new_pos

                            h_info = {
                                'name': h[1],
                                'pos': new_pos,
                                'id': h_Q,
                                'type': h_type,
                                'sent_id': sent_id
                            }

                            if h_Q not in distant_entity:
                                distant_entity[h_Q] = entity_num
                                entity_num += 1
                                distant_doc['vertexSet'].append([])
                            distant_doc['vertexSet'][distant_entity[h_Q]].append(h_info)

                        distant_doc['sents'].append(tokens)

                    all_Q = [x[0]['id'] for x in distant_doc['vertexSet']]

                    for h_Q, t_Q in itertools.permutations(all_Q, 2):
                        entity_pair = (h_Q, t_Q)

                        if entity_pair in pair_Q_to_rel:
                            r_P = pair_Q_to_rel[entity_pair]
                        else:
                            r_P = 'P0'

                        if r_P == 'P0':
                            continue

                        if h_Q + '#' + t_Q not in distant_label:
                            distant_label.append(h_Q + '#' + t_Q)
                            distant_doc['labels'].append({
                                'h': distant_entity[h_Q],
                                't': distant_entity[t_Q],
                                'r': r_P,
                            })

                    if sum([len(x) for x in distant_doc['sents']]) > 510:
                        continue
                    if sum([len(x) for x in distant_doc['sents']]) <= 128:
                        continue
                    if len(distant_doc['labels']) < 4 or len(distant_doc['vertexSet']) < 4 or drop_doc:
                        continue

                    distant_docs.append(distant_doc)

        with open('distant_doc/distant_all/corpus_' + file + '.json', 'w') as fout:
            fout.write(json.dumps(distant_docs))
    return []


def match_triple():
    print("matching triples")
    threads = 5
    all_files = sorted(os.listdir(base_path))
    # all_files = ['wiki_05', 'wiki_06']

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


def get_pair_Q_to_rel():
    print("loading entitys")
    global name_to_Q
    if debug:
        with open('zero_shot_data/entity.txt') as f:
            for i, line in enumerate(f):
                if debug:
                    if i > 1000000:
                        break
                entity_info = json.loads(line.strip())
                names = [entity_info['title']] + entity_info['aliases']
                for name in names:
                    name_to_Q[name.lower()] = entity_info['id']
    else:
        name_to_Q_2 = json.load(open('all_name_to_Q.json', 'r'))
        for k in name_to_Q_2:
            name_to_Q[k.lower()] = name_to_Q_2[k]
        del name_to_Q_2

    print("entity num = ", len(name_to_Q.keys()))

    print("loading triples")
    rels = set()
    global pair_Q_to_rel
    global effect_Q
    if debug:
        with open('resources/triple.txt') as f:
            for i, line in enumerate(f):
                if debug:
                    if i > 5000000:
                        break
                try:
                    drop, h, r, t = line.strip().split('\t')
                except:
                    continue
                pair_Q_to_rel[(h, t)] = r
                rels.add(r)
                effect_Q.add(h)
                effect_Q.add(t)
    else:
        with open('all_triple.txt') as f:
            for i, line in enumerate(f):
                h, r, t = line.strip().split('\t')
                pair_Q_to_rel[(h, t)] = r
                rels.add(r)
                effect_Q.add(h)
                effect_Q.add(t)

    print("triple num = ", len(pair_Q_to_rel))
    print("relation num = ", len(rels))
    return pair_Q_to_rel


if __name__ == '__main__':
    get_pair_Q_to_rel()
    match_triple()
