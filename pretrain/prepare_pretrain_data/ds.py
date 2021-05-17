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

debug = False

def get_entitys(doc, soup, sent_big):
    # print ("===============")
    # print ("sent = ", sent_big)
    match_entitys = set()
    # add anchors
    for c in soup.children:
        if c.name == 'a' and c.text in sent_big:
            link_name = urllib.parse.unquote(c['href']).lower()
            text_name = c.text.lower()
            # print("match", link_name, ' ;; ',text_name, ' ;; ', sent_big)
            # match link name first
            if link_name in name_to_Q:
                match_entitys.add((link_name, text_name))
            elif text_name in name_to_Q:
                match_entitys.add((text_name, text_name))

    # print ("begin = ", match_entitys)

    name_to_type = {}
    for ent in doc.ents:
        # add spacy entitys
        name = ent.text.lower()
        if name in name_to_Q:
            match_entitys.add((name, name))
            name_to_type[name] = ent.label_

    # print ("after = ", match_entitys)

    # remove names not in name_to_Q
    for name_pair in list(match_entitys):
        if (name_to_Q[name_pair[0]] not in effect_Q) or (len(name_pair[1]) < 3):
            match_entitys.remove(name_pair)

    match_entitys = get_longest(match_entitys)

    # print ("final = ", match_entitys)

    return match_entitys, name_to_type

def match_triple_proc(files):
    P_sents = defaultdict(list)
    thresh = 5000
    if debug:
        thresh = 2
    for file_num, file in enumerate(files):
        print(file, files)
        with open(base_path + file) as f:
            for line_num, line in enumerate((f.readlines())):
                if debug:
                    if line_num > 3:
                        break

                article = json.loads(line.strip())

                soup = bs(article['text'], "html.parser")
                org_content = soup.text

                try:
                    sentences = nltk.sent_tokenize(org_content)
                except:
                    print("error in tokenize = ", org_content)
                    continue

                for sent in sentences:
                    doc = nlp(sent)
                    words = [token.text.lower() for token in doc]
                    tokens = [token.text for token in doc]
                    if len(words) > 36:
                        continue

                    entitys, name_to_type = get_entitys(doc, soup, sent)
                    if len(entitys) < 2:
                        continue

                    # 确定vertex具体位置
                    entity_to_pos = defaultdict(list)
                    for link_name, text_name in entitys:
                        entity_name_tokens = WordPunctTokenizer().tokenize(text_name)
                        for start in range(len(words) - len(entity_name_tokens)):
                            if entity_name_tokens == words[start: start + len(entity_name_tokens)]:
                                pos = list(range(start, start + len(entity_name_tokens)))
                                entity_to_pos[(link_name, text_name)].append(pos)

                    # if len(entity_to_pos) != len(entitys):
                    # 	print("entity number not match")
                    # 	print(sent.split(), '\n', words, '\n', entitys, '\n', entity_to_pos)

                    if len(entity_to_pos) < 2:
                        continue

                    triples = []
                    min_P_id = 0
                    min_P_num = 10000000000
                    Ps = ['P0']
                    for h, t in itertools.permutations(entity_to_pos.keys(), 2):
                        h_Q = name_to_Q[h[0]]
                        t_Q = name_to_Q[t[0]]
                        entity_pair = (h_Q, t_Q)

                        if entity_pair in pair_Q_to_rel:
                            r_P = pair_Q_to_rel[entity_pair]
                            Ps.append(r_P)
                        else:
                            r_P = 'P0'
                        if len(P_sents[r_P]) < min_P_num:
                            min_P_num = len(P_sents[r_P])
                            min_P_id = r_P

                        if h[1] in name_to_type:
                            h_type = name_to_type[h[1]]
                        else:
                            h_type = "None"

                        h_info = {
                            'name': h[1],
                            'pos': entity_to_pos[h],
                            'id': h_Q,
                            'type': h_type
                        }

                        if t[1] in name_to_type:
                            t_type = name_to_type[t[1]]
                        else:
                            t_type = "None"
                        t_info = {
                            'name': t[1],
                            'pos': entity_to_pos[t],
                            'id': t_Q,
                            'type': t_type
                        }
                        triples.append({
                            'h': h_info,
                            't': t_info,
                            'r': r_P
                        })

                    sent_info = {
                        'tokens': tokens,
                        'triples': triples
                    }

                    if len(P_sents[min_P_id]) < thresh:
                        P_sents[min_P_id].append(sent_info)
                    elif random.random() < 0.5:
                        # random replace
                        r_P = random.choice(Ps)
                        P_sents[r_P][random.randint(0, thresh - 1)] = sent_info
                    else:
                        pass

                # print("--------------------")
                # print('org = ', org_content)
                # pprint(P_sents)
                # input()

        with open('distant/corpus_' + file + '.json', 'w') as fout:
            fout.write(json.dumps(P_sents))
    return []

def match_triple():
    print("matching triples")
    threads = 3
    all_files = sorted(os.listdir(base_path))
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
    with open('zero_shot_data/entity.txt') as f:
        for i, line in enumerate(f):
            if debug:
                if i > 10000000:
                    break
            entity_info = json.loads(line.strip())
            names = [entity_info['title']] + entity_info['aliases']
            for name in names:
                name_to_Q[name.lower()] = entity_info['id']
    print("entity num = ", i)

    print("loading triples")
    rels = set()
    global pair_Q_to_rel
    global effect_Q
    with open('zero_shot_data/triple.txt') as f:
        for i, line in enumerate(f):
            if debug:
                if i > 50000000:
                    break
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

# with open('corpus.json', 'w') as f:
# 	f.write('[\n')
# 	for i in range(len(corpus) - 1):
# 		f.write(json.dumps(corpus[i]) + ',\n')
# 	f.write(json.dumps(corpus[-1]) + '\n')
# 	f.write(']')
