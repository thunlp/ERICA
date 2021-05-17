import sys
sys.path.append("../../code")
import numpy as np
import os
import json
import random
from multiprocessing import Pool
import math

output_data = []

def convert_pos(pos):
    output_pos = []
    last_pos = None
    current_pos = []
    for p in pos:
        if last_pos != None:
            if p == last_pos + 1:
                current_pos.append(p)
            else:
                output_pos.append(current_pos)
                current_pos = [p]
        else:
            current_pos.append(p)
        last_pos = p
    output_pos.append(current_pos)
    new_output_pos = []
    for pos in output_pos:
        new_output_pos.append([pos[0], pos[-1] + 1])
    return new_output_pos

def check_void_mention(vertexSet):
    sent2pos = {}
    for vertexes in vertexSet:
        for vertex in vertexes:
            if vertex['sent_id'] not in sent2pos:
                sent2pos[vertex['sent_id']] = []
            sent2pos[vertex['sent_id']].append(vertex)
    for vertexes in vertexSet:
        for vertex in vertexes:
            for v2 in sent2pos[vertex['sent_id']]:
                if v2 != vertex:
                    s1 = vertex['pos'][0]
                    e1 = vertex['pos'][1] - 1
                    s2 = v2['pos'][0]
                    e2 = v2['pos'][1] - 1
                    if s2 <= e1 and e2 >= e1:
                        return True
                    if s2 <= s1 and e2 >= s1:
                        return True
    return False

def check_void_mention_2(sents, vertexSet):
    for vertexes in vertexSet:
        for vertex in vertexes:
            sent = sents[vertex['sent_id']]
            for word in sent[vertex['pos'][0]: vertex['pos'][1]]:
                if len(word) == 0:
                    return True
    return False

data_len = 100000

c1 = 0
c2 = 0
file_num = 0
for file in sorted(os.listdir('***load_data_path***')):
    print(file)
    data = json.load(open('***load_data_path***' + file, 'r'))
    for item in data:
        if len(item['labels']) <= 30 and len(item['vertexSet']) <= 20:
            sent_len = sum([len(x) for l in item['tokenized_sents'] for x in l])
            if sent_len <= 490 and sent_len >= 128:
                for v_1, vertexes in enumerate(item['vertexSet']):
                    new_vertexes = []
                    for vertex in vertexes:
                        if vertex not in new_vertexes:
                            new_vertexes.append(vertex)
                    item['vertexSet'][v_1] = new_vertexes

                for i_1 in range(len(item['vertexSet'])):
                    new_vertexSet = []
                    for i_2 in range(len(item['vertexSet'][i_1])):
                        for pos in convert_pos(item['vertexSet'][i_1][i_2]['pos']):
                            new_vertexSet.append({'pos': pos, 'type': item['vertexSet'][i_1][i_2]['type'], 'sent_id': item['vertexSet'][i_1][i_2]['sent_id'], 'name': item['vertexSet'][i_1][i_2]['name'], 'id': item['vertexSet'][i_1][i_2]['id']})

                    item['vertexSet'][i_1] = new_vertexSet

                if check_void_mention(item['vertexSet']):
                    c1 += 1
                    continue
                elif check_void_mention_2(item['tokenized_sents'], item['vertexSet']):
                    c1 += 1
                    continue
                else:
                    c2 += 1

                output_data.append(item)

        if len(output_data) >= data_len:
            random.shuffle(output_data)
            json.dump(output_data, open('***save_data_path***/train_distant_' + str(file_num) + '.json', 'w'))
            file_num += 1
            output_data = []
    print(c1)
    print(c2)
    print('\n')
