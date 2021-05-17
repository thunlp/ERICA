from collections import defaultdict


def get_longest(e):
    new_e = []
    for worda in e:
        is_prefix = False
        for wordb in e:
            if worda[1] == wordb[1]:
                continue
            if worda[1] in wordb[1]:
                is_prefix = True
                break
        if not is_prefix:
            new_e.append(worda)
    return new_e
