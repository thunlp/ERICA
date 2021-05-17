from __future__ import print_function
import time
import numpy as np
import pandas as pd
import json
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
sns.set_theme(style="dark")
p_info = json.load(open('P_info.json', 'r'))
id2name = {v['id']: v['name'] for v in p_info}
pca_flag = 0
rel2id = {'P1376': 79, 'P607': 27, 'P136': 73, 'P137': 63, 'P131': 2, 'P527': 11, 'P1412': 38, 'P206': 33, 'P205': 77, 'P449': 52, 'P127': 34, 'P123': 49, 'P86': 66, 'P840': 85, 'P355': 72, 'P737': 93, 'P740': 84, 'P190': 94, 'P576': 71, 'P749': 68, 'P112': 65, 'P118': 40, 'P17': 1, 'P19': 14, 'P3373': 19, 'P6': 42, 'P276': 44, 'P1001': 24, 'P580': 62, 'P582': 83, 'P585': 64, 'P463': 18, 'P676': 87, 'P674': 46, 'P264': 10, 'P108': 43, 'P102': 17, 'P25': 81, 'P27': 3, 'P26': 26, 'P20': 37, 'P22': 30, 'Na': 0, 'P807': 95, 'P800': 51, 'P279': 78, 'P1336': 88, 'P577': 5, 'P570': 8, 'P571': 15, 'P178': 36, 'P179': 55, 'P272': 75, 'P170': 35, 'P171': 80, 'P172': 76, 'P175': 6, 'P176': 67, 'P39': 91, 'P30': 21, 'P31': 60, 'P36': 70, 'P37': 58, 'P35': 54, 'P400': 31, 'P403': 61, 'P361': 12, 'P364': 74, 'P569': 7, 'P710': 41, 'P1344': 32, 'P488': 82, 'P241': 59, 'P162': 57, 'P161': 9, 'P166': 47, 'P40': 20, 'P1441': 23, 'P156': 45, 'P155': 39, 'P150': 4, 'P551': 90, 'P706': 56, 'P159': 29, 'P495': 13, 'P58': 53, 'P194': 48, 'P54': 16, 'P57': 28, 'P50': 22, 'P1366': 86, 'P1365': 92, 'P937': 69, 'P140': 50, 'P69': 25, 'P1198': 96, 'P1056': 89}

def get_data(f1_name, f2_name):
    f1 = open(f1_name, 'r')
    embs = []
    for line in f1.readlines():
        emb = line.strip('\n').strip('\t').split('\t')
        emb = [float(xx) for xx in emb]
        embs.append(emb)
    f2 = open(f2_name, 'r')
    name2type = {}
    type2id = {'PER': 0, 'TIME': 1, 'LOC': 2, 'NUM': 3, 'ORG': 4, 'MISC': 5}
    id2type = {0: 'PER', 1: 'TIME', 2: 'LOC', 3: 'NUM', 4: 'ORG', 5: 'MISC'}
    ids = []
    names = []
    types = []
    flag = 0
    for line in f2.readlines():
        if flag == 0:
            flag = 1
            continue
        line = line.strip('\n').split('\t')
        name2type[line[0]] = line[1]
        names.append(line[0])
        types.append(line[1])
        ids.append(type2id[line[1]])
    X = np.array(embs)
    y = np.array(ids)
    feat_cols = [ names[i] for i in range(X.shape[1]) ]
    return X, y, feat_cols, ids, names, types

def plot(X, y, feat_cols, ids, names, types, ax):
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = types
    df['label'] = df['y']
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    if pca_flag:
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df[feat_cols].values)
        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1] 
        df['pca-three'] = pca_result[:,2]
    else:
        N = 10000
        df_subset = df.loc[rndperm[:N],:].copy()
        data_subset = df_subset[feat_cols].values
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data_subset)
        df_subset['pca-one'] = pca_result[:,0]
        df_subset['pca-two'] = pca_result[:,1] 
        df_subset['pca-three'] = pca_result[:,2]
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)
        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
    colors = [(0.86, 0.3712, 0.33999999999999997), (0.8287999999999999, 0.86, 0.33999999999999997), (0.33999999999999997, 0.86, 0.3712), (0.33999999999999997, 0.8287999999999999, 0.86), (0.3712, 0.33999999999999997, 0.86), (0.86, 0.33999999999999997, 0.8287999999999999)]
    color_dict = {'PER': colors[0], 'TIME': colors[1], 'LOC': colors[2], 'NUM': colors[3], 'ORG': colors[4], 'MISC': colors[5]}

    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    aa = sns.scatterplot(
        x="pca-one" if pca_flag else "tsne-2d-one", y="pca-two" if pca_flag else 'tsne-2d-two',
        hue="y",
        palette=color_dict,
        data=df.loc[rndperm,:] if pca_flag else df_subset,
        legend="full",
        alpha=0.5,
        ax = ax,
    )
    ax.set_xlabel(' ',fontsize=20)
    ax.set_ylabel(' ',fontsize=20)

    legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 6}, title='entity', fontsize='xx-small')
    legend.get_title().set_fontsize('8') #legend 'Title' fontsize
    plt.xticks([])
    plt.yticks([])
    return aa

def get_data_rel(f1_name, f2_name):
    f1 = open(f1_name, 'r')
    f2 = open(f2_name, 'r')
    embs = []
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    rel2num = {}
    flag = 0
    for line in lines2:
        if flag == 0:
            flag = 1
            continue
        t = line.strip('\n').split('\t')[1]
        if t not in rel2num:
            rel2num[t] = 0
        rel2num[t] += 1
    # retain = []
    # for i in range(10):
    #     retain.append(list(rel2num.keys())[i])
    retain = ['P17', 'P150', 'P569', 'P175', 'P131', 'P27', 'P57', 'P179', 'P1001', 'P176']
    
    for i, line in enumerate(lines1):
        if not lines2[i+1].strip('\n').split('\t')[1] in retain:
            continue
        emb = line.strip('\n').strip('\t').split('\t')
        emb = [float(xx) for xx in emb]
        embs.append(emb)

    name2type = {}
    type2id = rel2id
    id2type = {v: k for k,v in type2id.items()}
    ids = []
    names = []
    types = []
    flag = 0
    dd = []
    for line in lines2:
        if flag == 0:
            flag = 1
            continue
        if not line.strip('\n').split('\t')[1] in retain:
            continue
        line = line.strip('\n').split('\t')
        if line[1] not in dd:
            dd.append(line[1])
        name2type[line[0]] = line[1]
        names.append(line[0])
        types.append(line[1])
        ids.append(type2id[line[1]])

    X = np.array(embs)
    y = np.array(ids)

    feat_cols = [ names[0] + str(i) for i in range(X.shape[1]) ]
    return X, y, feat_cols, ids, names, types, len(dd)


def plot_rel(X, y, feat_cols, ids, names, types, len_dd, ax):
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = types
    df['label'] = df['y']
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))
    np.random.seed(43)
    rndperm = np.random.permutation(df.shape[0])

    if pca_flag:
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df[feat_cols].values)
        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1] 
        df['pca-three'] = pca_result[:,2]
    else:
        N = 10000
        df_subset = df.loc[rndperm[:N],:].copy()
        data_subset = df_subset[feat_cols].values
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data_subset)
        df_subset['pca-one'] = pca_result[:,0]
        df_subset['pca-two'] = pca_result[:,1] 
        df_subset['pca-three'] = pca_result[:,2]
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)
        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
    retain = ['P17', 'P150', 'P569', 'P175', 'P131', 'P27', 'P57', 'P179', 'P1001', 'P176']
    colors = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804), (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), (0.5764705882352941, 0.47058823529411764, 0.3764705882352941), (0.8549019607843137, 0.5450980392156862, 0.7647058823529411), (0.5490196078431373, 0.5490196078431373, 0.5490196078431373), (0.8, 0.7254901960784313, 0.4549019607843137), (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]
    color_dict = {'P17': colors[0], 'P150': colors[1], 'P569': colors[2], 'P175': colors[3], 
            'P131': colors[4], 'P27': colors[5], 'P57': colors[6], 'P179': colors[7], 'P1001': colors[8], 'P176': colors[9]}

    plt.xlim(-22, 22)
    plt.ylim(-22, 22)
    aa = sns.scatterplot(
        x="pca-one" if pca_flag else "tsne-2d-one", y="pca-two" if pca_flag else 'tsne-2d-two',
        hue="y",
        palette=color_dict,
        # palette=sns.color_palette("hls", len_dd),
        data=df.loc[rndperm,:] if pca_flag else df_subset,
        # legend=False,
        legend="brief",
        alpha=0.5,
        ax = ax,
    )
    ax.set_xlabel(' ',fontsize=20)
    ax.set_ylabel(' ',fontsize=20)
    current_palette = sns.color_palette()
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 6}, title='relation', fontsize='xx-small')
    legend.get_title().set_fontsize('8') #legend 'Title' fontsize
    plt.xticks([])
    plt.yticks([])
    return aa

plt.figure(figsize=(8,8))

with PdfPages("tsne.pdf") as pdf:
    plt.rc('font',family='Times New Roman')
    ax1 = plt.subplot(2, 2, 1)
    X, y, feat_cols, ids, names, types = get_data('embedding.tsv', 'name.tsv')
    aa = plot(X, y, feat_cols, ids, names, types, ax1)
    aa.set_xlabel('BERT: entity',fontsize=20)

    ax2 = plt.subplot(2, 2, 2)
    X, y, feat_cols, ids, names, types = get_data('embedding_dw.tsv', 'name_dw.tsv')
    aa = plot(X, y, feat_cols, ids, names, types, ax2)
    aa.set_xlabel('ERICA-BERT: entity',fontsize=20)

    ax3 = plt.subplot(2, 2, 3)
    X, y, feat_cols, ids, names, types, len_dd = get_data_rel('embedding_rel.tsv', 'name_rel.tsv')
    aa = plot_rel(X, y, feat_cols, ids, names, types, len_dd, ax3)
    aa.set_xlabel('BERT: relation',fontsize=20)

    ax4 = plt.subplot(2, 2, 4)
    X, y, feat_cols, ids, names, types, len_dd = get_data_rel('embedding_rel_dw.tsv', 'name_rel_dw.tsv')
    aa = plot_rel(X, y, feat_cols, ids, names, types, len_dd, ax4)
    aa.set_xlabel('ERICA-BERT: relation',fontsize=20)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.05, hspace=0.15)
    #plt.show()
    pdf.savefig(bbox_inches='tight')
    plt.close()